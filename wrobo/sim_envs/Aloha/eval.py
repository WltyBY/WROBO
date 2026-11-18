import os

os.environ["MUJOCO_GL"] = "egl"

import cv2
import torch
import random
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict
from torchvision import transforms
from time import time, sleep
from datetime import datetime
from collections import deque
from torch import autocast

from wrobo.sim_envs.Aloha.utils.constants import DT
from wrobo.sim_envs.Aloha.utils.sim_envs import make_sim_env, BOX_POSE
from wrobo.utils.file_operations import open_yaml, open_json
from wrobo.methods.registry import get_trainer_class, parse_run_name
from wrobo.training.dataset.instruction import UnifiedInstructionProvider
from wrobo.utils.load_model_weights import load_pretrained_weights
from wrobo.utils.normalization import normalization_parameters
from wrobo.training.utils.mics import dummy_context


class AlohaEvaluator:
    def __init__(self, eval_args):
        """
        Args:
            eval_args:
                - log_dir: dir of training logs
                - ckpt_name: filename of checkpoint to evaluate, e.g. "checkpoint_best.pth"
                - seed: random seed for evaluation
        """
        self.get_eval_args(eval_args)
        self.init_random()
        self.get_norm_stats()
        self.device = self.get_device()
        self.img_transform = self.get_img_transform()

        if torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16

        self._load_model()

        self.env = None
        self.env_max_reward = None
        self._init_env()

        self.eval_log_dir = os.path.join(
            self.log_dir,
            f"eval_{self.ckpt_name.replace('.pth', '')}_max_timestep_{self.max_timesteps}_temporal_agg_{self.temporal_agg}",
        )
        os.makedirs(self.eval_log_dir, exist_ok=True)
        self.log_file = os.path.join(self.eval_log_dir, "log.txt")
        with open(self.log_file, "w"):
            pass  # create or clear log file

    def get_eval_args(self, eval_args):
        self.log_dir = eval_args.log_dir
        self.config_file = eval_args.config_file
        self.task_name = eval_args.task_name
        self.num_rollouts = eval_args.num_rollouts
        self.ckpt_name = eval_args.ckpt_name
        self.max_timesteps = eval_args.max_timesteps
        self.onscreen_render = eval_args.onscreen_render
        self.args_gpu = eval_args.gpu
        self.temporal_agg = eval_args.temporal_agg
        self.random_seed = eval_args.seed
        self.instruction_arg = getattr(eval_args, "instruction", None)

        self.config_dict = open_yaml(
            os.path.join(self.log_dir, "Config_and_Trainer", self.config_file)
        )
        self.policy_config = self.config_dict["Policy"]

        data_key_lst = self.config_dict["Training_settings"]["data_keys"]
        self.img_keys = [
            i.replace("observations/", "")
            for i in data_key_lst
            if "image" in i and "observations/" in i
        ]
        # Language conditioning is driven by data_keys: if the model was trained with an
        # "instruction" channel, eval must feed it text tokens too.
        self.use_instruction = "instruction" in data_key_lst

        # Resolve which Trainer built this model from the run-name segment of the log
        # path (".../<method_name>/<TrainerName>__BS_..__.../...") so we reconstruct the
        # policy with the method that trained it, not a hardcoded one.
        self.trainer_cls = self._resolve_trainer_cls()

    def _resolve_trainer_cls(self):
        """Walk the log_dir path segments and return the Trainer class named by the
        run-name segment (the one whose leading token is a registered Trainer)."""
        segments = os.path.normpath(self.log_dir).split(os.sep)
        for seg in segments:
            trainer_name, _ = parse_run_name(seg)
            try:
                return get_trainer_class(trainer_name)
            except KeyError:
                continue
        raise ValueError(
            f"Could not find a registered Trainer name in log_dir '{self.log_dir}'. "
            "Expected a segment like 'ACTTrainer__BS_8__EPOCH_300__...'."
        )

    def init_random(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        os.environ["PYTHONHASHSEED"] = str(self.random_seed)

    def get_norm_stats(self):
        self.norm_stats = open_json(os.path.join(self.log_dir, "norm_stats.json"))

    def get_img_transform(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _preprocess_image(self, curr_image: np.ndarray, cam_name: str) -> torch.Tensor:
        """
        Preprocess a single raw camera frame (H, W, C) uint8 into a normalized
        (C, H, W) float tensor, mirroring EpisodicDataset.__getitem__.

        - If self.img_transform is set, apply it (ToTensor scales to [0, 1] and
          Normalize uses ImageNet stats).
        - Otherwise, fall back to /255 scaling + per-channel normalization with the
          dataset's image stats from norm_stats, exactly as the dataset's no-transform
          branch does. Stats are per-channel over the (C, H, W) layout, so we permute
          (H, W, C) -> (C, H, W) before normalizing.
        """
        if self.img_transform is not None:
            return self.img_transform(curr_image)  # (C, H, W)

        # (H, W, C) uint8 -> (C, H, W) float in [0, 1]
        obs = torch.from_numpy(np.asarray(curr_image)).permute(2, 0, 1).float() / 255.0
        if cam_name in self.norm_stats:
            mean, std = normalization_parameters(self.norm_stats[cam_name], "standard")
            mean = torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1)
            std = torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1)
            obs = (obs - mean) / std
        else:
            warnings.warn(
                f"No normalization stats for image key '{cam_name}', "
                "using only /255 scaling. Please check if this is intended.",
                UserWarning,
                stacklevel=2,
            )
        return obs

    def get_device(self):
        """
        args_gpu:
            None  -> use all visible GPUs
            []    -> CPU
            [0,2] -> only see GPU 0 and 2
        """
        # limit visible GPUs if specified
        if self.args_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.args_gpu))

        use_cuda = torch.cuda.is_available() and (self.args_gpu != [])
        device = torch.device("cuda:0" if use_cuda else "cpu")

        return device

    def _load_model(self) -> None:
        # Re-inject camera_names (same rule as training; the config copied to the log dir
        # predates ACT's runtime injection). Harmless for methods that ignore it.
        self.policy_config["camera_names"] = self.img_keys

        # Build the policy via the Trainer that trained it; aux carries method-specific
        # eval state (e.g. miniVLA's tokenizer), or None.
        self.policy, self.policy_aux = self.trainer_cls.build_policy(
            self.policy_config, self.log_dir
        )
        model_config = getattr(self.policy, "config", None)
        self.action_dim = (model_config.get("action_dim") if isinstance(model_config, dict)
                           else getattr(model_config, "action_dim", None))
        if self.action_dim is None:
            self.action_dim = self.policy_config.get(
                "action_dim", len(self.norm_stats["action_abs"]["mean"])
            )

        self.history_width = self.policy_config["history_width"]
        load_pretrained_weights(self.policy, os.path.join(self.log_dir, self.ckpt_name))

        # Language methods: prepare the instruction provider from the copied library so
        # eval can render an instruction per rollout (matching training's source).
        self.instruction_provider = None
        if self.use_instruction and self.instruction_arg is None:
            from wrobo.training.utils.dataset_snapshot import resolve_eval_instruction_library
            library_path = resolve_eval_instruction_library(
                os.path.join(self.log_dir, "Config_and_Trainer")
            )
            if library_path is not None and os.path.isfile(library_path):
                self.instruction_provider = UnifiedInstructionProvider(library_path)
            elif self.instruction_arg is None:
                raise FileNotFoundError(
                    "Model was trained with a language 'instruction' channel but no "
                    "instruction library was found in the log snapshot and no "
                    "--instruction was provided. Cannot build the eval instruction."
                )

        self.policy.to(self.device)

        print("Compiling network...")
        self.policy = torch.compile(self.policy)
        torch.compiler.reset()

        self.policy.eval()

    def _policy_action_chunk_size(self) -> int:
        """Action-chunk length from the config, method-agnostic.

        ACT/miniVLA keep it at the Policy top level; wVLA's single source of truth is
        action_expert.action_chunk_size. Prefer the top-level value when present, else
        fall back to the action_expert block.
        """
        if "action_chunk_size" in self.policy_config:
            return self.policy_config["action_chunk_size"]
        return self.policy_config["action_expert"]["action_chunk_size"]

    def _init_env(self) -> None:
        self.env = make_sim_env(self.task_name, self.random_seed, self.max_timesteps)
        self.env_max_reward = self.env.task.max_reward
        if self.max_timesteps is None:
            self.max_timesteps = self.env.task.max_timesteps

    def _pre_process(self, data: np.ndarray, key: str) -> np.ndarray:
        center, scale = normalization_parameters(
            self.norm_stats[key], self.trainer_cls.normalization_mode
        )
        center, scale = np.asarray(center, dtype=np.float32), np.asarray(scale, dtype=np.float32)
        return (np.asarray(data, dtype=np.float32) - center) / scale

    def _post_process(self, data: np.ndarray, key: str) -> np.ndarray:
        center, scale = normalization_parameters(
            self.norm_stats[key], self.trainer_cls.normalization_mode
        )
        center, scale = np.asarray(center, dtype=np.float32), np.asarray(scale, dtype=np.float32)
        return np.asarray(data, dtype=np.float32) * scale + center

    def _save_videos(self, video, dt, video_path=None):
        if isinstance(video, list):
            cam_names = list(video[0].keys())
            h, w, _ = video[0][cam_names[0]].shape
            w = w * len(cam_names)
            fps = round(1 / dt)
            out = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            for ts, image_dict in enumerate(video):
                images = []
                for cam_name in cam_names:
                    image = image_dict[cam_name]
                    image = image[:, :, [2, 1, 0]]  # RGB -> BGR
                    images.append(image)

                max_h = max(img.shape[0] for img in images)
                images_resized = []
                for img in images:
                    h = img.shape[0]
                    if h != max_h:
                        scale = max_h / h
                        img = cv2.resize(img, (int(img.shape[1] * scale), max_h))
                    images_resized.append(img)

                frame = np.concatenate(images_resized, axis=1)
                out.write(frame)
            out.release()
            print(f"Saved video to: {video_path}")
        elif isinstance(video, dict):
            cam_names = list(video.keys())
            all_cam_videos = []
            for cam_name in cam_names:
                all_cam_videos.append(video[cam_name])
            all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

            n_frames, h, w, _ = all_cam_videos.shape
            fps = int(1 / dt)
            out = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            for t in range(n_frames):
                image = all_cam_videos[t]
                image = image[:, :, [2, 1, 0]]  # swap B and R channel
                out.write(image)
            out.release()
            print(f"Saved video to: {video_path}")

    def _get_rollout_instruction(self, rollout_id: int) -> str:
        """Resolve the language instruction for a rollout.

        Priority: an explicit --instruction (free-form, used verbatim), else render a
        template from the copied instruction_library.json using the env task's current
        slots (color_order, randomized per rollout). Selection is stable per seed and rollout_id
        so repeated evals are reproducible. Tokenization is done later by the Trainer's
        build_policy_inputs, with the exact training tokenizer/vocab.
        """
        if self.instruction_arg is not None:
            return self.instruction_arg
        if self.instruction_provider is None:
            raise RuntimeError(
                "Language instruction requested but no provider/--instruction available."
            )
        # Get color order from environment and build parameters
        slots = [str(c) for c in self.env._task.color_order]
        params = {f"c{i}": s for i, s in enumerate(slots)}

        # Find the first template task in the library (should be task 0 for Aloha)
        task_id = None
        for tid, task in self.instruction_provider.task_map.items():
            if task["type"] == "template":
                task_id = tid
                break

        if task_id is None:
            raise ValueError("No template task found in instruction library")

        # Build task_ref with parameters
        task_ref = {"task_id": task_id, "params": params}

        # Use unified interface: sample_instruction with stable RNG for deterministic selection
        rng = np.random.default_rng(self.random_seed + rollout_id)
        return self.instruction_provider.sample_instruction(task_ref, rng=rng)

    @torch.inference_mode()
    def evaluate(
        self,
        onscreen_cam: str = "image_angle",
        save_episode: bool = True,
    ) -> Dict[str, float]:
        query_frequency = self._policy_action_chunk_size()
        if self.temporal_agg:
            num_queries = self._policy_action_chunk_size()
            query_frequency = 1

        episode_returns = []
        highest_rewards = []
        model_infer_time = []
        num_steps_lst = []

        for rollout_id in range(self.num_rollouts):
            rn_ = self.env._task.randomize_target_objs()
            BOX_POSE[0] = np.concatenate(rn_) if isinstance(rn_, (list, tuple)) else rn_

            # For language-conditioned models, resolve this rollout's instruction once
            # (the env's color_order was just randomized by randomize_target_objs).
            rollout_instruction = None
            if self.use_instruction:
                rollout_instruction = self._get_rollout_instruction(rollout_id)
                self.print_to_log_file(
                    f"Rollout {rollout_id:3d}: instruction = {rollout_instruction!r}"
                )

            ts = self.env.reset()
            proprio_states_buffer = deque(maxlen=self.history_width)
            num_cam = len(self.img_keys)
            img_buffers = [deque(maxlen=self.history_width) for _ in range(num_cam)]

            if self.onscreen_render:
                plt.ion()
                fig, ax = plt.subplots()
                img = ax.imshow(
                    self.env._physics.render(
                        height=480, width=640, camera_id=onscreen_cam
                    )
                )

            if self.temporal_agg:
                all_time_actions = torch.zeros(
                    [
                        self.max_timesteps,
                        self.max_timesteps + num_queries,
                        self.action_dim,
                    ],
                    device=self.device,
                )
                all_time_masks = torch.zeros(
                    [self.max_timesteps, self.max_timesteps + num_queries],
                    dtype=torch.bool,
                    device=self.device,
                )

            image_list = []
            qpos_list = []
            target_qpos_list = []
            rewards = []

            for t in range(self.max_timesteps):
                if self.onscreen_render:
                    img_data = self.env._physics.render(
                        height=480, width=640, camera_id=onscreen_cam
                    )
                    img.set_data(img_data)
                    plt.pause(DT)

                obs = ts.observation
                all_available_img_keys = [i for i in obs.keys() if "image" in i]
                image_list.append({k: obs[k] for k in all_available_img_keys})

                # Prepare single frame proprioceptive state in (1, 1, state_dim).
                # proprio_state is normalized with the same stats/formula as training
                # (EpisodicDataset normalizes non-image observations identically).
                proprio_state_np = np.array(obs["proprio_state"])
                # (1, state_dim)
                proprio_state_norm = self._pre_process(
                    proprio_state_np, "proprio_state"
                )[None]
                proprio_state_tensor = torch.from_numpy(proprio_state_norm).float()[
                    None
                ]  # (1, 1, state_dim) as (bs, his_width, state_dim)
                if len(proprio_states_buffer) == 0:
                    for _ in range(self.history_width):
                        proprio_states_buffer.append(proprio_state_tensor.clone())
                else:
                    proprio_states_buffer.append(proprio_state_tensor)

                # prepare single frame image in (1, num_cams, 1, C, H, W)
                for i, cam_name in enumerate(self.img_keys):
                    curr_image = ts.observation[cam_name]  # numpy (H_i, W_i, C)
                    curr_image = self._preprocess_image(
                        curr_image, cam_name
                    )  # torch (C, H_i, W_i)
                    curr_image = curr_image[
                        None, None
                    ]  # (1, 1, C, H_i, W_i)  batch=1, history=1

                    if len(img_buffers[i]) == 0:
                        for _ in range(self.history_width):
                            img_buffers[i].append(curr_image.clone())
                    else:
                        img_buffers[i].append(curr_image)

                # prepare model input by concatenating history
                propri_state_input = torch.cat(list(proprio_states_buffer), dim=1).to(
                    self.device
                )  # (1, history_width, state_dim)

                # Build image dict from buffers
                img_input = {}
                for i, cam_name in enumerate(self.img_keys):
                    cam_history = torch.cat(list(img_buffers[i]), dim=1).to(self.device)
                    img_input[cam_name] = cam_history

                if t % query_frequency == 0:
                    infer_start = time()
                    with (
                        autocast(self.device.type, dtype=self.amp_dtype, enabled=True)
                        if self.device.type == "cuda"
                        else dummy_context()
                    ):
                        # Shared observation bundle; the Trainer maps it to its policy's
                        # kwargs (ACT -> image+proprio_state; miniVLA -> image+state+text).
                        obs_bundle = {
                            "image": img_input,
                            "state": propri_state_input,
                        }
                        if self.use_instruction:
                            obs_bundle["instruction"] = rollout_instruction
                        policy_inputs = self.trainer_cls.build_policy_inputs(
                            obs_bundle, self.policy_aux, self.device
                        )
                        all_actions = self.policy(**policy_inputs)["pred"]
                        # (1, action_chunk_size, action_dim)
                    infer_duration = time() - infer_start
                    model_infer_time.append(infer_duration)
                    all_actions = all_actions.float()

                if self.temporal_agg:
                    all_time_actions[[t], t : t + num_queries] = all_actions
                    all_time_masks[[t], t : t + num_queries] = True

                    actions_for_curr_step = all_time_actions[
                        :, t
                    ]  # (max_timesteps, action_dim)
                    mask = all_time_masks[:, t]

                    if mask.any():
                        valid_actions = actions_for_curr_step[mask]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(valid_actions)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).to(self.device).unsqueeze(1)
                        )
                        raw_action = (valid_actions * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                    else:
                        raw_action = all_actions[:, 0]  # fallback
                else:
                    raw_action = all_actions[:, t % query_frequency]

                raw_action_np = raw_action.squeeze(0).cpu().numpy()
                action = self._post_process(raw_action_np, "action_abs")
                ts = self.env.step(action)

                qpos_list.append(proprio_state_np)
                target_qpos_list.append(action)
                rewards.append(ts.reward)

                # A rollout ends for one of two orthogonal reasons: task success
                # (reward hits env_max_reward) or the env's time limit being reached
                # (ts.last()). These mean different things, so check both. The success
                # check lets us stop early and record the true step count; the last()
                # check is a defensive guard so we never step past the env time_limit
                # and trigger dm_control's auto-reset (which returns a FIRST timestep
                # with reward=None and would poison the reward bookkeeping below). This
                # no longer relies on range(max_timesteps) staying aligned with the env
                # time_limit.
                if ts.reward is not None and ts.reward == self.env_max_reward:
                    self.print_to_log_file(
                        f"Rollout {rollout_id:3d}: Max reward reached at step {t}/{self.max_timesteps}, ending rollout."
                    )
                    break

                if ts.last():
                    self.print_to_log_file(
                        f"Rollout {rollout_id:3d}: Reached env time limit at step {t}/{self.max_timesteps}, ending rollout."
                    )
                    break

            if self.onscreen_render:
                plt.close(fig)
                plt.ioff()

            rewards_arr = np.array(rewards)
            episode_return = np.sum(rewards_arr)
            highest_reward = np.max(rewards_arr)

            episode_returns.append(episode_return)
            highest_rewards.append(highest_reward)
            num_steps_lst.append(len(rewards))

            self.print_to_log_file(
                f"Rollout {rollout_id:3d}: return = {episode_return:6.2f}, "
                f"highest_reward = {highest_reward}, max_reward = {self.env_max_reward}, "
                f"success = {highest_reward == self.env_max_reward}"
            )
            self.print_to_log_file("")

            if save_episode:
                video_path = os.path.join(
                    self.eval_log_dir,
                    f"video_rollout{rollout_id}.mp4",
                )
                self._save_videos(image_list, DT, video_path=video_path)

        success_mask = np.array(highest_rewards) == self.env_max_reward
        success_rate = np.mean(success_mask)
        avg_return = np.mean(episode_returns)
        num_steps_lst_success = np.array(num_steps_lst)[success_mask]
        num_steps_lst_fail = np.array(num_steps_lst)[~success_mask]

        avg_infer_time_sec = np.mean(model_infer_time)
        avg_infer_time_ms = avg_infer_time_sec * 1000.0
        infer_fps = 1.0 / avg_infer_time_sec
        gpu_name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else "CPU"
        )

        self.print_to_log_file(f"Success rate: {success_rate:.2%}")
        self.print_to_log_file(f"Average return: {avg_return:.2f}")
        self.print_to_log_file(
            f"Average return for successful episodes: {np.mean(np.array(episode_returns)[success_mask]):.2f}"
        )
        self.print_to_log_file(
            f"Average return for failed episodes: {np.mean(np.array(episode_returns)[~success_mask]):.2f}"
        )
        self.print_to_log_file(
            f"Number of steps: Mean: {np.mean(num_steps_lst):.2f}, Max: {np.max(num_steps_lst)}, Min: {np.min(num_steps_lst)}"
        )
        self.print_to_log_file(
            f"Steps for successful episodes: Mean: {np.mean(num_steps_lst_success):.2f}, Max: {np.max(num_steps_lst_success)}, Min: {np.min(num_steps_lst_success)}"
        )
        self.print_to_log_file(
            f"Steps for failed episodes: Mean: {np.mean(num_steps_lst_fail):.2f}, Max: {np.max(num_steps_lst_fail)}, Min: {np.min(num_steps_lst_fail)}"
        )
        self.print_to_log_file(
            f"Average Model inference time: {avg_infer_time_ms:.2f} ms"
        )
        self.print_to_log_file(f"Inference FPS: {infer_fps:.2f} Hz")
        self.print_to_log_file(f"Device: {gpu_name}")
        self.print_to_log_file("")

        reward_distribution = {}
        for r in range(self.env_max_reward + 1):
            count = sum(hr >= r for hr in highest_rewards)
            rate = count / self.num_rollouts
            self.print_to_log_file(
                f"Reward >= {r}: {count}/{self.num_rollouts} = {rate:.2%}"
            )
            reward_distribution[r] = {"count": count, "rate": rate}

        return {
            "success_rate": success_rate,
            "avg_return": avg_return,
            "reward_distribution": reward_distribution,
            "episode_returns": episode_returns,
            "highest_rewards": highest_rewards,
        }

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        timestamp = time()
        if add_timestamp:
            args = (f"{datetime.fromtimestamp(timestamp)}:", *args)

        for _ in range(5):
            try:
                with open(self.log_file, "a+") as f:
                    f.write(" ".join(map(str, args)) + "\n")
                break
            except IOError:
                sleep(0.5)

        if also_print_to_console:
            print(*args)


def parse_device(v):
    """Custom type: convert string to list[int] or None."""
    if v.lower() == "all":
        return None  # Do not set CUDA_VISIBLE_DEVICES, use all
    if v == "-1":
        return []  # Empty list -> CPU
    try:
        return [int(x) for x in v.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid format: {v}")


def build_eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--log_dir",
        type=str,
        help="Path of checkpoint directory",
        required=True,
    )
    p.add_argument(
        "--config_file",
        type=str,
        help="Name of config yaml, which contains setting of Policy, etc.",
        required=True,
    )
    p.add_argument(
        "--task_name",
        type=str,
        help="Task name to make envs, e.g. sim_transfer_cube",
        required=True,
    )
    p.add_argument(
        "--num_rollouts",
        type=int,
        default=50,
        help="Number of rollouts to evaluate",
    )
    p.add_argument(
        "--ckpt_name",
        type=str,
        default="checkpoint_best.pth",
        help="Name of checkpoint (final or best) to evaluate",
    )
    p.add_argument(
        "--max_timesteps",
        type=int,
        help="Maximum timesteps per rollout. If not specified, will use the episode length defined in the environment.",
    )
    p.add_argument(
        "--gpu",
        type=parse_device,
        default="0",
        help="CPU=-1; GPU ID=0; Multi-GPU=0,1,2...; All GPUs=all",
    )
    p.add_argument(
        "--onscreen_render",
        action="store_true",
        help="Whether to render the environment in real time during evaluation",
    )
    p.add_argument(
        "--temporal_agg",
        action="store_true",
        help="Whether to use temporal aggregation of actions for smoother execution",
    )
    p.add_argument("--seed", type=int, default=319, help="Random seed")
    p.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Optional free-form language instruction for language-conditioned models. "
        "If omitted, the instruction is rendered from the dataset templates using the "
        "env's per-rollout color order.",
    )
    return p


if __name__ == "__main__":
    parser = build_eval_parser()
    eval_args = parser.parse_args()
    evaluator = AlohaEvaluator(eval_args)
    eval_results = evaluator.evaluate()
