import os

os.environ["MUJOCO_GL"] = "egl"

import cv2
import torch
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict
from torchvision import transforms
from time import time, sleep
from datetime import datetime
from collections import deque
from torch import autocast

from wrobo.envs.Aloha.utils.constants import DT
from wrobo.envs.Aloha.utils.sim_envs import make_sim_env, BOX_POSE
from wrobo.utils.file_operations import open_yaml, open_json
from wrobo.methods.ACT.ACTTrainer import ACTTrainer
from wrobo.utils.load_model_weights import load_pretrained_weights
from wrobo.utils.others import dummy_context


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

        self.eval_log_dir = os.path.join(
            self.log_dir,
            f"evaluation_{self.ckpt_name.replace('.pth', '')}_temporal_agg_{self.temporal_agg}",
        )
        os.makedirs(self.eval_log_dir, exist_ok=True)
        self.log_file = os.path.join(self.eval_log_dir, "log.txt")
        with open(self.log_file, "w"):
            pass  # create or clear log file

        self._load_model()

        self.env = None
        self.env_max_reward = None
        self._init_env()

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

        self.config_dict = open_yaml(
            os.path.join(self.log_dir, "Config_and_Trainer", self.config_file)
        )
        self.policy_config = self.config_dict["Policy"]

        img_key_lst = self.config_dict["Training_settings"]["data_keys"]
        self.img_keys = [
            i.replace("observations/", "")
            for i in img_key_lst
            if "image" in i and "observations/" in i
        ]

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
        self.policy = ACTTrainer.get_policy(self.policy_config)
        self.state_dim = self.policy.config["proprio_dim"]
        self.history_width = self.policy.config["history_width"]
        load_pretrained_weights(self.policy, os.path.join(self.log_dir, self.ckpt_name))

        self.policy.to(self.device)

        print("Compiling network...")
        self.policy = torch.compile(self.policy)
        torch.compiler.reset()

        self.policy.eval()

    def _init_env(self) -> None:
        self.env = make_sim_env(self.task_name, self.random_seed)
        self.env_max_reward = self.env.task.max_reward

    def _pre_process(self, data: np.ndarray, key: str) -> np.ndarray:
        mean = np.asarray(self.norm_stats[key]["mean"])
        std = np.asarray(self.norm_stats[key]["std"])
        return (data - mean) / std

    def _post_process(self, data: np.ndarray, key: str) -> np.ndarray:
        mean = np.asarray(self.norm_stats[key]["mean"])
        std = np.asarray(self.norm_stats[key]["std"])
        return data * std + mean

    def _save_videos(self, video, dt, video_path=None):
        if isinstance(video, list):
            cam_names = list(video[0].keys())
            h, w, _ = video[0][cam_names[0]].shape
            w = w * len(cam_names)
            fps = int(1 / dt)
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

    @torch.inference_mode()
    def evaluate(
        self,
        onscreen_cam: str = "image_angle",
        save_episode: bool = True,
    ) -> Dict[str, float]:
        query_frequency = self.policy_config["action_chunk_size"]
        if self.temporal_agg:
            num_queries = self.policy_config["action_chunk_size"]
            query_frequency = 1

        episode_returns = []
        highest_rewards = []

        for rollout_id in range(self.num_rollouts):
            rn_ = self.env._task.randomize_target_objs()
            BOX_POSE[0] = np.concatenate(rn_) if isinstance(rn_, (list, tuple)) else rn_

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
                        self.state_dim,
                    ],
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

                # Prepare single frame proprioceptive state in (1, 1, state_dim)
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
                    curr_image = self.img_transform(curr_image)  # torch (C, H_i, W_i)
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
                img_input = []
                for i in range(num_cam):
                    # cat all history frames in the queue along the history dimension  → (1, history_width, C_i, H_i, W_i)
                    cam_history = torch.cat(list(img_buffers[i]), dim=1).to(self.device)
                    img_input.append(cam_history)

                if t % query_frequency == 0:
                    with (
                        autocast(self.device.type, dtype=self.amp_dtype, enabled=True)
                        if self.device.type == "cuda"
                        else dummy_context()
                    ):
                        all_actions = self.policy(
                            image=img_input, proprio_state=propri_state_input
                        )["pred"]
                        # (1, action_chunk_size, action_dim)
                    all_actions = all_actions.float()

                if self.temporal_agg:
                    all_time_actions[[t], t : t + num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[
                        :, t
                    ]  # (max_timesteps, action_dim)
                    mask = (actions_for_curr_step != 0).all(dim=1)
                    actions_for_curr_step = actions_for_curr_step[mask]
                    if len(actions_for_curr_step) > 0:
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).to(self.device).unsqueeze(1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
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

            if self.onscreen_render:
                plt.close(fig)
                plt.ioff()

            rewards_arr = np.array(rewards)
            episode_return = np.sum(rewards_arr[rewards_arr != None])
            highest_reward = np.max(rewards_arr)
            episode_returns.append(episode_return)
            highest_rewards.append(highest_reward)

            self.print_to_log_file(
                f"Rollout {rollout_id:2d}: return = {episode_return:6.2f}, "
                f"highest_reward = {highest_reward}, max_reward = {self.env_max_reward}, "
                f"success = {highest_reward == self.env_max_reward}"
            )

            if save_episode:
                video_path = os.path.join(
                    self.eval_log_dir,
                    f"video_rollout{rollout_id}.mp4",
                )
                self._save_videos(image_list, DT, video_path=video_path)

        success_rate = np.mean(np.array(highest_rewards) == self.env_max_reward)
        avg_return = np.mean(episode_returns)

        self.print_to_log_file("")
        self.print_to_log_file(f"Success rate: {success_rate:.2%}")
        self.print_to_log_file(f"Average return: {avg_return:.2f}")
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
        default=400,
        help="Maximum timesteps per rollout",
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
    return p


if __name__ == "__main__":
    parser = build_eval_parser()
    eval_args = parser.parse_args()
    evaluator = AlohaEvaluator(eval_args)
    eval_results = evaluator.evaluate()
