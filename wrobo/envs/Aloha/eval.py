import os

os.environ["MUJOCO_GL"] = "osmesa"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# os.environ["PYOPENGL_NO_ACCELERATE"] = "1"

import cv2
import torch
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Any, Tuple, List, Optional
from torchvision import transforms

from wrobo.envs.Aloha.utils.constants import DT
from wrobo.envs.Aloha.utils.sim_envs import make_sim_env, BOX_POSE
from wrobo.utils.file_operations import open_yaml, open_json
from wrobo.methods.ACT.ACTTrainer import ACTTrainer
from wrobo.utils.load_model_weights import load_pretrained_weights


class AlohaEvaluator:
    def __init__(self, eval_args: Dict[str, Any], verbose: bool = True):
        """
        Args:
            eval_args:
                - log_dir: dir of training logs
                - ckpt_name: filename of checkpoint to evaluate, e.g. "checkpoint_best.pth"
                - seed: random seed for evaluation
        """
        self.verbose = verbose

        self.get_eval_args(eval_args)
        self.init_random()
        self.get_norm_stats()
        self.device = self.get_device()
        self.img_transform = self.get_img_tranform()

        self.out_dir = os.path.join(self.log_dir, "evaluation")
        os.makedirs(self.out_dir, exist_ok=True)

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

    def get_img_tranform(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
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
        load_pretrained_weights(self.policy, os.path.join(self.log_dir, self.ckpt_name))

        self.policy.to(self.device)

        print("Compiling network...")
        self.policy = torch.compile(self.policy)

        self.policy.eval()

    def _init_env(self) -> None:
        self.env = make_sim_env(self.task_name, None)
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
                    image = image[:, :, [2, 1, 0]]  # swap B and R channel
                    images.append(image)
                images = np.concatenate(images, axis=1)
                out.write(images)
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
        """
        Args:
            onscreen_render: 是否实时渲染（覆盖 config 中的设置）

        Returns:
            Dict:
                - success_rate: 达到最大奖励（任务成功）的比例
                - avg_return: 平均累计奖励
                - reward_distribution: 各奖励阈值以上的次数和比例
        """

        query_frequency = self.policy_config.get("action_chunk_size", 1)

        episode_returns = []
        highest_rewards = []

        for rollout_id in range(self.num_rollouts):
            BOX_POSE[0] = self.env._task.randomize_target_objs()

            ts = self.env.reset()

            if self.onscreen_render:
                plt.ion()
                fig, ax = plt.subplots()
                img = ax.imshow(
                    self.env._physics.render(
                        height=480, width=640, camera_id=onscreen_cam
                    )
                )

            # 时序聚合缓冲区
            # if self.temporal_agg:
            #     all_time_actions = torch.zeros(
            #         [
            #             self.max_timesteps,
            #             self.max_timesteps + num_queries,
            #             self.state_dim,
            #         ],
            #         device=self.device,
            #     )

            # 存储可视化数据
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

                # Prepare tensor inputs for policy
                proprio_state_np = np.array(obs["proprio_state"])
                # (1, state_dim)
                proprio_state_norm = self._pre_process(
                    proprio_state_np, "proprio_state"
                )[None]
                proprio_state_tensor = (
                    torch.from_numpy(proprio_state_norm).float().to(self.device)[None]
                )  # (1, 1, state_dim) -> (bs, his_width, state_dim)

                # 获取当前图像
                curr_images = []
                for cam_name in self.img_keys:
                    curr_image = ts.observation[cam_name]
                    curr_image = self.img_transform(curr_image)  # (C, H, W)
                    curr_images.append(
                        curr_image[None]
                    )  # (1, C, H, W) -> (his_width, C, H, W)
                # (num_cams, 1, C, H, W) -> (num_cams, his_width, C, H, W)
                curr_image = torch.stack(curr_images, axis=0)
                curr_image = curr_image.to(self.device)[
                    None
                ]  # (bs, num_cams, his_width, C, H, W)

                if t % query_frequency == 0:
                    all_actions = self.policy(curr_image, proprio_state_tensor)[
                        "pred"
                    ]  # (1, action_chunk_size, action_dim)

                if False:
                    # if self.temporal_agg:
                    all_time_actions[[t], t : t + num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[
                        :, t
                    ]  # (max_timesteps, action_dim)
                    mask = (actions_for_curr_step != 0).all(dim=1)  # 非零且完整填充
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

                    # print("k=", t % query_frequency)
                    # print("first action:", all_actions[0,0][:3])
                    # print("used action:", all_actions[0, t % query_frequency][:3])

                raw_action_np = raw_action.squeeze(0).cpu().numpy()
                action = self._post_process(raw_action_np, "action_abs")
                ts = self.env.step(action)
                print("qpos:", proprio_state_np[:3])
                print("action:", action[:3])

                qpos_list.append(proprio_state_np)
                target_qpos_list.append(action)
                rewards.append(ts.reward)
                
            if self.onscreen_render:
                plt.close(fig)
                plt.ioff()

            # 计算当前 rollout 的累计奖励和最高奖励
            rewards_arr = np.array(rewards)
            episode_return = np.sum(rewards_arr[rewards_arr != None])
            highest_reward = np.max(rewards_arr)
            episode_returns.append(episode_return)
            highest_rewards.append(highest_reward)

            print(
                f"Rollout {rollout_id:2d}: return = {episode_return:6.2f}, "
                f"highest_reward = {highest_reward}, max_reward = {self.env_max_reward}, "
                f"success = {highest_reward == self.env_max_reward}"
            )

            if save_episode:
                video_path = os.path.join(
                    self.out_dir,
                    f"video_{self.ckpt_name.replace('.pth', '')}_rollout{rollout_id}.mp4",
                )
                self._save_videos(image_list, DT, video_path=video_path)

        # 统计成功率
        success_rate = np.mean(np.array(highest_rewards) == self.env_max_reward)
        avg_return = np.mean(episode_returns)

        # 详细报告各奖励阈值的达成比例
        print(f"\nSuccess rate: {success_rate:.2%}")
        print(f"Average return: {avg_return:.2f}\n")
        reward_distribution = {}
        for r in range(self.env_max_reward + 1):
            count = sum(hr >= r for hr in highest_rewards)
            rate = count / self.num_rollouts
            print(f"Reward >= {r}: {count}/{self.num_rollouts} = {rate:.2%}")
            reward_distribution[r] = {"count": count, "rate": rate}

        return {
            "success_rate": success_rate,
            "avg_return": avg_return,
            "reward_distribution": reward_distribution,
            "episode_returns": episode_returns,
            "highest_rewards": highest_rewards,
        }


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
        help="Name of checkpoint to evaluate",
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
    p.add_argument("--num_workers", type=int, default=12, help="Number of workers")
    p.add_argument("--seed", type=int, default=319, help="Random seed")
    return p


if __name__ == "__main__":
    parser = build_eval_parser()
    eval_args = parser.parse_args()
    evaluator = AlohaEvaluator(eval_args)
    eval_results = evaluator.evaluate()
