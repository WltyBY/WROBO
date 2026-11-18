# Copyright (c) 2023 Tony Z. Zhao
# Modifications by [Litingyu Wang] on [2026]: [New Task sim_transfer_stack_cube]

import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

from wrobo.sim_envs.Aloha.utils.constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, DT
from wrobo.sim_envs.Aloha.utils.sim_envs_ee import make_ee_sim_env
from wrobo.sim_envs.Aloha.utils.sim_envs import make_sim_env, BOX_POSE
from wrobo.sim_envs.Aloha.utils.scripted_policy import make_scripted_policy
from wrobo.sim_envs.Aloha.utils.instruction_templates import build_instruction_library
from wrobo.training.dataset.instruction import format_instruction_metadata
from wrobo.utils.file_operations import save_json
from wrobo.utils.image_codec import create_image_dataset

# Map env_state slot index -> color name (red=0, green=1, blue=2). Used to turn the
# integer stacking order exposed by the randomized task into human-readable colors.
STACK_IDX_TO_COLOR = {0: "red", 1: "green", 2: "blue"}


class AlohaDataCollector:

    def __init__(self, collect_args):

        self.get_collect_args(collect_args)
        np.random.seed(self.random_seed)

        self.build_env()
        self.build_policy()

        self.success = []
        self.saved_episodes = 0  # successful episodes saved so far
        self.attempt = 0  # total attempts (including failures) so far

    def get_collect_args(self, collect_args):
        self.dataset_dir = collect_args.dataset_dir
        self.task_name = collect_args.task_name
        self.max_timesteps = collect_args.max_timesteps
        self.camera_names = collect_args.camera_names
        self.num_episodes = collect_args.num_episodes
        self.inject_noise = collect_args.inject_noise
        self.onscreen_render = collect_args.onscreen_render
        self.render_cam_name = collect_args.render_cam_name
        self.skip_failure = collect_args.skip_failure
        self.random_seed = collect_args.seed
        # --compress is the master switch for on-disk compression of the .hdf5. When
        # off, everything is stored uncompressed (images as a dense uint8 array, and
        # action/proprio as plain gzip-free datasets). When on, action/proprio use the
        # existing gzip and images use --image_encoding (gzip/png/jpeg).
        self.compress = collect_args.compress
        self.image_encoding = collect_args.image_encoding
        self.jpeg_quality = collect_args.jpeg_quality

    def build_env(self):
        # build sim envs based on task_name. max_timesteps (a step count) drives the
        # env time_limit (in seconds, hence * DT) so that both the EE-phase rollout and
        # the replay env truncate after exactly max_timesteps steps. If not provided,
        # fall back to each task's built-in max_timesteps (max_timestep=None).
        self.env = make_sim_env(
            self.task_name,
            random_seed=self.random_seed,
            max_timestep=self.max_timesteps,
        )
        self.ee_env = make_ee_sim_env(
            self.task_name,
            random_seed=self.random_seed,
            max_timestep=self.max_timesteps,
        )
        # resolve the effective step count: explicit max_timesteps, else task default.
        if self.max_timesteps is None:
            self.max_timesteps = self.env.task.max_timesteps

    def build_policy(self):
        # build policy based on task_name
        self.policy = make_scripted_policy(
            self.task_name, self.inject_noise, self.random_seed
        )

    def run(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        # Create episodes/ subdirectory for the new format
        os.makedirs(os.path.join(self.dataset_dir, "episodes"), exist_ok=True)
        self._maybe_dump_instruction_library()

        while self.saved_episodes < self.num_episodes:
            print(
                f"\n=== Attempt {self.attempt}, target saved episode {self.saved_episodes} ==="
            )
            success_flag = self._collect_single_episode(episode_idx=self.saved_episodes)
            self.success.append(success_flag)

            if success_flag or not self.skip_failure:
                self.saved_episodes += 1
            self.attempt += 1

        print(f"\nData saved to: {self.dataset_dir}")
        print(
            f"Saved Success Rate: {np.sum(self.success)} / {self.saved_episodes} "
            f"(over {self.attempt} attempts)"
        )

    def _maybe_dump_instruction_library(self):
        """For tasks with a randomized stacking order, auto-save the language-instruction
        library next to the data so the dataset is self-contained (no manual copy).
        Per-trajectory HDF5 files store an ``all`` template-task reference with the
        concrete color order as parameters."""
        if not hasattr(self.env.task, "set_color_order"):
            return  # not the color-order stacking task; nothing to dump
        out_path = os.path.join(self.dataset_dir, "instruction_library.json")
        # sort_keys=False to preserve the curated template ordering.
        save_json(build_instruction_library(), out_path, sort_keys=False, indent=2)
        print(f"Saved instruction library to: {out_path}")

    def _collect_single_episode(self, episode_idx: int) -> bool:
        # ---------- Stage 1: use EE space scripted policy ----------
        ts = self.ee_env.reset()
        episode_ee = [ts]
        self.policy.reset()

        if self.onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation[self.render_cam_name])
            plt.ion()

        for _ in range(self.max_timesteps):
            action = self.policy(ts)
            ts = self.ee_env.step(action)
            episode_ee.append(ts)
            if self.onscreen_render:
                plt_img.set_data(ts.observation[self.render_cam_name])
                plt.pause(0.002)

        if self.onscreen_render:
            plt.close()

        # check success based on rewards obtained in ee_env
        ee_return = np.sum([t.reward for t in episode_ee[1:]])
        ee_max = np.max([t.reward for t in episode_ee[1:]])
        if ee_max == self.ee_env.task.max_reward:
            print(f"EE phase: Success, return={ee_return}")
        else:
            print(f"EE phase: Failed (max reward {ee_max})")
            return False

        joint_traj = [t.observation["proprio_state"] for t in episode_ee]
        gripper_ctrl_traj = [t.observation["gripper_ctrl"] for t in episode_ee]
        # replace gripper pose with gripper control
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl  # left hand gripper
            joint[6 + 7] = right_ctrl  # right hand gripper

        # save the initial object pose to ensure consistent object configuration in sim_env
        subtask_info = episode_ee[0].observation["env_state"].copy()
        # For the randomized-color-order stacking task, carry the per-episode stacking
        # order (chosen in the EE phase) over to the replay env so its reward matches.
        stack_color_order = episode_ee[0].observation.get("stack_color_order", None)
        if stack_color_order is not None:
            stack_color_order = np.asarray(stack_color_order).copy()
        del episode_ee

        # ---------- Stage 2: Replay joint trajectories in sim_env and record data ----------
        BOX_POSE[0] = (
            subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
        )
        # inject the stacking order before reset so the replay task's reward uses it.
        if stack_color_order is not None and hasattr(
            self.env.task, "set_color_order"
        ):
            idx_to_color = {0: "red", 1: "green", 2: "blue"}
            self.env.task.set_color_order(
                [idx_to_color[int(i)] for i in stack_color_order]
            )
        ts = self.env.reset()

        episode_replay = [ts]

        if self.onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation[self.render_cam_name])
            plt.ion()

        success_step = None
        for t in range(len(joint_traj)):
            action = joint_traj[t]
            ts = self.env.step(action)
            episode_replay.append(ts)
            if self.onscreen_render:
                plt_img.set_data(ts.observation[self.render_cam_name])
                plt.pause(0.02)

            if ts.reward is not None and ts.reward == self.env.task.max_reward:
                print(f"Replay phase: Max reward reached at step {t}, stopping early.")
                success_step = t
                break

            # Stop once the sim env hits its time limit (LAST timestep). joint_traj can
            # be one longer than the env's time_limit allows; stepping past it makes
            # dm_control auto-reset and return a FIRST timestep with reward=None, which
            # then poisons the reward reduction below.
            if ts.last():
                print(f"Replay phase: Reached env time limit at step {t}, stopping.")
                break

        if self.onscreen_render:
            plt.close()

        if success_step is not None:
            joint_traj = joint_traj[: success_step + 1]
            episode_replay = episode_replay[: success_step + 2]

        # check success based on rewards obtained in sim_env. Skip the initial reset
        # timestep (episode_replay[0], reward=None) and defensively drop any other None
        # rewards so the reduction never sees int + None.
        sim_rewards = [t.reward for t in episode_replay[1:] if t.reward is not None]
        sim_return = np.sum(sim_rewards) if sim_rewards else 0.0
        sim_max = np.max(sim_rewards) if sim_rewards else 0.0
        success_flag = sim_max == self.env.task.max_reward
        if success_flag:
            print(f"Replay phase: Success, return={sim_return}")
        else:
            print(f"Replay phase: Failed (max reward {sim_max})")

        # ---------- Save Data ----------
        # align timesteps: drop the last one to make sure obs/action length match
        # num_states = num_actions + 1, so we drop the last state to make them equal
        joint_traj = (
            joint_traj[:-1] if len(joint_traj) > 1 else joint_traj
        )  # -> max_timesteps
        episode_replay = (
            episode_replay[:-1] if len(episode_replay) > 1 else episode_replay
        )
        max_timesteps = len(joint_traj)  # actual saved length (<= self.max_timesteps)

        if self.skip_failure and not success_flag:
            print(f"Skipping failed episode (max reward {sim_max})")
            del episode_replay
            return success_flag

        data_dict = {
            "/observations/proprio_state": [],
            "/observations/proprio_vel": [],
            "/action_abs": [],
        }
        for cam_name in self.camera_names:
            data_dict[f"/observations/{cam_name}"] = []

        # append data in chronological order
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict["/observations/proprio_state"].append(
                ts.observation["proprio_state"]
            )
            data_dict["/observations/proprio_vel"].append(ts.observation["proprio_vel"])
            data_dict["/action_abs"].append(action)
            for cam_name in self.camera_names:
                data_dict[f"/observations/{cam_name}"].append(ts.observation[cam_name])

        t0 = time.time()
        # Resolve on-disk layout from the --compress master switch. compress off ->
        # everything uncompressed: images as a dense uint8 array ("raw"), non-image
        # datasets without gzip. compress on -> images use --image_encoding, non-image
        # datasets use gzip (as before).
        image_encoding = self.image_encoding if self.compress else "raw"
        array_compression = "gzip" if self.compress else None
        array_compression_opts = 1 if self.compress else None
        array_shuffle = self.compress

        dataset_path = os.path.join(self.dataset_dir, "episodes", f"episode_{episode_idx}")
        with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
            root.attrs["dataset"] = "Aloha"
            root.attrs["success"] = success_flag
            noise_flag = "with" if self.inject_noise else "without"
            root.attrs["comment"] = (
                f"Task_{self.task_name}_Img_{image_encoding}_{noise_flag}_action_noise"
            )
            root.attrs["sim"] = True
            root.attrs["seq_len"] = max_timesteps
            # The stacking order applies to the complete trajectory. Store one unified
            # string task reference; task 0 is the template task in the sidecar library.
            if stack_color_order is not None:
                color_names = [
                    STACK_IDX_TO_COLOR[int(i)] for i in stack_color_order
                ]
                params = {f"c{i}": color for i, color in enumerate(color_names)}
                root.attrs["instruction"] = format_instruction_metadata(
                    {"task_id": 0, "params": params}
                )

            obs = root.create_group("observations")
            for cam_name in self.camera_names:
                img_src = data_dict[f"/observations/{cam_name}"]  # list of (H, W, C)
                # (T, H, W, C) -> list of (C, H, W) RGB frames
                frames_chw = [np.transpose(im, (2, 0, 1)) for im in img_src]
                create_image_dataset(
                    obs,
                    cam_name,
                    frames_chw,
                    encoding=image_encoding,
                    jpeg_quality=self.jpeg_quality,
                )

            obs.create_dataset(
                "proprio_state",
                data=data_dict["/observations/proprio_state"],
                dtype="float32",
                compression=array_compression,
                compression_opts=array_compression_opts,
                shuffle=array_shuffle,
                chunks=(50, data_dict["/observations/proprio_state"][0].shape[0]),
            )
            obs.create_dataset(
                "proprio_vel",
                data=data_dict["/observations/proprio_vel"],
                dtype="float32",
                compression=array_compression,
                compression_opts=array_compression_opts,
                shuffle=array_shuffle,
                chunks=(50, data_dict["/observations/proprio_vel"][0].shape[0]),
            )
            root.create_dataset(
                "action_abs",
                data=data_dict["/action_abs"],
                dtype="float32",
                compression=array_compression,
                compression_opts=array_compression_opts,
                shuffle=array_shuffle,
                chunks=(50, data_dict["/action_abs"][0].shape[0]),
            )

        print(f"Saving episode {episode_idx}: {time.time() - t0:.1f} secs")
        del episode_replay
        return success_flag


def get_args():
    parser = argparse.ArgumentParser(description="Scripted data collection")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory to save collected episodes",
    )
    parser.add_argument(
        "--task_name", type=str, required=True, help="Name of the task to perform"
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=None,
        help="Maximum timesteps per episode. If not set, uses the task's built-in max_timesteps.",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["image_top"],
        help="List of camera names to record",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=50, help="Number of episodes to collect"
    )
    parser.add_argument(
        "--inject_noise",
        action="store_true",
        help="Inject noise into the scripted policy",
    )
    parser.add_argument(
        "--onscreen_render",
        action="store_true",
        help="Enable on-screen rendering during collection",
    )
    parser.add_argument(
        "--render_cam_name",
        type=str,
        default="image_angle",
        help="Camera name used for rendering",
    )
    parser.add_argument(
        "--skip_failure",
        action="store_true",
        help="Skip saving episodes that fail in the replay phase",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress the .hdf5 on disk. When set, action/proprio datasets use gzip "
        "and images use --image_encoding. When omitted, everything is stored "
        "uncompressed (images as a dense uint8 array, non-image datasets without gzip) "
        "and --image_encoding is ignored.",
    )
    parser.add_argument(
        "--image_encoding",
        type=str,
        default="png",
        choices=["gzip", "png", "jpeg"],
        help="Image storage layout when --compress is set (ignored otherwise): "
        "'gzip' (dense gzip'd uint8 array), 'png' (lossless per-frame byte strings), "
        "or 'jpeg' (lossy per-frame byte strings, smallest).",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality in [0, 100], only used when --compress and "
        "--image_encoding jpeg.",
    )
    parser.add_argument(
        "--seed", type=int, default=319, help="Random seed for reproducibility"
    )

    return parser.parse_args()


if __name__ == "__main__":
    collect_args = get_args()
    collector = AlohaDataCollector(collect_args)
    collector.run()
