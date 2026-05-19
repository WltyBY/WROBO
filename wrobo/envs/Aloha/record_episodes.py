import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

from wrobo.envs.Aloha.utils.constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from wrobo.envs.Aloha.utils.sim_envs_ee import make_ee_sim_env
from wrobo.envs.Aloha.utils.sim_envs import make_sim_env, BOX_POSE
from wrobo.envs.Aloha.utils.scripted_policy import make_scripted_policy


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
        self.episode_len = collect_args.episode_len
        self.camera_names = collect_args.camera_names
        self.num_episodes = collect_args.num_episodes
        self.inject_noise = collect_args.inject_noise
        self.onscreen_render = collect_args.onscreen_render
        self.render_cam_name = "image_angle"
        self.skip_failure = collect_args.skip_failure
        self.random_seed = collect_args.seed

    def build_env(self):
        # build sim envs based on task_name
        self.env = make_sim_env(self.task_name, random_seed=self.random_seed)
        self.ee_env = make_ee_sim_env(self.task_name, random_seed=self.random_seed)

    def build_policy(self):
        # build policy based on task_name
        self.policy = make_scripted_policy(
            self.task_name, self.inject_noise, self.random_seed
        )

    def run(self):
        os.makedirs(self.dataset_dir, exist_ok=True)

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
            f"Saved Success Rate: {np.sum(self.success)} / {len(self.saved_episodes)} "
            f"(over {self.attempt} attempts)"
        )

    def _collect_single_episode(self, episode_idx: int) -> bool:
        # ---------- Stage 1: use EE space scripted policy ----------
        ts = self.ee_env.reset()
        episode_ee = [ts]
        self.policy.reset()

        if self.onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation[self.render_cam_name])
            plt.ion()

        for _ in range(self.episode_len):
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
        del episode_ee

        # ---------- Stage 2: Replay joint trajectories in sim_env and record data ----------
        BOX_POSE[0] = (
            subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
        )
        ts = self.env.reset()

        episode_replay = [ts]

        if self.onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation[self.render_cam_name])
            plt.ion()

        for t in range(len(joint_traj)):
            action = joint_traj[t]
            ts = self.env.step(action)
            episode_replay.append(ts)
            if self.onscreen_render:
                plt_img.set_data(ts.observation[self.render_cam_name])
                plt.pause(0.02)

        if self.onscreen_render:
            plt.close()

        # check success based on rewards obtained in ee_env
        sim_return = np.sum([t.reward for t in episode_replay[1:]])
        sim_max = np.max([t.reward for t in episode_replay[1:]])
        success_flag = sim_max == self.env.task.max_reward
        if success_flag:
            print(f"Replay phase: Success, return={sim_return}")
        else:
            print(f"Replay phase: Failed (max reward {sim_max})")

        # ---------- Save Data ----------
        # align timesteps: drop the last one to make sure obs/action length match max_timesteps
        # num_states = num_actions + 1, so we drop the last state to make them equal
        joint_traj = joint_traj[:-1]  # change length to episode_len
        episode_replay = episode_replay[:-1]  # length becomes episode_len + 1
        max_timesteps = len(joint_traj)  # i.e., episode_len

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
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_idx}")
        with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
            root.attrs["dataset"] = "Aloha"
            root.attrs["success"] = success_flag
            noise_flag = "with" if self.inject_noise else "without"
            root.attrs["comment"] = f"{self.task_name}_compressed_{noise_flag}_noise"
            root.attrs["sim"] = True
            root.attrs["seq_len"] = max_timesteps

            obs = root.create_group("observations")
            for cam_name in self.camera_names:
                img_src = data_dict[f"/observations/{cam_name}"]  # shape: (T, H, W, C)
                T = len(img_src)
                H, W, C = img_src[0].shape
                img_target = np.transpose(
                    np.stack(img_src, axis=0), (0, 3, 1, 2)
                )  # (T, C, H, W)
                obs.create_dataset(
                    cam_name,
                    data=img_target,
                    dtype="uint8",
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    chunks=(1, C, H, W),
                )

            obs.create_dataset(
                "proprio_state",
                data=data_dict["/observations/proprio_state"],
                dtype="float32",
                compression="gzip",
                compression_opts=1,
                shuffle=True,
                chunks=(50, data_dict["/observations/proprio_state"][0].shape[0]),
            )
            obs.create_dataset(
                "proprio_vel",
                data=data_dict["/observations/proprio_vel"],
                dtype="float32",
                compression="gzip",
                compression_opts=1,
                shuffle=True,
                chunks=(50, data_dict["/observations/proprio_vel"][0].shape[0]),
            )
            root.create_dataset(
                "action_abs",
                data=data_dict["/action_abs"],
                dtype="float32",
                compression="gzip",
                compression_opts=1,
                shuffle=True,
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
        "--episode_len", type=int, default=400, help="Maximum timesteps per episode"
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
        "--seed", type=int, default=319, help="Random seed for reproducibility"
    )

    return parser.parse_args()


if __name__ == "__main__":
    collect_args = get_args()
    collector = AlohaDataCollector(collect_args)
    collector.run()
