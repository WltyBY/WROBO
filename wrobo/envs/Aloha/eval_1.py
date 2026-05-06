import os
import pickle

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
from wrobo.utils.file_operations import open_yaml, open_json
from wrobo.methods.ACT.ACTTrainer import ACTTrainer
from wrobo.utils.load_model_weights import load_pretrained_weights

from Envs.act.utils import set_seed
from Envs.act.sim_env import BOX_POSE

def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')

def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = ts.observation['images'][cam_name].transpose(2, 0, 1)
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(319)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    onscreen_render = config["onscreen_render"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"

    config_dict = open_yaml(os.path.join(ckpt_dir, "Config_and_Trainer", "ACT.yaml"))
    policy_config = config_dict["Policy"]

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = ACTTrainer.get_networks(policy_config)
    load_pretrained_weights(policy, ckpt_path)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")

    stats = open_json(os.path.join(ckpt_dir, "norm_stats.json"))
    pre_process = lambda s_qpos: (
        s_qpos - np.asarray(stats["proprio_state"]["mean"])
    ) / np.asarray(stats["proprio_state"]["std"])
    post_process = lambda a: a * np.asarray(stats["action_abs"]["std"]) + np.asarray(
        stats["action_abs"]["mean"]
    )

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from Envs.act.sim_env import make_sim_env

        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config["action_chunk_size"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["action_chunk_size"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if "sim_transfer_cube" in task_name:
            from Envs.act.utils import sample_box_pose
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in task_name:
            from Envs.act.utils import sample_insertion_pose
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(
                env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            )
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(
                        height=480, width=640, camera_id=onscreen_cam
                    )
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])
                else:
                    image_list.append({"main": obs["image"]})
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(curr_image[None], qpos[None])["pred"]

                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
    
        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )

        if save_episode:
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, f"video{rollout_id}.mp4"),
            )

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)

    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))

    return success_rate, avg_return


if __name__ == "__main__":

    config = {
        'num_epochs': 400,
        'ckpt_dir': "./Logs/ACT_wrobo/BS_256_GPU_NUM_1_EPOCH_400_SEED_319_PRETRAINED_False/w_KL_10.0/fold_0",
        'episode_len': 400,
        'state_dim': 14,
        'lr': 1e-5,
        'policy_class': "ACT",
        'onscreen_render': False,
        'task_name': "sim_transfer_cube",
        'seed': 319,
        'temporal_agg': False,
        'camera_names': ["top"],
        'real_robot': False,
    }
    ckpt_names = [f"checkpoint_best.pth"]
    results = []
    for ckpt_name in ckpt_names:
        success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
        results.append([ckpt_name, success_rate, avg_return])

    for ckpt_name, success_rate, avg_return in results:
        print(f"{ckpt_name}: {success_rate=} {avg_return=}")
    print()
    exit()
