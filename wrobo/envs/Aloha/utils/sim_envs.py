import os
import collections

import numpy as np

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from wrobo.envs.Aloha.utils.constants import (
    XML_DIR,
    DT,
    PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN,
    PUPPET_GRIPPER_POSITION_NORMALIZE_FN,
    PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN,
    START_ARM_POSE,
)

BOX_POSE = [None]


def make_sim_env(task_name, random_seed=319):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if "sim_transfer_cube" in task_name:
        xml_path = os.path.join(XML_DIR, f"bimanual_viperx_transfer_cube.xml")
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random_seed=random_seed)
        env = control.Environment(
            physics,
            task,
            time_limit=20,
            control_timestep=DT,
            n_sub_steps=None,
            flat_observation=False,
        )
    else:
        raise NotImplementedError(f"Unknown task name: {task_name}")
    return env


class BimanualViperXTask(base.Task):
    def __init__(self, random_seed=None):
        super().__init__(random=random_seed)

    def before_step(self, action, physics):
        # joints
        left_arm_action = action[:6]
        right_arm_action = action[7 : 7 + 6]
        # grippers, [0, 1]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7 + 6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
            normalized_left_gripper_action
        )
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
            normalized_right_gripper_action
        )

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate(
            [
                left_arm_action,
                full_left_gripper_action,
                right_arm_action,
                full_right_gripper_action,
            ]
        )
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        # qpos_raw: [left_arm(6), left_gripper(2), right_arm(6), right_gripper(2)]
        # get raw qpos for joints and 0-1 mapped qpos for grippers
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate(
            [left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos]
        )

    @staticmethod
    def get_qvel(physics):
        # qvel_raw: [left_arm(6), left_gripper(2), right_arm(6), right_gripper(2)]
        # get raw qvel for joints and normalized qvel for grippers
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate(
            [left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel]
        )

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["proprio_state"] = self.get_qpos(physics)
        obs["proprio_vel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["image_top"] = physics.render(height=480, width=640, camera_id="top")
        obs["image_angle"] = physics.render(height=480, width=640, camera_id="angle")
        obs["image_front_close"] = physics.render(
            height=480, width=640, camera_id="front_close"
        )

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = (
            "red_box",
            "vx300s_left/10_left_gripper_finger",
        ) in all_contact_pairs
        touch_right_gripper = (
            "red_box",
            "vx300s_right/10_right_gripper_finger",
        ) in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward

    def randomize_target_objs(self):
        x_range = [0.0, 0.2]
        y_range = [0.4, 0.6]
        z_range = [0.05, 0.05]

        ranges = np.vstack([x_range, y_range, z_range])  # 3, 2
        cube_position = self._random.uniform(ranges[:, 0], ranges[:, 1])  # (x, y, z)

        cube_quat = np.array([1, 0, 0, 0])  # no rotation, (qw, qx, qy, qz)
        return np.concatenate([cube_position, cube_quat])
