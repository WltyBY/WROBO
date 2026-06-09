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


def make_sim_env(task_name, random_seed=319, max_timestep=None):
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
    if "sim_transfer_stack_cube" in task_name:
        xml_path = os.path.join(XML_DIR, "bimanual_viperx_transfer_cube_stack.xml")
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeStackTask(random_seed=random_seed)
        env = control.Environment(
            physics,
            task,
            time_limit=(task.max_timesteps if max_timestep is None else max_timestep)
            * DT,
            control_timestep=DT,
            n_sub_steps=None,
            flat_observation=False,
        )
    elif "sim_transfer_cube" in task_name:
        xml_path = os.path.join(XML_DIR, f"bimanual_viperx_transfer_cube.xml")
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random_seed=random_seed)
        env = control.Environment(
            physics,
            task,
            time_limit=(task.max_timesteps if max_timestep is None else max_timestep)
            * DT,
            control_timestep=DT,
            n_sub_steps=None,
            flat_observation=False,
        )
    elif "sim_insertion" in task_name:
        xml_path = os.path.join(XML_DIR, f"bimanual_viperx_insertion.xml")
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random_seed=random_seed)
        env = control.Environment(
            physics,
            task,
            time_limit=(task.max_timesteps if max_timestep is None else max_timestep)
            * DT,
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
        obs["image_left_wrist"] = physics.render(
            height=240, width=320, camera_id="left_wrist"
        )
        obs["image_right_wrist"] = physics.render(
            height=240, width=320, camera_id="right_wrist"
        )

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.max_reward = 4
        self.max_timesteps = 400

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


class InsertionTask(BimanualViperXTask):
    def __init__(self, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.max_reward = 4
        self.max_timesteps = 400

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7 * 2 :] = BOX_POSE[0]  # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = (
            "red_peg",
            "vx300s_right/10_right_gripper_finger",
        ) in all_contact_pairs
        touch_left_gripper = (
            ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        )

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = (
            ("socket-1", "table") in all_contact_pairs
            or ("socket-2", "table") in all_contact_pairs
            or ("socket-3", "table") in all_contact_pairs
            or ("socket-4", "table") in all_contact_pairs
        )
        peg_touch_socket = (
            ("red_peg", "socket-1") in all_contact_pairs
            or ("red_peg", "socket-2") in all_contact_pairs
            or ("red_peg", "socket-3") in all_contact_pairs
            or ("red_peg", "socket-4") in all_contact_pairs
        )
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if (
            touch_left_gripper
            and touch_right_gripper
            and (not peg_touch_table)
            and (not socket_touch_table)
        ):  # grasp both
            reward = 2
        if (
            peg_touch_socket and (not peg_touch_table) and (not socket_touch_table)
        ):  # peg and socket touching
            reward = 3
        if pin_touched:  # successful insertion
            reward = 4
        return reward

    def randomize_target_objs(self):
        x_range = [0.1, 0.2]
        y_range = [0.4, 0.6]
        z_range = [0.05, 0.05]

        ranges = np.vstack([x_range, y_range, z_range])
        peg_position = self._random.uniform(ranges[:, 0], ranges[:, 1])

        peg_quat = np.array([1, 0, 0, 0])
        peg_pose = np.concatenate([peg_position, peg_quat])

        # Socket
        x_range = [-0.2, -0.1]
        y_range = [0.4, 0.6]
        z_range = [0.05, 0.05]

        ranges = np.vstack([x_range, y_range, z_range])
        socket_position = self._random.uniform(ranges[:, 0], ranges[:, 1])

        socket_quat = np.array([1, 0, 0, 0])
        socket_pose = np.concatenate([socket_position, socket_quat])

        return peg_pose, socket_pose


class TransferCubeStackTask(BimanualViperXTask):
    """
    The task is to pass and stack the three colored blocks in order.
    - Initially, three differently colored blocks are located in the right gripper area (random positions)
    - They must be passed in order (red -> green -> blue) from the right arm to the left arm
    - Each time the left arm catches a block, it is stacked on top of the previous block
    """

    def __init__(self, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.max_reward = 9
        self.max_timesteps = 2400
        self.color_order = ["red", "green", "blue"]
        self.current_color_idx = 0
        self.stacked_count = 0
        self.stack_target_positions = None  # compute in initialize_episode

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7 * 3 :] = BOX_POSE[0]  # three objects

        # reset task state
        self.current_color_idx = 0
        self.stacked_count = 0
        # target stacking positions (above the left arm side table, increasing height layer by layer)
        base_stack = np.array([-0.10, 0.5, 0.02])
        self.stack_target_positions = [
            base_stack,
            base_stack + np.array([0, 0, 0.04]),
            base_stack + np.array([0, 0, 0.08]),
        ]
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        return physics.data.qpos.copy()[-21:]

    def get_reward(self, physics):
        if self.current_color_idx >= 3:
            return self.max_reward

        target_color = self.color_order[self.current_color_idx]
        target_geom = f"{target_color}_box"

        # compute contacts
        all_contacts = []
        for i in range(physics.data.ncon):
            g1 = physics.model.id2name(physics.data.contact[i].geom1, "geom")
            g2 = physics.model.id2name(physics.data.contact[i].geom2, "geom")
            all_contacts.append((g1, g2))

        def touching(box, part):
            return (box, part) in all_contacts or (part, box) in all_contacts

        touch_left_gripper = touching(target_geom, "vx300s_left/10_left_gripper_finger")
        touch_right_gripper = touching(
            target_geom, "vx300s_right/10_right_gripper_finger"
        )
        touch_table = touching(target_geom, "table")

        # get current block position
        base_idx = -21 + self.current_color_idx * 7
        box_pos = physics.data.qpos[base_idx : base_idx + 3]
        target_pos = self.stack_target_positions[self.stacked_count]
        dist_to_target = np.linalg.norm(box_pos - target_pos)

        if (
            dist_to_target < 0.015
            and not touch_left_gripper
            and not touch_right_gripper
        ):
            # successfully stacked the current block, update progress
            self.stacked_count += 1
            self.current_color_idx += 1
            return self.stacked_count * 3

        progress = 0
        if touch_right_gripper and not touch_table:
            progress = 1  # successfully grasped by right arm
        if touch_left_gripper and not touch_right_gripper and not touch_table:
            progress = 2  # successfully passed to left arm

        return self.stacked_count * 3 + progress

    def randomize_target_objs(self):
        """
        Generate random initial poses for three blocks (right arm workspace, non-overlapping).
        Support two arrangement modes: horizontal (divide X into three segments) or vertical (divide Y into three segments), randomly chosen.
        """
        # right arm workspace (adjusted based on actual robot range and original transfer_cube task)
        x_range = [0.0, 0.2]
        y_range = [0.4, 0.6]
        z_fixed = 0.05  # fixed height

        offset = 0.04  # minimum distance between blocks to avoid overlap

        # Randomly select the arrangement pattern: 0 = horizontal (along X), 1 = vertical (along Y).
        mode = self._random.choice([0, 1])

        # Divide the corresponding dimension into three equal-length sub-intervals.
        if mode == 0:  # Horizontal arrangement: X is divided into three segments
            x_min, x_max = x_range
            x_segments = np.linspace(x_min, x_max, 4)  # [x0,x1,x2,x3]
            all_poses = []
            for i in range(3):
                x_low = x_segments[i] + offset
                x_high = x_segments[i + 1] - offset
                x = self._random.uniform(x_low, x_high)
                y = self._random.uniform(y_range[0], y_range[1])
                z = z_fixed
                quat = np.array([1, 0, 0, 0])
                pose = np.concatenate([[x, y, z], quat])  # 7 dim pose
                all_poses.append(pose)
        else:  # Vertical arrangement: Y is divided into three segments
            y_min, y_max = y_range
            y_segments = np.linspace(y_min, y_max, 4)
            all_poses = []
            for i in range(3):
                x = self._random.uniform(x_range[0], x_range[1])
                y_low = y_segments[i] + offset
                y_high = y_segments[i + 1] - offset
                y = self._random.uniform(y_low, y_high)
                z = z_fixed
                quat = np.array([1, 0, 0, 0])
                pose = np.concatenate([[x, y, z], quat])
                all_poses.append(pose)

        # Randomly shuffle the order of the three positions to decouple color from position.
        self._random.shuffle(all_poses)

        return np.concatenate(all_poses)  # 21 dim pose
