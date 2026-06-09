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
    PUPPET_GRIPPER_POSITION_CLOSE,
    START_ARM_POSE,
)


def make_ee_sim_env(task_name, random_seed=319, max_timestep=None):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
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
        xml_path = os.path.join(XML_DIR, "bimanual_viperx_ee_transfer_cube_stack.xml")
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeStackEETask(random_seed=random_seed)
        env = control.Environment(
            physics,
            task,
            time_limit=(task.max_timesteps if max_timestep is None else max_timestep) * DT,
            control_timestep=DT,
            n_sub_steps=None,
            flat_observation=False,
        )
    elif "sim_transfer_cube" in task_name:
        xml_path = os.path.join(XML_DIR, f"bimanual_viperx_ee_transfer_cube.xml")
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeEETask(random_seed=random_seed)
        env = control.Environment(
            physics,
            task,
            time_limit=(task.max_timesteps if max_timestep is None else max_timestep) * DT,
            control_timestep=DT,
            n_sub_steps=None,
            flat_observation=False,
        )
    elif "sim_insertion" in task_name:
        xml_path = os.path.join(XML_DIR, f"bimanual_viperx_ee_insertion.xml")
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionEETask(random_seed=random_seed)
        env = control.Environment(
            physics,
            task,
            time_limit=(task.max_timesteps if max_timestep is None else max_timestep) * DT,
            control_timestep=DT,
            n_sub_steps=None,
            flat_observation=False,
        )
    else:
        raise NotImplementedError
    return env


class BimanualViperXEETask(base.Task):
    def __init__(self, random_seed=None):
        super().__init__(random=random_seed)
        self.max_timesteps=None

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(
            physics.data.ctrl,
            np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]),
        )

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:16] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(
            physics.data.mocap_pos[1], np.array([0.31718881, 0.49999888, 0.29525084])
        )
        np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array(
            [
                PUPPET_GRIPPER_POSITION_CLOSE,
                -PUPPET_GRIPPER_POSITION_CLOSE,
                PUPPET_GRIPPER_POSITION_CLOSE,
                -PUPPET_GRIPPER_POSITION_CLOSE,
            ]
        )
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
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
        # note: it is important to do .copy()
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

        # used in scripted policy to obtain starting pose
        obs["mocap_pose_left"] = np.concatenate(
            [physics.data.mocap_pos[0], physics.data.mocap_quat[0]]
        ).copy()
        obs["mocap_pose_right"] = np.concatenate(
            [physics.data.mocap_pos[1], physics.data.mocap_quat[1]]
        ).copy()

        # used when replaying joint trajectory
        obs["gripper_ctrl"] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.max_reward = 4
        self.max_timesteps = 400

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = self.randomize_target_objs()
        box_start_idx = physics.model.name2id("red_box_joint", "joint")
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

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


class InsertionEETask(BimanualViperXEETask):
    def __init__(self, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.max_reward = 4
        self.max_timesteps = 400

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize peg and socket position
        peg_pose, socket_pose = self.randomize_target_objs()
        id2index = (
            lambda j_id: 16 + (j_id - 16) * 7
        )  # first 16 is robot qpos, 7 is pose dim # hacky

        peg_start_id = physics.model.name2id("red_peg_joint", "joint")
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)
        # print(f"randomized cube position to {cube_position}")

        socket_start_id = physics.model.name2id("blue_socket_joint", "joint")
        socket_start_idx = id2index(socket_start_id)
        np.copyto(
            physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose
        )
        # print(f"randomized cube position to {cube_position}")

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


class TransferCubeStackEETask(BimanualViperXEETask):
    """
    The task is to pass and stack three colored blocks in order.
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
        self.initialize_robots(physics)

        # Generate random poses for the three blocks
        all_poses = self.randomize_target_objs()

        # Set poses for each block using joint_qposadr
        joint_names = ["red_box_joint", "green_box_joint", "blue_box_joint"]
        for i, jname in enumerate(joint_names):
            joint_id = physics.model.name2id(jname, "joint")
            start_idx = physics.model.jnt_qposadr[joint_id]
            pose = all_poses[i * 7 : (i + 1) * 7]
            np.copyto(physics.data.qpos[start_idx : start_idx + 7], pose)

        # Reset task state
        self.current_color_idx = 0
        self.stacked_count = 0
        # Stacking target positions (left arm side table)
        base_stack = np.array([-0.10, 0.5, 0.02])
        self.stack_target_positions = [
            base_stack,
            base_stack + np.array([0, 0, 0.04]),
            base_stack + np.array([0, 0, 0.08]),
        ]

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        # Return current poses of all three blocks (21 dim)
        # We need to extract them in order: red, green, blue
        joint_names = ["red_box_joint", "green_box_joint", "blue_box_joint"]
        poses = []
        for jname in joint_names:
            joint_id = physics.model.name2id(jname, "joint")
            start_idx = physics.model.jnt_qposadr[joint_id]
            poses.append(physics.data.qpos[start_idx : start_idx + 7].copy())
        return np.concatenate(poses)

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

        touch_right_gripper = touching(
            target_geom, "vx300s_right/10_right_gripper_finger"
        )
        touch_left_gripper = touching(target_geom, "vx300s_left/10_left_gripper_finger")
        touch_table = touching(target_geom, "table")

        # get current block position
        joint_id = physics.model.name2id(f"{target_color}_box_joint", "joint")
        start_idx = physics.model.jnt_qposadr[joint_id]
        box_pos = physics.data.qpos[start_idx : start_idx + 3]
        target_pos = self.stack_target_positions[self.stacked_count]
        dist_to_target = np.linalg.norm(box_pos - target_pos)

        # Check if block is successfully stacked (reached target and not held)
        if (
            dist_to_target < 0.015
            and not touch_right_gripper
            and not touch_left_gripper
        ):
            self.stacked_count += 1
            self.current_color_idx += 1
            return self.stacked_count * 3

        progress = 0
        if touch_right_gripper and not touch_table:
            progress = 1  # grasped by right arm
        if touch_left_gripper and not touch_right_gripper and not touch_table:
            progress = 2  # passed to left arm

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
