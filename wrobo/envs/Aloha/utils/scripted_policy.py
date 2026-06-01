import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from wrobo.envs.Aloha.utils.constants import SIM_TASK_CONFIGS
from wrobo.envs.Aloha.utils.sim_envs_ee import make_ee_sim_env


def make_scripted_policy(task_name, inject_noise=False, random_seed=319):
    if "sim_transfer_cube_stack" in task_name:
        return StackTransferPolicy(inject_noise, random_seed)
    elif "sim_transfer_cube" in task_name:
        return PickAndTransferPolicy(inject_noise, random_seed)
    elif "sim_insertion" in task_name:
        return InsertionPolicy(inject_noise, random_seed)
    else:
        raise NotImplementedError(f"No scripted policy for {task_name}")


class BasePolicy:
    def __init__(self, inject_noise=False, random_seed=319):
        self.inject_noise = inject_noise
        self.rs = np.random.RandomState(random_seed)
        self.reset()

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        # Linear interpolation between two waypoints based on current time step t
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint["xyz"]
        curr_quat = curr_waypoint["quat"]
        curr_grip = curr_waypoint["gripper"]
        next_xyz = next_waypoint["xyz"]
        next_quat = next_waypoint["quat"]
        next_grip = next_waypoint["gripper"]
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]["t"] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]["t"] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(
            self.curr_left_waypoint, next_left_waypoint, self.step_count
        )
        right_xyz, right_quat, right_gripper = self.interpolate(
            self.curr_right_waypoint, next_right_waypoint, self.step_count
        )

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + self.rs.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + self.rs.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])

    def reset(self):
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation["mocap_pose_right"]
        init_mocap_pose_left = ts_first.observation["mocap_pose_left"]

        box_info = np.array(ts_first.observation["env_state"])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(
            axis=[0.0, 1.0, 0.0], degrees=-60
        )

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_left[:3],
                "quat": init_mocap_pose_left[3:],
                "gripper": 0,
            },  # sleep
            {
                "t": 100,
                "xyz": meet_xyz + np.array([-0.1, 0, -0.02]),
                "quat": meet_left_quat.elements,
                "gripper": 1,
            },  # approach meet position
            {
                "t": 260,
                "xyz": meet_xyz + np.array([0.02, 0, -0.02]),
                "quat": meet_left_quat.elements,
                "gripper": 1,
            },  # move to meet position
            {
                "t": 310,
                "xyz": meet_xyz + np.array([0.02, 0, -0.02]),
                "quat": meet_left_quat.elements,
                "gripper": 0,
            },  # close gripper
            {
                "t": 360,
                "xyz": meet_xyz + np.array([-0.1, 0, -0.02]),
                "quat": np.array([1, 0, 0, 0]),
                "gripper": 0,
            },  # move left
            {
                "t": 400,
                "xyz": meet_xyz + np.array([-0.1, 0, -0.02]),
                "quat": np.array([1, 0, 0, 0]),
                "gripper": 0,
            },  # stay
        ]

        self.right_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_right[:3],
                "quat": init_mocap_pose_right[3:],
                "gripper": 0,
            },  # sleep
            {
                "t": 90,
                "xyz": box_xyz + np.array([0, 0, 0.08]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # approach the cube
            {
                "t": 130,
                "xyz": box_xyz + np.array([0, 0, -0.015]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # go down
            {
                "t": 170,
                "xyz": box_xyz + np.array([0, 0, -0.015]),
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
            },  # close gripper
            {
                "t": 200,
                "xyz": meet_xyz + np.array([0.05, 0, 0]),
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
            },  # approach meet position
            {
                "t": 220,
                "xyz": meet_xyz,
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
            },  # move to meet position
            {
                "t": 310,
                "xyz": meet_xyz,
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # open gripper
            {
                "t": 360,
                "xyz": meet_xyz + np.array([0.1, 0, 0]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # move to right
            {
                "t": 400,
                "xyz": meet_xyz + np.array([0.1, 0, 0]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # stay
        ]


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation["mocap_pose_right"]
        init_mocap_pose_left = ts_first.observation["mocap_pose_left"]

        peg_info = np.array(ts_first.observation["env_state"])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation["env_state"])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(
            axis=[0.0, 1.0, 0.0], degrees=-60
        )

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(
            axis=[0.0, 1.0, 0.0], degrees=60
        )

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_left[:3],
                "quat": init_mocap_pose_left[3:],
                "gripper": 0,
            },  # sleep
            {
                "t": 120,
                "xyz": socket_xyz + np.array([0, 0, 0.08]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 1,
            },  # approach the cube
            {
                "t": 170,
                "xyz": socket_xyz + np.array([0, 0, -0.03]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 1,
            },  # go down
            {
                "t": 220,
                "xyz": socket_xyz + np.array([0, 0, -0.03]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 0,
            },  # close gripper
            {
                "t": 285,
                "xyz": meet_xyz + np.array([-0.1, 0, 0]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 0,
            },  # approach meet position
            {
                "t": 340,
                "xyz": meet_xyz + np.array([-0.05, 0, 0]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 0,
            },  # insertion
            {
                "t": 400,
                "xyz": meet_xyz + np.array([-0.05, 0, 0]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 0,
            },  # insertion
        ]

        self.right_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_right[:3],
                "quat": init_mocap_pose_right[3:],
                "gripper": 0,
            },  # sleep
            {
                "t": 120,
                "xyz": peg_xyz + np.array([0, 0, 0.08]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 1,
            },  # approach the cube
            {
                "t": 170,
                "xyz": peg_xyz + np.array([0, 0, -0.03]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 1,
            },  # go down
            {
                "t": 220,
                "xyz": peg_xyz + np.array([0, 0, -0.03]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 0,
            },  # close gripper
            {
                "t": 285,
                "xyz": meet_xyz + np.array([0.1, 0, lift_right]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 0,
            },  # approach meet position
            {
                "t": 340,
                "xyz": meet_xyz + np.array([0.05, 0, lift_right]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 0,
            },  # insertion
            {
                "t": 400,
                "xyz": meet_xyz + np.array([0.05, 0, lift_right]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 0,
            },  # insertion
        ]


class StackTransferPolicy(BasePolicy):
    """
    Scripted policy for the sim_transfer_cube_stack task.
    - Right arm picks each block (tilted), passes to meeting point.
    - Left arm receives in vertical pose (rotate 90° about X), closes gripper.
    - Left arm rotates to horizontal pose, moves to stack target, opens gripper.
    - For next block, left arm returns to meeting area with vertical pose.
    """

    def __init__(self, inject_noise=False, random_seed=319):
        super().__init__(inject_noise, random_seed)
        self.block_colors = ["red", "green", "blue"]

    @staticmethod
    def slerp(q1, q2, t):
        """Spherical linear interpolation (for quaternion arrays)."""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = np.dot(q1, q2)
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0
        return s1 * q1 + s2 * q2

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation["mocap_pose_right"]
        init_mocap_pose_left = ts_first.observation["mocap_pose_left"]
        env_state = ts_first.observation["env_state"]
        block_poses = [env_state[i * 7 : (i + 1) * 7] for i in range(3)]

        meet_xyz = np.array([0, 0.5, 0.25])
        base_stack = np.array([-0.10, 0.5, 0.02])

        right_pick_quat = (
            Quaternion(init_mocap_pose_right[3:])
            * Quaternion(axis=[0, 1, 0], degrees=-60)
        ).elements
        left_grasp_quat = Quaternion(axis=[1, 0, 0], degrees=90).elements
        left_release_quat = (
            Quaternion(axis=[1, 0, 0], degrees=90)
            * Quaternion(axis=[0, 0, 1], degrees=-90)
        ).elements

        all_left_wp = []
        all_right_wp = []
        cumulative_t = 0

        meet_pre = meet_xyz + np.array([-0.1, 0, -0.02])
        meet_center = meet_xyz + np.array([0.02, 0, -0.02])

        for idx in range(3):
            block_xyz = block_poses[idx][:3]
            stack_target = base_stack + np.array([0, 0, idx * 0.04])

            # ----- Right arm trajectory (strictly increasing timestamps) -----
            right_wp = [
                {
                    "t": 0,
                    "xyz": init_mocap_pose_right[:3],
                    "quat": init_mocap_pose_right[3:],
                    "gripper": 0,
                },
                {
                    "t": 90,
                    "xyz": block_xyz + [0, 0, 0.08],
                    "quat": right_pick_quat,
                    "gripper": 1,
                },
                {
                    "t": 130,
                    "xyz": block_xyz + [0, 0, -0.015],
                    "quat": right_pick_quat,
                    "gripper": 1,
                },
                {
                    "t": 170,
                    "xyz": block_xyz + [0, 0, -0.015],
                    "quat": right_pick_quat,
                    "gripper": 0,
                },
                {
                    "t": 200,
                    "xyz": meet_center + [0.03, 0, 0],
                    "quat": right_pick_quat,
                    "gripper": 0,
                },
                {"t": 220, "xyz": meet_center, "quat": right_pick_quat, "gripper": 0},
                {"t": 310, "xyz": meet_center, "quat": right_pick_quat, "gripper": 1},
                {
                    "t": 360,
                    "xyz": meet_center + [0.1, 0, 0],
                    "quat": right_pick_quat,
                    "gripper": 1,
                },
                {
                    "t": 420,
                    "xyz": meet_center + [0.1, 0, 0],
                    "quat": right_pick_quat,
                    "gripper": 1,
                },
            ]

            # ----- Left arm trajectory -----
            left_wp = []
            if idx == 0:
                left_wp.append(
                    {
                        "t": 0,
                        "xyz": init_mocap_pose_left[:3],
                        "quat": init_mocap_pose_left[3:],
                        "gripper": 1,
                    }
                )
                left_wp.append(
                    {"t": 60, "xyz": meet_pre, "quat": left_grasp_quat, "gripper": 1}
                )
            else:
                left_wp.append(
                    {"t": 0, "xyz": meet_pre, "quat": left_grasp_quat, "gripper": 1}
                )

            left_wp.append(
                {"t": 170, "xyz": meet_pre, "quat": left_grasp_quat, "gripper": 1}
            )
            left_wp.append(
                {
                    "t": 220,
                    "xyz": meet_center + [0.0, 0, -0.025],
                    "quat": left_grasp_quat,
                    "gripper": 1,
                }
            )
            left_wp.append(
                {
                    "t": 310,
                    "xyz": meet_center + [0.01, 0, -0.025],
                    "quat": left_grasp_quat,
                    "gripper": 0,
                }
            )

            # Simultaneous movement and rotation (t=310~470)
            start_pos = meet_center
            end_pos = stack_target + [0, 0, 0.05]
            start_quat = left_grasp_quat
            end_quat = left_release_quat
            left_wp.append(
                {
                    "t": 350,
                    "xyz": start_pos + 0.25 * (end_pos - start_pos),
                    "quat": self.slerp(start_quat, end_quat, 0.25),
                    "gripper": 0,
                }
            )
            left_wp.append(
                {
                    "t": 390,
                    "xyz": start_pos + 0.50 * (end_pos - start_pos),
                    "quat": self.slerp(start_quat, end_quat, 0.50),
                    "gripper": 0,
                }
            )
            left_wp.append(
                {
                    "t": 430,
                    "xyz": start_pos + 0.75 * (end_pos - start_pos),
                    "quat": self.slerp(start_quat, end_quat, 0.75),
                    "gripper": 0,
                }
            )
            left_wp.append({"t": 470, "xyz": end_pos, "quat": end_quat, "gripper": 0})

            # Descend and release
            left_wp.append(
                {
                    "t": 480,
                    "xyz": stack_target + [0, 0, 0.02],
                    "quat": left_release_quat,
                    "gripper": 0,
                }
            )
            left_wp.append(
                {"t": 510, "xyz": stack_target, "quat": left_release_quat, "gripper": 0}
            )
            left_wp.append(
                {"t": 530, "xyz": stack_target, "quat": left_release_quat, "gripper": 1}
            )

            if idx < 2:
                # Lift and return for next block
                left_wp.append(
                    {
                        "t": 560,
                        "xyz": stack_target + [0, 0, 0.08],
                        "quat": left_release_quat,
                        "gripper": 1,
                    }
                )
                start_pos_back = stack_target + [0, 0, 0.08]
                end_pos_back = meet_pre
                start_quat_back = left_release_quat
                end_quat_back = left_grasp_quat
                left_wp.append(
                    {
                        "t": 590,
                        "xyz": start_pos_back + 0.25 * (end_pos_back - start_pos_back),
                        "quat": self.slerp(start_quat_back, end_quat_back, 0.25),
                        "gripper": 1,
                    }
                )
                left_wp.append(
                    {
                        "t": 620,
                        "xyz": start_pos_back + 0.50 * (end_pos_back - start_pos_back),
                        "quat": self.slerp(start_quat_back, end_quat_back, 0.50),
                        "gripper": 1,
                    }
                )
                left_wp.append(
                    {
                        "t": 650,
                        "xyz": start_pos_back + 0.75 * (end_pos_back - start_pos_back),
                        "quat": self.slerp(start_quat_back, end_quat_back, 0.75),
                        "gripper": 1,
                    }
                )
                left_wp.append(
                    {"t": 680, "xyz": end_pos_back, "quat": end_quat_back, "gripper": 1}
                )
                left_wp.append(
                    {"t": 700, "xyz": meet_pre, "quat": left_grasp_quat, "gripper": 1}
                )
            else:
                # FIX: For the last block, add a very long wait to avoid running out of waypoints
                last_left_t = left_wp[-1]["t"]
                left_wp.append(
                    {
                        "t": last_left_t + 2000,
                        "xyz": stack_target,
                        "quat": left_release_quat,
                        "gripper": 1,
                    }
                )
                last_right_t = right_wp[-1]["t"]
                right_wp.append(
                    {
                        "t": last_right_t + 2000,
                        "xyz": right_wp[-1]["xyz"],
                        "quat": right_wp[-1]["quat"],
                        "gripper": right_wp[-1]["gripper"],
                    }
                )

            # Apply time offset
            for wp in right_wp:
                wp["t"] += cumulative_t
            for wp in left_wp:
                wp["t"] += cumulative_t

            all_right_wp.extend(right_wp)
            all_left_wp.extend(left_wp)

            # FIX: Update cumulative_t to the last waypoint time + 1 (strictly increasing across blocks)
            cumulative_t = left_wp[-1]["t"] + 1

        self.left_trajectory = all_left_wp
        self.right_trajectory = all_right_wp
        self.curr_left_waypoint = self.left_trajectory[0]
        self.curr_right_waypoint = self.right_trajectory[0]


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    if "sim_transfer_cube_stack" in task_name:
        policy = StackTransferPolicy(inject_noise)
        env = make_ee_sim_env("sim_transfer_cube_stack")
    elif "sim_transfer_cube" in task_name:
        policy = PickAndTransferPolicy(inject_noise)
        env = make_ee_sim_env("sim_transfer_cube")
    elif "sim_insertion" in task_name:
        policy = InsertionPolicy(inject_noise)
        env = make_ee_sim_env("sim_insertion")
    else:
        raise NotImplementedError
    episode_len = env.task.episode_len
    print("Task class:", type(env.task))

    for episode_idx in range(2):
        print("Episode", episode_idx)
        ts = env.reset()
        policy.reset()

        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation["image_angle"])
            plt.ion()

        for step in range(episode_len):
            print(step)
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation["image_angle"])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == "__main__":
    # test_task_name = "sim_transfer_cube_scripted"
    test_task_name = "sim_transfer_cube_stack_scripted"
    test_policy(test_task_name)
