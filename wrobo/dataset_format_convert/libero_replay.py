"""Replay LIBERO demonstrations into aligned, pre-action observations.

Simulation imports are lazy, so importing this module does not require LIBERO,
robosuite or a training framework. Each worker process creates and closes an
independent environment for each replay. No environment or GL object is cached
between calls.
"""

import math
import os
from pathlib import Path

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10")


def is_noop(action, prev_action=None, threshold=1e-4):
    """OpenVLA's motion-norm test, preserving changes in gripper command."""
    current = np.asarray(action)
    if current.shape != (7,):
        raise ValueError("LIBERO actions must have seven components")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("noop threshold must be finite and nonnegative")
    if not (np.linalg.norm(current[:-1]) < threshold):
        return False
    if prev_action is None:
        return True
    previous = np.asarray(prev_action)
    if previous.shape != (7,):
        raise ValueError("Previous LIBERO action must have seven components")
    return bool(current[-1] == previous[-1])


def create_environment(task, *, image_size=256):
    """Resolve a canonical BDDL by suite/name, never by converter task_id.

    The official benchmark uses exactly this suite/name BDDL layout. Resolving
    its files directly also avoids importing the torch-dependent benchmark
    package just to obtain a filename.
    """
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 1:
        raise ValueError("image_size must be a positive integer")
    suite, name = task.get("suite"), task.get("name")
    if suite not in SUITES:
        raise ValueError("Unknown official LIBERO suite: {!r}".format(suite))
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Task name must be an official LIBERO task filename stem")

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = Path(get_libero_path("bddl_files")) / suite / (name + ".bddl")
    if not bddl.is_file():
        raise FileNotFoundError("Official LIBERO BDDL not found for {}/{}: {}".format(suite, name, bddl))
    environment = task.get("parsed_environment", {})
    kwargs = environment.get("env_kwargs", {})
    controller = kwargs.get("controller_configs", {}).get("type", "OSC_POSE")
    if controller != "OSC_POSE":
        raise ValueError("LIBERO replay expects OSC_POSE commands, got {!r}".format(controller))
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), controller=controller,
        robots=kwargs.get("robots", ["Panda"]),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=image_size, camera_widths=image_size,
        camera_depths=False, camera_segmentations=None,
        control_freq=kwargs.get("control_freq", 20),
    )
    try:
        env.seed(319)
    except BaseException:
        env.close()
        raise
    return env


def _rotation_vector(quaternion):
    """robosuite's xyzw quaternion-to-axis-angle formula, without importing it."""
    quaternion = np.asarray(quaternion)
    if quaternion.shape != (4,):
        raise ValueError("robot0_eef_quat must have shape (4,)")
    w = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - w * w)
    if denominator == 0:
        return np.zeros(3, dtype=np.float64)
    return quaternion[:3] * (2.0 * np.arccos(w)) / denominator


def _observation_arrays(obs):
    position = np.array(obs["robot0_eef_pos"], copy=True)
    quaternion = np.array(obs["robot0_eef_quat"], copy=True)
    gripper = np.array(obs["robot0_gripper_qpos"], copy=True)
    orientation = _rotation_vector(quaternion)
    return {
        "obs/joint_states": np.array(obs["robot0_joint_pos"], copy=True),
        "obs/gripper_states": gripper,
        "obs/ee_pos": position,
        "obs/ee_ori": orientation,
        "obs/ee_states": np.concatenate([position, orientation]),
        "robot_states": np.concatenate([gripper, position, quaternion]),
        "obs/agentview_rgb": np.array(obs["agentview_image"], copy=True),
        "obs/eye_in_hand_rgb": np.array(obs["robot0_eye_in_hand_image"], copy=True),
    }


def replay_demo(task, demo, *, image_size=256, filter_noops=False, noop_threshold=1e-4):
    """Restore, settle and replay all kept actions, then check final success.

    A removed no-op is neither recorded nor executed, and its gripper command
    does not become the previous kept command. Observations and simulator states
    are recorded before each action, while rewards/dones come from that action's
    real environment step. Early success never truncates the replay.
    """
    actions = np.array(demo["actions"][:], copy=True)
    if actions.ndim != 2 or actions.shape[1] != 7 or actions.shape[0] < 1:
        raise ValueError("Source actions must have shape (T, 7) with T > 0")
    if actions.dtype.kind not in "fiu" or not np.isfinite(actions).all():
        raise ValueError("Source actions must contain finite numeric values")
    initial = np.array(demo["states"][0], copy=True)
    if initial.ndim != 1 or not np.isfinite(initial).all():
        raise ValueError("Source states[0] must be a finite flattened simulator state")
    if not math.isfinite(noop_threshold) or noop_threshold < 0:
        raise ValueError("noop_threshold must be finite and nonnegative")

    env = None
    try:
        env = create_environment(task, image_size=image_size)
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(10):
            obs, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

        settled_state = np.array(env.sim.get_state().flatten(), copy=True)
        template = _observation_arrays(obs)
        records = {name: [] for name in template}
        states, rewards, dones, source_indices = [], [], [], []
        previous = None
        num_noops = 0
        for index, action in enumerate(actions):
            if filter_noops and is_noop(action, previous, threshold=noop_threshold):
                num_noops += 1
                continue
            states.append(np.array(env.sim.get_state().flatten(), copy=True))
            for name, value in _observation_arrays(obs).items():
                records[name].append(value)
            source_indices.append(index)
            obs, reward, done, _ = env.step(action.tolist())
            rewards.append(reward)
            dones.append(done)
            previous = action

        arrays = {
            name: np.stack(values) if values else np.empty((0,) + template[name].shape, dtype=template[name].dtype)
            for name, values in records.items()
        }
        source_indices = np.asarray(source_indices, dtype=np.int64)
        arrays.update(
            actions=actions[source_indices],
            states=np.stack(states) if states else np.empty((0,) + settled_state.shape, dtype=settled_state.dtype),
            rewards=np.asarray(rewards) if rewards else np.empty(0, dtype=np.float64),
            dones=np.asarray(dones, dtype=np.uint8),
        )
        return {
            "arrays": arrays, "source_indices": source_indices,
            "success": bool(env.check_success()), "num_noops": num_noops,
            "model_file": env.sim.model.get_xml(),
            "init_state": states[0].copy() if states else settled_state,
        }
    finally:
        if env is not None:
            env.close()
