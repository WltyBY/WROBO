"""Deploy wrobo policies with independent HDF5 and camera MP4 recording."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
import logging
import math
from numbers import Real
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np

from wrobo.methods.registry import METHOD_TO_TRAINER

from .teleoperate import (
    TeleoperationOptions, _check_rgb_configs, _close_preview, _degree_settings,
    _numeric_keys, _preview_camera_configs, _preview_image_keys, _preview_metadata,
    _runtime_backend,
)
from .utils.device import parse_device
from .utils.setup_backend import _managed_backend
from .utils.setup_config import HardwareSetupConfig, load_setup_config


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DeploymentOptions:
    control_fps: float = 30
    mode: str = "sync"
    chunk_steps: int | None = None
    prefetch_threshold: float = 0.5
    new_action_weight: float = 0.7
    new_action_fuse: str = "aggregate"
    instruction: str | None = None
    camera_map: dict[str, str] = field(default_factory=dict)
    record: bool = False
    save_video: bool = False
    preview: bool = False
    keyboard_debug: bool = False
    reset_on_touch: bool = False
    reset_arm_load_threshold: float = 100.0
    reset_gripper_load_threshold: float = 100.0
    reset_load_duration: float = 0.15
    reset_position_threshold: float = 20.0
    dataset_dir: str | Path | None = None
    video_dir: str | Path | None = None
    img_encoding: str = "jpeg"
    jpeg_quality: int = 95
    writer_queue_size: int = 30
    writer_batch_size: int = 8
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.control_fps, bool) or not isinstance(self.control_fps, Real) or not math.isfinite(self.control_fps) or self.control_fps <= 0:
            raise ValueError("control_fps must be a positive finite number.")
        if not isinstance(self.save_video, bool):
            raise ValueError("save_video must be a boolean.")
        if self.save_video:
            if not isinstance(self.video_dir, (str, Path)) or not str(self.video_dir).strip():
                raise ValueError("--save-video requires a nonempty --video-dir.")
            self.video_dir = Path(self.video_dir).expanduser().resolve()
            if self.video_dir.exists() and not self.video_dir.is_dir():
                raise ValueError(f"Video destination is not a directory: {self.video_dir}")
        elif self.video_dir is not None:
            raise ValueError("--video-dir requires --save-video.")
        # Recording has the same destination, image codec and metadata contract
        # as the already validated teleoperation entry point.
        recording = TeleoperationOptions(
            fps=self.control_fps, record=self.record, preview=self.preview,
            dataset_dir=self.dataset_dir, img_encoding=self.img_encoding,
            jpeg_quality=self.jpeg_quality, writer_queue_size=self.writer_queue_size,
            writer_batch_size=self.writer_batch_size, attributes=self.attributes,
        )
        self.control_fps = recording.fps
        self.dataset_dir = recording.dataset_dir
        self.attributes = recording.attributes
        if self.mode not in ("sync", "async"):
            raise ValueError("mode must be sync or async.")
        if self.new_action_fuse not in ("latest", "aggregate"):
            raise ValueError("new_action_fuse must be latest or aggregate.")
        if self.chunk_steps is not None and (
            isinstance(self.chunk_steps, bool) or not isinstance(self.chunk_steps, int)
            or self.chunk_steps <= 0
        ):
            raise ValueError("chunk_steps must be a positive integer.")
        for name in ("prefetch_threshold", "new_action_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1.")
        if not isinstance(self.keyboard_debug, bool):
            raise ValueError("keyboard_debug must be a boolean.")
        if not isinstance(self.reset_on_touch, bool):
            raise ValueError("reset_on_touch must be a boolean.")
        for name in (
            "reset_arm_load_threshold", "reset_gripper_load_threshold",
            "reset_load_duration", "reset_position_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
        if self.instruction is not None and (
            not isinstance(self.instruction, str) or not self.instruction.strip()
        ):
            raise ValueError("instruction must be a nonempty string.")
        if not isinstance(self.camera_map, dict) or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in self.camera_map.items()
        ):
            raise ValueError("camera_map must map model camera names to native camera names.")
        self.camera_map = dict(self.camera_map)


def _column_keys(metadata, name, native_keys):
    keys = metadata.get(name)
    if keys is None:
        # Collection obtains these columns from the same native feature maps.
        keys = native_keys
    if isinstance(keys, (str, bytes)) or not isinstance(keys, (tuple, list)) or not keys:
        raise ValueError(f"Policy {name} must be a nonempty sequence of feature names.")
    if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError(f"Policy {name} must contain unique, nonempty feature names.")
    dimension = metadata.get("state_dim" if name == "state_keys" else "action_dim", len(keys))
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension != len(keys):
        raise ValueError(
            f"Policy {name} has {dimension} columns, but the selected ordering has {len(keys)}. "
            "Check the robot configuration or provide the training data_description.json."
        )
    return tuple(keys)


def _deployment_features(robot, metadata, options):
    native_actions = _numeric_keys(robot.action_features, "Robot actions")
    if len(native_actions) != len(robot.action_features):
        raise ValueError("Deployment requires scalar numeric robot actions.")
    native_states = _numeric_keys(robot.observation_features, "Robot observations")
    state_keys = _column_keys(metadata, "state_keys", native_states)
    action_keys = _column_keys(metadata, "action_keys", native_actions)
    if not set(state_keys).issubset(native_states):
        raise ValueError("Policy state names must be available in the native robot observations.")
    if set(action_keys) != set(native_actions):
        raise ValueError("Policy action names must match all native robot action names.")
    camera_keys = metadata.get("camera_keys", [])
    if not isinstance(camera_keys, (list, tuple)) or any(
        not isinstance(key, str) or not key for key in camera_keys
    ) or len(set(camera_keys)) != len(camera_keys):
        raise ValueError("Policy camera_keys must contain unique camera names.")
    if set(options.camera_map).difference(camera_keys):
        raise ValueError("--camera-map contains cameras that the trained policy does not use.")
    cameras = {
        name: options.camera_map.get(name, name.removeprefix("image_"))
        for name in camera_keys
    }
    if len(set(cameras.values())) != len(cameras):
        raise ValueError("Each model camera must map to a different native camera.")
    image_keys = {}
    for name, feature in robot.observation_features.items():
        if isinstance(feature, (tuple, list)):
            if name in cameras.values() and (len(feature) != 3 or feature[2] != 3):
                raise ValueError(f"Model camera {name!r} requires an RGB (H, W, 3) observation.")
            if options.record or options.save_video:
                if len(feature) != 3 or feature[2] != 3 or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in feature
                ):
                    raise ValueError(f"Recording camera {name!r} requires positive RGB dimensions in the robot YAML.")
                image_keys[name] = tuple(feature)
    if any(
        name not in robot.observation_features
        or not isinstance(robot.observation_features[name], (tuple, list))
        for name in cameras.values()
    ):
        raise ValueError(
            f"Missing model cameras: {cameras}. Configure these native cameras or use --camera-map MODEL=NATIVE."
        )
    return state_keys, action_keys, cameras, image_keys


class ObservationHistory:
    """Capture owned RGB/state history in the model's training column order."""

    def __init__(self, state_keys, cameras, width, instruction=None):
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("Policy history_width must be a positive integer.")
        self.state_keys = state_keys
        self.cameras = cameras
        self.width = width
        self.instruction = instruction
        self.clear()

    def clear(self):
        self.states = deque(maxlen=self.width)
        self.images = {name: deque(maxlen=self.width) for name in self.cameras}

    def update(self, observation):
        try:
            state = np.asarray([observation[key] for key in self.state_keys], dtype=np.float32)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Robot observations are missing the policy's scalar states.") from exc
        if state.shape != (len(self.state_keys),) or not np.isfinite(state).all():
            raise ValueError("Policy states must be finite scalar numbers.")
        frames = {}
        for model_name, native_name in self.cameras.items():
            if native_name not in observation:
                raise ValueError(f"Robot observation is missing camera {native_name!r}.")
            frame = np.asarray(observation[native_name])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape) <= 0:
                raise ValueError(f"Camera {native_name!r} must provide uint8 RGB HWC frames.")
            frames[model_name] = frame.copy()
        count = self.width if not self.states else 1
        for _ in range(count):
            self.states.append(state.copy())
            for name, frame in frames.items():
                self.images[name].append(frame)

    def snapshot(self):
        result = {
            "state": np.stack(self.states),
            "images": {name: np.stack(frames) for name, frames in self.images.items()},
        }
        if self.instruction is not None:
            result["instruction"] = self.instruction
        return result


def run_deployment(
    robot_config: HardwareSetupConfig,
    predictor,
    options: DeploymentOptions,
    *,
    backend_factory=None,
    processors_factory=None,
    keyboard_factory=None,
    inference_factory=None,
    recorder_factory=None,
    video_recorder_factory=None,
    preview_factory=None,
    clock=perf_counter,
    sleep=None,
):
    """Execute active model rounds with optional native-bus manual reset."""
    if robot_config.kind != "robot":
        raise ValueError("--robot.config must contain a robot section.")
    metadata = predictor.metadata
    if not isinstance(metadata, dict):
        raise ValueError("The policy must expose a metadata mapping.")
    if metadata.get("action_key") != "action_abs":
        raise ValueError(
            "Real-robot deployment requires action_abs targets; "
            f"this policy selects {metadata.get('action_key')!r}."
        )
    instruction = options.instruction if options.instruction is not None else metadata.get("instruction")
    if metadata.get("requires_instruction", False) and (
        not isinstance(instruction, str) or not instruction.strip()
    ):
        raise ValueError("This trained policy requires --instruction before starting deployment.")
    if options.preview and not _preview_camera_configs(robot_config.parameters):
        raise ValueError("--preview requires cameras in the robot YAML.")
    if processors_factory is None:
        from lerobot.processor import make_default_processors

        processors_factory = make_default_processors
    if keyboard_factory is None:
        from .utils.keyboard import ArrowKeys

        keyboard_factory = lambda: ArrowKeys(debug=options.keyboard_debug)
    if inference_factory is None:
        from .utils.inference import ChunkInference

        inference_factory = ChunkInference
    if sleep is None:
        from lerobot.utils.robot_utils import precise_sleep

        sleep = precise_sleep
    backend_factory = backend_factory or _runtime_backend
    robot_snapshot = robot_config.to_dict()
    if options.record:
        from .utils.trajectory import (
            EpisodeRecorder, ensure_data_description, prepare_data_description, prepare_episode_metadata,
        )

        recorder_factory = recorder_factory or EpisodeRecorder
        episode_metadata = dict(options.attributes)
        success, comment = episode_metadata.pop("success", True), episode_metadata.pop("comment", "")
        episode_metadata.setdefault("sim", False)
        episode_metadata = prepare_episode_metadata(episode_metadata)
    else:
        episode_metadata, success, comment = {}, True, ""
    if options.save_video and video_recorder_factory is None:
        from .utils.episode_video import CameraVideoRecorder

        video_recorder_factory = CameraVideoRecorder
    inference = inference_factory(
        predictor, mode=options.mode, chunk_steps=options.chunk_steps,
        prefetch_threshold=options.prefetch_threshold, new_weight=options.new_action_weight,
        new_action_fuse=options.new_action_fuse,
    )
    recorder = None
    video_recorder = None
    phase = "waiting"
    episode_start = None
    result = {"frames": 0, "rounds": 0, "saved": [], "discarded": 0, "pending": None, "inference_waits": 0}
    if options.save_video:
        result["videos"] = []

    def finish_video():
        nonlocal video_recorder
        if video_recorder is not None:
            current = video_recorder
            video_recorder = None
            try:
                videos = current.finish()
            except BaseException:
                try:
                    current.abort()
                except Exception:
                    logger.exception("Could not clean up the failed camera video recording.")
                raise
            if videos:
                result["videos"].append({name: str(path) for name, path in videos.items()})
                for name, path in videos.items():
                    print(f"Saved camera {name!r} video: {path}.")

    def retain_pending():
        nonlocal recorder
        try:
            if recorder is not None:
                current = recorder
                recorder = None
                pending = current.close()
                result["pending"] = str(pending) if pending is not None else None
                if pending is not None:
                    print(f"Unconfirmed trajectory retained at {pending}; it has not been added to episodes/.")
        finally:
            finish_video()

    try:
        with ExitStack() as stack:
            backend = stack.enter_context(_managed_backend(backend_factory(robot_config)))
            robot = backend.device
            state_keys, action_keys, cameras, image_keys = _deployment_features(robot, metadata, options)
            if options.save_video and not image_keys:
                raise ValueError("--save-video requires at least one RGB camera in the robot YAML.")
            native_config = getattr(backend, "native_config", robot_config.parameters)
            _check_rgb_configs(native_config)
            trained_config = metadata.get("robot_config")
            if isinstance(trained_config, dict):
                trained_units = _degree_settings(trained_config.get("robot", trained_config))
                native_units = _degree_settings(native_config)
                if any(native_units.get(key) != value for key, value in trained_units.items()):
                    raise ValueError("Robot use_degrees settings differ from the training data description.")
            touch_reset = None
            if options.reset_on_touch:
                from .utils.touch_reset import ManualResetOutOfRange, TouchReset

                # Capability checks run before connecting or changing torque.
                touch_reset = TouchReset(
                    robot, arm_load_threshold=options.reset_arm_load_threshold,
                    gripper_load_threshold=options.reset_gripper_load_threshold,
                    load_duration=options.reset_load_duration,
                    position_threshold=options.reset_position_threshold,
                )
            history = ObservationHistory(state_keys, cameras, metadata.get("history_width", 1), instruction)
            if options.record:
                description = prepare_data_description(
                    state_keys=state_keys, action_keys=action_keys,
                    robot_config=robot_snapshot, teleop_config=None,
                )
                ensure_data_description(options.dataset_dir, description)
            keyboard = stack.enter_context(keyboard_factory())
            preview = None
            if options.preview:
                if preview_factory is None:
                    from .utils.camera_preview import CameraPreview

                    preview_factory = CameraPreview
                preview = preview_factory("WROBO deployment cameras")
                stack.callback(_close_preview, preview)
                preview_names = _preview_image_keys(robot)
                preview_metadata = _preview_metadata(_preview_camera_configs(native_config))
            _, action_processor, observation_processor = processors_factory()
            backend.connect()
            if not robot.is_calibrated:
                raise ValueError(
                    "Calibration is missing or does not match the motors. "
                    "Use setup calibrate --allow-existing with this hardware YAML first."
                )
            keyboard.clear()
            print(f"Deployment ready ({options.mode}). Right starts a round; Right stops it. Ctrl+C exits.")
            print(f"State columns: {list(state_keys)}; action columns: {list(action_keys)}.")
            if touch_reset is not None:
                print("Touch reset enabled. After stopping, wait for the baseline-ready message; "
                      "arm joints and gripper unlock separately when moved by hand.")

            def handle_event(event):
                nonlocal phase, recorder, video_recorder, episode_start
                previous_phase = phase
                if event == "right":
                    if phase == "waiting":
                        if touch_reset is not None:
                            try:
                                touch_reset.resume()
                            except ManualResetOutOfRange as exc:
                                print(f"Manual reset incomplete: {exc}")
                                keyboard.clear()
                                if options.keyboard_debug:
                                    print("[keyboard] right: waiting -> waiting (manual reset out of range)", flush=True)
                                return
                        if options.record:
                            recorder = recorder_factory(
                                options.dataset_dir, state_keys=state_keys, action_keys=action_keys,
                                image_keys=image_keys, encoding=options.img_encoding, jpeg_quality=options.jpeg_quality,
                                metadata=episode_metadata, fps=options.control_fps, robot_config=robot_snapshot,
                                teleop_config=None, writer_queue_size=options.writer_queue_size,
                                writer_batch_size=options.writer_batch_size,
                            )
                        if options.save_video:
                            video_recorder = video_recorder_factory(
                                options.video_dir, image_keys=image_keys, fps=options.control_fps,
                                writer_queue_size=options.writer_queue_size,
                            )
                        history.clear()
                        inference.start()
                        episode_start = clock()
                        phase = "running"
                        result["rounds"] += 1
                        print("Model control started. Right stops this round.")
                    elif phase == "running":
                        inference.stop()
                        if touch_reset is not None:
                            touch_reset.hold(clock())
                        cancel = getattr(predictor, "cancel", None)
                        if callable(cancel):
                            cancel()
                        phase = "review" if options.record else "waiting"
                        finish_video()
                        print("Model control stopped. Reset the real environment; " + (
                            "Right saves, Left discards." if options.record else "Right starts the next round."
                        ))
                        if touch_reset is not None:
                            print("Holding the stopped pose. Let the arm and gripper settle; "
                                  "then move each group by hand to unlock it.")
                    else:
                        saved = recorder.finish(success=success, comment=comment)
                        if saved is not None:
                            result["saved"].append(str(saved))
                            print(f"Saved {saved}.")
                        recorder = None
                        phase = "waiting"
                        print("Right starts the next round.")
                    keyboard.clear()
                elif event == "left" and phase == "review":
                    recorder.abort()
                    recorder = None
                    result["discarded"] += 1
                    phase = "waiting"
                    keyboard.clear()
                    print("Trajectory discarded. Right starts the next round.")
                if options.keyboard_debug and event is not None:
                    print(f"[keyboard] {event}: {previous_phase} -> {phase}", flush=True)

            next_tick = clock()
            try:
                while True:
                    handle_event(keyboard.poll())
                    sample_time = clock()
                    observation = robot.get_observation()
                    processed_observation = observation_processor(observation)
                    if phase == "running":
                        # Video captures camera observations independently of
                        # HDF5 samples and whether an action is ready this tick.
                        if video_recorder is not None:
                            video_recorder.append(observation)
                        history.update(processed_observation)
                        action = inference.action(history.snapshot())
                        # A synchronous model call can take longer than one
                        # period. Honor a stop pressed during that call before
                        # sending its result to the robot.
                        handle_event(keyboard.poll())
                        if phase == "running" and action is not None:
                            vector = np.asarray(action, dtype=np.float32)
                            if vector.shape != (len(action_keys),) or not np.isfinite(vector).all():
                                raise ValueError("Predicted action must be a finite vector matching robot actions.")
                            target = dict(zip(action_keys, (float(value) for value in vector)))
                            target = action_processor((target, observation))
                            sent_action = robot.send_action(target)
                            if recorder is not None:
                                recorder.append(processed_observation, sent_action, timestamp=sample_time - episode_start)
                            result["frames"] += 1
                        elif phase == "running":
                            result["inference_waits"] += 1
                    if touch_reset is not None and phase != "running":
                        # A stop during a blocking prediction uses a later clock
                        # than sample_time. Never feed that stale time to reset.
                        for message in touch_reset.poll(clock()):
                            print(message)
                    if preview is not None:
                        if not preview.show({name: observation[name] for name in preview_names}, preview_metadata):
                            break
                    next_tick += 1 / options.control_fps
                    delay = next_tick - clock()
                    if delay > 0:
                        sleep(delay)
                    else:
                        next_tick = clock()
            except KeyboardInterrupt:
                retain_pending()
            except BaseException:
                try:
                    retain_pending()
                except Exception:
                    logger.exception("Could not retain the pending deployment trajectory.")
                raise
            else:
                retain_pending()
    finally:
        failing = sys.exc_info()[0] is not None
        try:
            inference.stop()
            cancel = getattr(predictor, "cancel", None)
            if callable(cancel):
                # Closing the RPC client also prevents a worker still preparing
                # an observation payload from starting a request after cancel.
                predictor.close()
            inference.close()
        except Exception:
            if not failing:
                raise
            logger.exception("Inference cleanup also failed while handling a deployment error.")
    return result


_CHUNK_STEPS_ARGUMENT = (
    "chunk_steps", "chunk-steps",
    {"type": int, "help": "Use this many leading actions per prediction; default uses the whole chunk."},
)
_MODE_ARGUMENTS = {
    "sync": (_CHUNK_STEPS_ARGUMENT,),
    "async": (
        _CHUNK_STEPS_ARGUMENT,
        ("prefetch_threshold", "prefetch-threshold", {
            "type": float,
            "help": "Request a new chunk at or below this remaining fraction (default: 0.5).",
        }),
        ("new_action_weight", "new-action-weight", {
            "type": float,
            "help": "New prediction's weight for aggregate fusion (default: 0.7).",
        }),
        ("new_action_fuse", "new_action_fuse", {
            "choices": ("latest", "aggregate"),
            "help": "latest replaces overlapping targets; aggregate blends old and new (default: aggregate).",
        }),
    ),
}


def _mode_parameters(args):
    """Resolve only the selected namespace; reject explicitly supplied others."""
    parameters = {}
    for mode, arguments in _MODE_ARGUMENTS.items():
        for name, flag, _ in arguments:
            dest = f"{mode}_{name}"
            if hasattr(args, dest):
                if mode != args.mode:
                    raise ValueError(f"--{mode}.{flag} requires --mode {mode}; selected mode is {args.mode}.")
                parameters[name] = getattr(args, dest)
    return parameters


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a trained wrobo policy on native LeRobot hardware, locally or through a policy server.",
        epilog="Right starts/stops each round. With --record: reset the environment, then Right saves or Left discards. "
        "Ctrl+C exits. No episode duration or step limit.",
        allow_abbrev=False,
    )
    parser.add_argument("--robot.config", dest="robot_config", required=True, help="Robot YAML exported by setup.")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda[:N], or HOST:PORT:DEVICE for cloud inference.")
    parser.add_argument("--checkpoint", help="Explicit checkpoint file path; required for local inference.")
    parser.add_argument("--method", type=str.upper, choices=tuple(METHOD_TO_TRAINER), required=True,
                        help="Registered method name; also checked against the cloud policy.")
    parser.add_argument("--config-file", help="Explicit training YAML file path; required for local inference.")
    parser.add_argument("--data-description", help="Optional training data_description.json for explicit column ordering and units.")
    parser.add_argument("--norm-stats-key", metavar="FILE", help="Explicit normalization JSON file path; required for local inference.")
    parser.add_argument("--norm-stats-group", help="Exact group name if the normalization JSON contains multiple datasets.")
    parser.add_argument("--instruction", help="Task text for language-conditioned policies.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for local model inference.")
    parser.add_argument("--amp", action="store_true", help="Enable autocast for local model inference.")
    parser.add_argument("--rpc-timeout", type=float, default=30, help="Cloud RPC deadline in seconds; does not limit round length.")
    parser.add_argument("--mode", choices=tuple(_MODE_ARGUMENTS), default="sync", help="Synchronous chunks or asynchronous prefetch; configure with --MODE.PARAM.")
    parser.add_argument("--control-fps", type=float, default=30, help="Robot control frequency; match training data sampling frequency (default: 30).")
    parser.add_argument("--camera-map", action="append", default=[], metavar="MODEL=NATIVE", help="Override image_wrist -> wrist camera mapping.")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--keyboard-debug", action="store_true",
                        help="Log native arrow events, ignored repeats/replays and round phase transitions.")
    parser.add_argument("--reset-on-touch", action="store_true",
                        help="After stopping, hold position and unlock arm joints and gripper separately on manual load (SO100/SO101 Feetech STS/SMS only).")
    parser.add_argument("--reset-arm-load-threshold", type=float, default=100.0,
                        help="Arm signed Present_Load change for unlocking; raw PWM units, default 100.")
    parser.add_argument("--reset-gripper-load-threshold", type=float, default=100.0,
                        help="Gripper signed Present_Load change for unlocking; raw PWM units, default 100.")
    parser.add_argument("--reset-load-duration", type=float, default=0.15,
                        help="Required continuous load or position change in seconds, default 0.15.")
    parser.add_argument("--reset-position-threshold", type=float, default=20.0,
                        help="Raw position change for unlocking when load is saturated; default 20 encoder ticks.")
    parser.add_argument("--record", action="store_true",
                        help="Record robot inference states, actual executed actions, camera frames and timestamps in HDF5.")
    parser.add_argument("--save-video", action="store_true",
                        help="Independently record camera MP4s for each running round; requires --video-dir.")
    parser.add_argument("--dataset-dir", help="Recording destination; required with --record.")
    parser.add_argument("--video-dir", help="Camera MP4 destination; required with --save-video, independent of --record.")
    parser.add_argument("--img-encoding", choices=("png", "jpeg", "gzip", "raw"), default="jpeg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--writer-queue-size", type=int, default=30)
    parser.add_argument("--writer-batch-size", type=int, default=8)
    parser.add_argument("--attr", action="append", default=[], metavar="NAME=VALUE")
    for mode, arguments in _MODE_ARGUMENTS.items():
        group = parser.add_argument_group(f"{mode} mode", f"These parameters require --mode {mode}.")
        for name, flag, settings in arguments:
            # Suppressing defaults distinguishes explicit options even when
            # they equal a default, so a mismatched namespace cannot be ignored.
            group.add_argument(f"--{mode}.{flag}", dest=f"{mode}_{name}",
                               default=argparse.SUPPRESS, **settings)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    predictor = None
    try:
        mode_parameters = _mode_parameters(args)
        camera_map = {}
        for assignment in args.camera_map:
            key, separator, value = assignment.partition("=")
            if not separator or not key or not value or key in camera_map:
                raise ValueError("--camera-map requires unique MODEL=NATIVE assignments.")
            camera_map[key] = value
        if args.attr:
            from wrobo.utils.hdf5_metadata import _parse_assignment

            attributes = dict(_parse_assignment(item) for item in args.attr)
        else:
            attributes = {}
        options = DeploymentOptions(
            control_fps=args.control_fps, mode=args.mode, **mode_parameters,
            instruction=args.instruction, camera_map=camera_map, record=args.record,
            save_video=args.save_video,
            video_dir=args.video_dir,
            preview=args.preview, keyboard_debug=args.keyboard_debug, dataset_dir=args.dataset_dir,
            reset_on_touch=args.reset_on_touch,
            reset_arm_load_threshold=args.reset_arm_load_threshold,
            reset_gripper_load_threshold=args.reset_gripper_load_threshold,
            reset_load_duration=args.reset_load_duration,
            reset_position_threshold=args.reset_position_threshold,
            img_encoding=args.img_encoding, jpeg_quality=args.jpeg_quality,
            writer_queue_size=args.writer_queue_size, writer_batch_size=args.writer_batch_size,
            attributes=attributes,
        )
        from .utils.policy import ModelOptions, load_policy

        target = parse_device(args.device)
        if target.is_remote:
            if any((args.checkpoint, args.config_file, args.data_description, args.norm_stats_key, args.norm_stats_group, args.compile, args.amp)):
                raise ValueError("Cloud model files and loading options belong on policy_server; specify --method and --device on the robot client.")
            from .utils.policy_transport import RemotePolicyClient

            predictor = RemotePolicyClient(target, timeout=args.rpc_timeout, instruction=args.instruction)
            actual_method = predictor.metadata.get("method")
            if actual_method != args.method:
                raise ValueError(f"Requested method {args.method!r} does not match cloud policy {actual_method!r}.")
        else:
            missing = [name for name in ("checkpoint", "config_file", "norm_stats_key") if not getattr(args, name)]
            if missing:
                raise ValueError("Local inference requires " + ", ".join(f"--{name.replace('_', '-')}" for name in missing) + ".")
            predictor = load_policy(ModelOptions(
                checkpoint=args.checkpoint,
                method=args.method, config_file=args.config_file, data_description=args.data_description,
                norm_stats_key=args.norm_stats_key, norm_stats_group=args.norm_stats_group,
                device=args.device, instruction=args.instruction,
                compile=args.compile, amp=args.amp,
            ))
        robot_config = load_setup_config(args.robot_config)
        result = run_deployment(robot_config, predictor, options)
    except (ImportError, OSError, ValueError, RuntimeError, TypeError, AssertionError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Initialization interrupted.\n")
    finally:
        if predictor is not None:
            predictor.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
