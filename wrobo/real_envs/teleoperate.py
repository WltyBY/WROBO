"""LeRobot teleoperation with optional camera preview and wrobo HDF5 recording."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from contextlib import ExitStack
from dataclasses import dataclass, field, fields, is_dataclass
from numbers import Real
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from .utils.setup_backend import LeRobotSetupBackend, _managed_backend, calibration_diagnostics
from .utils.setup_config import HardwareSetupConfig, load_setup_config


logger = logging.getLogger(__name__)
_OWNED_ATTRIBUTES = {
    "FPS", "fps", "seq_len", "state_keys", "action_keys", "interrupted",
    "robot_config", "teleop_config",
}


@dataclass(kw_only=True)
class TeleoperationOptions:
    fps: float = 30
    record: bool = False
    preview: bool = False
    keyboard_debug: bool = False
    dataset_dir: str | Path | None = None
    img_encoding: str = "jpeg"
    jpeg_quality: int = 95
    writer_queue_size: int = 30
    writer_batch_size: int = 8
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.fps, bool) or not isinstance(self.fps, Real) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be a positive finite number.")
        self.fps = float(self.fps)
        if not isinstance(self.record, bool):
            raise ValueError("record must be a boolean.")
        if not isinstance(self.preview, bool):
            raise ValueError("preview must be a boolean.")
        if not isinstance(self.keyboard_debug, bool):
            raise ValueError("keyboard_debug must be a boolean.")
        if self.keyboard_debug and not self.record:
            raise ValueError("--keyboard-debug requires --record.")
        if self.img_encoding not in ("png", "jpeg", "gzip", "raw"):
            raise ValueError("img_encoding must be png, jpeg, gzip, or raw.")
        if isinstance(self.jpeg_quality, bool) or not isinstance(self.jpeg_quality, int) or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer between 1 and 100.")
        for name in ("writer_queue_size", "writer_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.attributes, dict):
            raise ValueError("attributes must be a mapping.")
        self.attributes = copy.deepcopy(self.attributes)
        if not self.record:
            if self.dataset_dir is not None or self.attributes:
                raise ValueError("--dataset-dir and --attr require --record.")
            return
        if not isinstance(self.dataset_dir, (str, Path)) or not str(self.dataset_dir).strip():
            raise ValueError("--record requires a nonempty --dataset-dir.")
        self.dataset_dir = Path(self.dataset_dir).expanduser().resolve()
        if self.dataset_dir.exists() and not self.dataset_dir.is_dir():
            raise ValueError(f"Dataset destination is not a directory: {self.dataset_dir}")
        owned = _OWNED_ATTRIBUTES.intersection(self.attributes)
        if owned:
            raise ValueError(f"These attributes are reserved or unsupported: {', '.join(sorted(owned))}.")
        if "success" in self.attributes and not isinstance(self.attributes["success"], bool):
            raise ValueError("The success attribute must be a boolean.")
        for name in ("comment", "dataset"):
            if name in self.attributes and not isinstance(self.attributes[name], str):
                raise ValueError(f"The {name} attribute must be a string.")
        if "dataset" in self.attributes and not self.attributes["dataset"].strip():
            raise ValueError("The dataset attribute must be a nonempty string.")
        if self.attributes:
            from wrobo.utils.hdf5_metadata import _prepare_attributes

            _prepare_attributes(self.attributes)


def _numeric_keys(features, label):
    if not isinstance(features, dict) or not features:
        raise ValueError(f"{label} must expose a nonempty feature mapping.")
    keys = []
    for name, feature in features.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} feature names must be nonempty strings.")
        if isinstance(feature, type) and issubclass(feature, Real) and feature is not bool:
            keys.append(name)
    return tuple(keys)


def _degree_settings(config, prefix=()):
    if is_dataclass(config):
        entries = ((entry.name, getattr(config, entry.name)) for entry in fields(config))
    elif isinstance(config, dict):
        entries = config.items()
    elif isinstance(config, SimpleNamespace):
        entries = vars(config).items()
    else:
        return {}
    settings = {}
    for name, value in entries:
        if name == "use_degrees":
            settings[prefix] = value
        elif name != "cameras":
            settings.update(_degree_settings(value, (*prefix, name)))
    return settings


def _check_rgb_configs(config):
    if is_dataclass(config):
        entries = ((entry.name, getattr(config, entry.name)) for entry in fields(config))
    elif isinstance(config, dict):
        entries = config.items()
    elif isinstance(config, SimpleNamespace):
        entries = vars(config).items()
    else:
        return
    for name, value in entries:
        if name == "cameras" and isinstance(value, dict):
            for camera in value.values():
                mode = camera.get("color_mode", "RGB") if isinstance(camera, dict) else getattr(camera, "color_mode", "RGB")
                mode = getattr(mode, "value", mode)
                if str(mode).upper() != "RGB":
                    raise ValueError("Recording requires RGB cameras; set color_mode: RGB in the robot YAML.")
                depth = camera.get("use_depth", False) if isinstance(camera, dict) else getattr(camera, "use_depth", False)
                if depth:
                    raise ValueError("The wrobo RGB recording format requires use_depth: false.")
        else:
            _check_rgb_configs(value)


def _recording_features(robot, teleop, record):
    action_keys = _numeric_keys(robot.action_features, "Robot actions")
    teleop_keys = _numeric_keys(teleop.action_features, "Teleoperator actions")
    if not action_keys or len(action_keys) != len(robot.action_features):
        raise ValueError("Teleoperation requires scalar numeric robot actions.")
    if set(action_keys) != set(teleop_keys) or len(teleop_keys) != len(teleop.action_features):
        raise ValueError("Robot and teleoperator action names must match exactly.")
    observations = robot.observation_features
    state_keys = _numeric_keys(observations, "Robot observations")
    if not state_keys or not set(action_keys).issubset(state_keys):
        raise ValueError("Robot observations must expose positions for all action names.")
    images = {}
    if record:
        for name, feature in observations.items():
            if isinstance(feature, (tuple, list)):
                if len(feature) != 3 or feature[2] != 3 or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in feature
                ):
                    raise ValueError(
                        f"Unsupported camera feature {name!r}: expected an RGB (H, W, 3) shape. "
                        "Set positive camera width and height in the robot YAML."
                    )
                images[name] = tuple(feature)
    return state_keys, action_keys, images


def _preview_camera_configs(config, prefix=""):
    """Match native camera settings to the observation names of single/dual arms."""
    if is_dataclass(config):
        entries = ((entry.name, getattr(config, entry.name)) for entry in fields(config))
    elif isinstance(config, dict):
        entries = config.items()
    elif isinstance(config, SimpleNamespace):
        entries = vars(config).items()
    else:
        return {}
    cameras = {}
    arm_prefixes = {
        "left": "left_", "left_arm": "left_", "left_arm_config": "left_",
        "right": "right_", "right_arm": "right_", "right_arm_config": "right_",
    }
    for name, value in entries:
        if name == "cameras":
            if not isinstance(value, dict):
                raise ValueError("cameras must map camera names to their settings.")
            for camera_name, camera in value.items():
                if not isinstance(camera_name, str) or not camera_name:
                    raise ValueError("Camera names must be nonempty strings.")
                cameras[f"{prefix}{camera_name}"] = camera
        else:
            child_prefix = arm_prefixes.get(name, f"{name}_")
            cameras.update(_preview_camera_configs(value, f"{prefix}{child_prefix}"))
    return cameras


def _preview_metadata(configs):
    descriptions = {}
    for name, camera in configs.items():
        def setting(key, default=None):
            return camera.get(key, default) if isinstance(camera, dict) else getattr(camera, key, default)

        mode = setting("color_mode", "RGB")
        rotation = setting("rotation", 0)
        descriptions[name] = {
            "device": str(setting("index_or_path", setting("serial_number_or_name", ""))),
            "rotation": getattr(rotation, "value", rotation),
            "fps": setting("fps"),
            "color_mode": getattr(mode, "value", mode),
        }
    return descriptions


def _preview_image_keys(robot):
    # Capture dimensions may remain None until cameras connect; the preview uses
    # each actual frame's shape, whereas recordings still need fixed dimensions.
    names = tuple(
        name for name, feature in robot.observation_features.items()
        if isinstance(feature, (tuple, list)) and len(feature) == 3 and feature[2] == 3
    )
    if not names:
        raise ValueError("--preview requires robot camera observations with three color channels.")
    return names


def _close_preview(preview):
    try:
        preview.close()
    except Exception:
        logger.warning("Could not close the camera preview window.", exc_info=True)


def _runtime_backend(config):
    return LeRobotSetupBackend(config, enable_cameras=True)


def _check_calibration(robot_backend, teleop_backend, robot_config, teleop_config):
    """Report each failed arm and its files without changing motor calibration."""
    problems = []
    for label, backend, config in (
        ("follower (robot)", robot_backend, robot_config),
        ("leader (teleop)", teleop_backend, teleop_config),
    ):
        device = backend.device
        if device.is_calibrated:
            continue
        problems.append(
            f"{label}: id={config.parameters['id']!r}, "
            f"port={config.parameters.get('port', 'not specified')!r}"
        )
        calibration_files = getattr(backend, "calibration_files", None)
        paths = calibration_files() if callable(calibration_files) else [config.calibration_file]
        for path in sorted({Path(path) for path in paths}, key=str):
            status = "exists" if path.is_file() else "missing"
            problems.append(f"  Calibration file: {path} [{status}]")
        calibration = getattr(device, "calibration", None)
        if isinstance(calibration, dict):
            problems.append(f"  Loaded calibration entries: {len(calibration)}")
        problems.extend(f"  {line}" for line in calibration_diagnostics(device))
        problems.append(
            "  Check that this id's JSON was loaded and the port belongs to this arm; "
            "existing calibration must match the values stored in its motors."
        )
    if problems:
        raise ValueError(
            "Calibration is missing or does not match the motors.\n"
            + "\n".join(problems)
            + "\nUse python -m wrobo.real_envs.setup calibrate --config YOUR_HARDWARE_YAML "
            "--allow-existing for each failed arm after checking its port and calibration file. "
            "At the driver's prompt, ENTER reuses the loaded calibration; 'c' starts a new calibration."
        )


def run_teleoperation(
    robot_config: HardwareSetupConfig,
    teleop_config: HardwareSetupConfig,
    options: TeleoperationOptions,
    *,
    backend_factory=None,
    processors_factory=None,
    keyboard_factory=None,
    clock=perf_counter,
    sleep=None,
    recorder_factory=None,
    preview_factory=None,
):
    """Run native driver actions; publish an episode only after a save decision."""
    if robot_config.kind != "robot" or teleop_config.kind != "teleop":
        raise ValueError("--robot.config must contain robot; --teleop.config must contain teleop.")
    if options.preview and not _preview_camera_configs(robot_config.parameters):
        raise ValueError("--preview requires cameras in the robot YAML.")
    backend_factory = backend_factory or _runtime_backend
    if processors_factory is None:
        from lerobot.processor import make_default_processors

        processors_factory = make_default_processors
    if sleep is None:
        from lerobot.utils.robot_utils import precise_sleep

        sleep = precise_sleep
    if options.record:
        from .utils.trajectory import (
            EpisodeRecorder, ensure_data_description, prepare_data_description, prepare_episode_metadata,
        )

        recorder_factory = recorder_factory or EpisodeRecorder
        if keyboard_factory is None:
            from .utils.keyboard import ArrowKeys

            keyboard_factory = lambda: ArrowKeys(debug=options.keyboard_debug)
        metadata = dict(options.attributes)
        success = metadata.pop("success", True)
        comment = metadata.pop("comment", "")
        metadata.setdefault("sim", False)
        robot_snapshot = robot_config.to_dict()
        teleop_snapshot = teleop_config.to_dict()
        metadata = prepare_episode_metadata(metadata)
    else:
        metadata, success, comment = {}, True, ""

    result = {"frames": 0, "saved": [], "discarded": 0, "pending": None}
    recorder = None
    phase = "waiting"
    episode_start = None

    def retain_pending():
        nonlocal recorder
        if recorder is not None:
            pending = recorder.close()
            result["pending"] = str(pending) if pending is not None else None
            if pending is not None:
                print(f"Unconfirmed trajectory retained at {pending}; it has not been added to episodes/.")
            recorder = None

    with ExitStack() as stack:
        preview = None
        if options.preview:
            if preview_factory is None:
                from .utils.camera_preview import CameraPreview

                preview_factory = CameraPreview
            preview = preview_factory("WROBO teleoperation cameras")
            stack.callback(_close_preview, preview)
        robot_backend = stack.enter_context(_managed_backend(backend_factory(robot_config)))
        teleop_backend = stack.enter_context(_managed_backend(backend_factory(teleop_config)))
        robot, teleop = robot_backend.device, teleop_backend.device
        state_keys, action_keys, image_keys = _recording_features(robot, teleop, options.record)
        if preview is not None:
            preview_names = _preview_image_keys(robot)
            native = getattr(robot_backend, "native_config", robot_config.parameters)
            preview_metadata = _preview_metadata(_preview_camera_configs(native))
        robot_units = _degree_settings(getattr(robot_backend, "native_config", {}))
        teleop_units = _degree_settings(getattr(teleop_backend, "native_config", {}))
        if robot_units != teleop_units:
            raise ValueError("Robot and teleoperator use_degrees settings must match, including each sub-arm.")
        if options.record:
            _check_rgb_configs(getattr(robot_backend, "native_config", {}))
            description = prepare_data_description(
                state_keys=state_keys, action_keys=action_keys,
                robot_config=robot_snapshot, teleop_config=teleop_snapshot,
            )
            ensure_data_description(options.dataset_dir, description)
            keyboard = stack.enter_context(keyboard_factory())
        else:
            keyboard = None
        teleop_processor, action_processor, observation_processor = processors_factory()
        teleop_backend.connect()
        robot_backend.connect()
        _check_calibration(robot_backend, teleop_backend, robot_config, teleop_config)
        if keyboard is not None:
            keyboard.clear()
            print("Ready. Right arrow starts a trajectory; Ctrl+C exits.")
        else:
            print("Teleoperation ready. Ctrl+C exits.")
        if preview is not None:
            print("Camera preview: q / Esc / close the window exits teleoperation.")
        next_tick = clock()
        try:
            while True:
                event = keyboard.poll() if keyboard is not None else None
                previous_phase = phase
                if event == "right":
                    if phase == "waiting":
                        recorder = recorder_factory(
                            options.dataset_dir, state_keys=state_keys, action_keys=action_keys,
                            image_keys=image_keys, encoding=options.img_encoding,
                            jpeg_quality=options.jpeg_quality, metadata=metadata, fps=options.fps,
                            robot_config=robot_snapshot, teleop_config=teleop_snapshot,
                            writer_queue_size=options.writer_queue_size,
                            writer_batch_size=options.writer_batch_size,
                        )
                        episode_start = clock()
                        phase = "recording"
                        print("Recording. Right arrow ends this trajectory.")
                    elif phase == "recording":
                        phase = "review"
                        print("Recording ended. Reset the real environment; then Right saves, Left discards.")
                    else:
                        saved = recorder.finish(success=success, comment=comment)
                        if saved is not None:
                            result["saved"].append(str(saved))
                            print(f"Saved {saved}. \n")
                        recorder = None
                        phase = "waiting"
                        print("Right arrow starts the next trajectory.")
                    keyboard.clear()
                elif event == "left" and phase == "review":
                    recorder.abort()
                    recorder = None
                    result["discarded"] += 1
                    phase = "waiting"
                    keyboard.clear()
                    print("Trajectory discarded. Right arrow starts the next trajectory.")

                if options.keyboard_debug and event is not None:
                    print(f"[keyboard] {event}: {previous_phase} -> {phase}", flush=True)

                sample_time = clock()
                observation = robot.get_observation()
                processed_observation = observation_processor(observation)
                raw_action = teleop.get_action()
                teleop_action = teleop_processor((raw_action, observation))
                target = action_processor((teleop_action, observation))
                sent_action = robot.send_action(target)
                if phase == "recording":
                    recorder.append(processed_observation, sent_action, timestamp=sample_time - episode_start)
                result["frames"] += 1
                if preview is not None:
                    missing = set(preview_names).difference(observation)
                    if missing:
                        raise RuntimeError(f"Camera preview observations are missing: {', '.join(sorted(missing))}.")
                    preview_frames = {name: observation[name] for name in preview_names}
                    if not preview.show(preview_frames, preview_metadata):
                        break
                next_tick += 1 / options.fps
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
                logger.exception("Could not close the pending trajectory after a control-loop failure.")
            raise
        else:
            retain_pending()
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run LeRobot teleoperation, optionally previewing cameras and recording wrobo HDF5 episodes.",
        epilog="Recording: Right starts, Right ends; reset the environment, then Right saves or Left discards. "
        "Each decision returns to waiting for Right to start the next trajectory. Ctrl+C exits.",
        allow_abbrev=False,
    )
    parser.add_argument("--robot.config", dest="robot_config", required=True, help="Follower YAML exported by setup.")
    parser.add_argument("--teleop.config", dest="teleop_config", required=True, help="Leader YAML exported by setup.")
    parser.add_argument("--fps", type=float, default=30, help="Control frequency; does not limit episode length.")
    parser.add_argument("--record", action="store_true", help="Enable keyboard-controlled trajectory collection.")
    parser.add_argument("--keyboard-debug", action="store_true",
                        help="With --record, print native arrow events, repeat filtering, and recording phase changes.")
    parser.add_argument("--preview", action="store_true",
                        help="Show all robot cameras in one OpenCV GUI window; q, Esc, or close exits teleoperation.")
    parser.add_argument("--dataset-dir", help="wrobo dataset destination; required with --record.")
    parser.add_argument("--img-encoding", choices=("png", "jpeg", "gzip", "raw"),
                        default=TeleoperationOptions.img_encoding)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--writer-queue-size", type=int, default=30,
                        help="Maximum queued camera samples; a full queue waits without dropping frames.")
    parser.add_argument("--writer-batch-size", type=int, default=8,
                        help="Camera samples per HDF5 append batch; does not limit episode length.")
    parser.add_argument("--attr", action="append", default=[], metavar="NAME=VALUE", help="Recording attrs, using JSON or text values.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.attr:
            from wrobo.utils.hdf5_metadata import _parse_assignment

            attributes = dict(_parse_assignment(assignment) for assignment in args.attr)
        else:
            attributes = {}
        options = TeleoperationOptions(
            fps=args.fps, record=args.record, preview=args.preview, keyboard_debug=args.keyboard_debug,
            dataset_dir=args.dataset_dir,
            img_encoding=args.img_encoding, jpeg_quality=args.jpeg_quality, attributes=attributes,
            writer_queue_size=args.writer_queue_size, writer_batch_size=args.writer_batch_size,
        )
        result = run_teleoperation(
            load_setup_config(args.robot_config), load_setup_config(args.teleop_config), options,
        )
    except (ImportError, OSError, ValueError, RuntimeError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Initialization interrupted.\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
