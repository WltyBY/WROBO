"""wrobo hardware configuration and LeRobot setup: python -m wrobo.real_envs.setup."""

import argparse
import json
import sys
from pathlib import Path

from .utils.setup_backend import (
    available_device_types,
    calibrate_hardware,
    normalize_setup_config,
    setup_hardware_motors,
)
from .utils.setup_config import HardwareSetupConfig, load_setup_config
from .utils.discovery import discover_cameras, discover_ports, identify_port
from .utils.camera_debug import debug_cameras


def _set_parameter(parameters: dict, key: str, value: str, kind: str) -> None:
    """Accept a dotted device field and a safe YAML scalar, list, or mapping."""
    import yaml

    parts = key.split(".")
    if any(not part for part in parts):
        raise ValueError(f"Invalid device parameter --{kind}.{key}; nested keys use dots.")
    if parts[0] in ("type", "id", "calibration_dir", "port"):
        raise ValueError(
            f"Set {parts[0]!r} using its dedicated command-line argument --{kind}.{parts[0]}."
        )
    current = parameters
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot add a nested parameter below {part!r}.")
        current = child
    try:
        current[parts[-1]] = yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML value for {key!r}: {exc}") from exc


class _DeviceParameterAction(argparse.Action):
    """Keep driver-specific assignments in command-line order."""

    def __call__(self, parser, namespace, values, option_string=None):
        parameters = getattr(namespace, self.dest)
        if parameters is None:
            parameters = {"robot": [], "teleop": []}
            setattr(namespace, self.dest, parameters)
        kind, key = option_string[2:].split(".", 1)
        parameters[kind].append((key, values))


def generate_config(args, device_parameters) -> HardwareSetupConfig:
    selected = [
        kind for kind in ("robot", "teleop")
        if device_parameters[kind] or any(
            getattr(args, f"{kind}_{field}") is not None
            for field in ("type", "id", "port", "calibration_dir")
        )
    ]
    if len(selected) != 1:
        raise ValueError("Specify exactly one device group: --robot.* or --teleop.*.")
    kind = selected[0]
    missing = [
        f"--{kind}.{field}" for field in ("type", "id")
        if getattr(args, f"{kind}_{field}") is None
    ]
    if missing:
        raise ValueError(f"Required device arguments: {', '.join(missing)}.")
    device_type = getattr(args, f"{kind}_type")
    device_id = getattr(args, f"{kind}_id")
    port = getattr(args, f"{kind}_port")
    directory = getattr(args, f"{kind}_calibration_dir")
    if directory is not None and not directory.strip():
        raise ValueError(f"--{kind}.calibration_dir must be a nonempty directory path.")
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Configuration exists: {output}. Use --force to replace it.")
    calibration_dir = (
        Path(directory).expanduser().resolve()
        if directory is not None
        else output.parent / "calibrations" / kind / device_type
    )
    parameters = {"id": device_id, "calibration_dir": calibration_dir}
    if port is not None:
        parameters["port"] = port
    for key, value in device_parameters[kind]:
        _set_parameter(parameters, key, value, kind)
    config = normalize_setup_config(
        HardwareSetupConfig(
            device_type=device_type,
            parameters=parameters,
            kind=kind,
        )
    )
    config.save(output, force=args.force)
    return config


def build_parser(device_fields=()) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover hardware, export wrobo settings, and initialize through LeRobot drivers.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("types", help="List device types supported by wrobo's driver mapping.")
    ports = commands.add_parser("ports", help="List serial ports without opening them.")
    ports.add_argument(
        "--identify", action="store_true", help="Identify a port by unplugging its USB cable."
    )
    cameras = commands.add_parser("cameras", help="Discover cameras or show a live debug preview.")
    cameras.add_argument("--backend", choices=("opencv", "realsense"), default="opencv")
    cameras.add_argument("--debug", action="store_true", help="Open live OpenCV GUI preview windows.")
    source = cameras.add_mutually_exclusive_group()
    source.add_argument("--config", help="Preview cameras from a robot YAML without connecting the robot.")
    source.add_argument("--camera", help="Preview one camera index/path or RealSense serial instead of discovering all.")
    cameras.add_argument("--rotation", type=int, choices=(0, 90, 180, -90), help="Temporary debug rotation in degrees.")
    cameras.add_argument("--width", type=int, help="Debug output width; set together with --height.")
    cameras.add_argument("--height", type=int, help="Debug output height; set together with --width.")
    cameras.add_argument("--fps", type=int, help="Requested camera frame rate for debugging.")

    create = commands.add_parser(
        "export-config", aliases=["create-config"],
        help="Validate and save hardware settings, including driver defaults.",
        description="Export one robot or teleoperator. Additional driver fields use "
        "--robot.FIELD VALUE or --teleop.FIELD VALUE; nested fields use dots "
        "and values are parsed as YAML. Both --FIELD VALUE and --FIELD=VALUE are accepted.",
        allow_abbrev=False,
    )
    known_device_fields = {
        f"--{kind}.{field}"
        for kind in ("robot", "teleop")
        for field in ("type", "id", "port", "calibration_dir")
    }
    for kind, title in (("robot", "Robot / follower"), ("teleop", "Teleoperator / leader")):
        group = create.add_argument_group(title)
        group.add_argument(
            f"--{kind}.type", dest=f"{kind}_type", metavar="TYPE",
            help="Device type, e.g. so101; required for this group. See 'types'.",
        )
        group.add_argument(
            f"--{kind}.id", dest=f"{kind}_id", metavar="ID",
            help="Unique calibration name; required for this group.",
        )
        group.add_argument(
            f"--{kind}.port", dest=f"{kind}_port", metavar="PORT",
            help="Serial path, when required by the selected driver.",
        )
        group.add_argument(
            f"--{kind}.calibration_dir", dest=f"{kind}_calibration_dir", metavar="DIRECTORY",
            help="Directory holding ID.json; defaults beside the output YAML.",
        )
    create.set_defaults(device_parameters=None)
    for field in dict.fromkeys(device_fields):
        if field not in known_device_fields:
            create.add_argument(
                field, dest="device_parameters", action=_DeviceParameterAction,
                metavar="VALUE", help=argparse.SUPPRESS,
            )
    create.add_argument("--output", required=True, help="Destination YAML file.")
    create.add_argument("--force", action="store_true", help="Replace an existing YAML file.")

    show = commands.add_parser("show-config", help="Print a configuration with its resolved calibration path.")
    show.add_argument("--config", required=True)
    motors = commands.add_parser("setup-motors", help="Use the driver's interactive motor ID/baudrate setup.")
    motors.add_argument("--config", required=True)
    calibrate = commands.add_parser(
        "calibrate", help="Calibrate through the driver and save initialization metadata to YAML."
    )
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument(
        "--allow-existing", action="store_true",
        help="Allow the driver to reuse or replace an existing calibration.",
    )
    return parser


def main(argv=None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    device_fields = [
        argument.partition("=")[0] for argument in arguments
        if argument.startswith(("--robot.", "--teleop."))
    ]
    parser = build_parser(device_fields)
    args = parser.parse_args(arguments)
    try:
        if args.command == "types":
            result = available_device_types()
        elif args.command == "ports":
            result = identify_port() if args.identify else discover_ports()
        elif args.command == "cameras":
            if args.debug:
                result = debug_cameras(
                    args.backend, config_path=args.config, camera_id=args.camera,
                    rotation=args.rotation, width=args.width, height=args.height, fps=args.fps,
                )
            else:
                if any(value is not None for value in (
                    args.config, args.camera, args.rotation, args.width, args.height, args.fps,
                )):
                    raise ValueError("Camera preview options require --debug.")
                result = discover_cameras(args.backend)
        elif args.command in ("export-config", "create-config"):
            device_parameters = args.device_parameters or {"robot": [], "teleop": []}
            config = generate_config(args, device_parameters)
            result = {
                "config": str(Path(args.output).expanduser().resolve()),
                "calibration_file": str(config.calibration_file),
            }
        else:
            config = load_setup_config(args.config)
            if args.command == "show-config":
                result = config.to_dict()
            elif args.command == "setup-motors":
                setup_hardware_motors(config)
                result = {"motor_setup": "complete", "type": config.device_type}
            else:
                result = calibrate_hardware(config, allow_existing=args.allow_existing)
                config.extra_sections["initialization"] = result
                config.save(args.config, force=True)
    except (ImportError, ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
