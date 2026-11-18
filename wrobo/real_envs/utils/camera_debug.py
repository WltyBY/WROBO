"""Camera-only live previews through LeRobot capture and a shared GUI grid."""

from dataclasses import replace
import logging
from pathlib import Path
import time
from typing import Any

from .discovery import discover_cameras
from .camera_preview import CameraPreview


logger = logging.getLogger(__name__)


def _positive_int(name: str, value: int | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")


def _camera_parameters(parameters: dict, prefix: str = "") -> dict[str, dict]:
    """Extract camera sections, including the children of a dual-arm robot."""
    result = {}
    for key, value in parameters.items():
        if key == "cameras":
            if not isinstance(value, dict):
                raise ValueError("cameras must be a mapping of names to camera settings.")
            entries = {f"{prefix}{name}": settings for name, settings in value.items()}
            if any(not isinstance(name, str) or not name or not isinstance(settings, dict)
                   for name, settings in value.items()):
                raise ValueError("Each camera must have a nonempty name and a settings mapping.")
        elif isinstance(value, dict):
            entries = _camera_parameters(value, f"{prefix}{key}_")
        else:
            continue
        for name, settings in entries.items():
            if name in result:
                raise ValueError(f"Duplicate camera name: {name!r}.")
            result[name] = settings
    return result


def _load_camera_configs(
    backend: str,
    *,
    config_path: str | Path | None = None,
    camera_id: str | int | None = None,
    rotation: int | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
) -> dict[str, Any]:
    """Decode camera settings without importing or validating any robot driver."""
    if backend not in ("opencv", "realsense"):
        raise ValueError("Camera backend must be 'opencv' or 'realsense'.")
    if config_path is not None and camera_id is not None:
        raise ValueError("Use either --config or --camera, not both.")
    if rotation is not None and (
        isinstance(rotation, bool) or rotation not in (0, 90, 180, -90)
    ):
        raise ValueError("rotation must be 0, 90, 180, or -90.")
    for name, value in (("width", width), ("height", height), ("fps", fps)):
        _positive_int(name, value)
    if (width is None) != (height is None):
        raise ValueError("Set --width and --height together; they describe the output image.")

    import draccus

    if backend == "opencv":
        from lerobot.cameras.opencv import OpenCVCameraConfig as config_class

        camera_type, id_field = "opencv", "index_or_path"
    else:
        from lerobot.cameras.realsense import RealSenseCameraConfig as config_class

        camera_type, id_field = "intelrealsense", "serial_number_or_name"

    if config_path is not None:
        import yaml

        with Path(config_path).expanduser().open(encoding="utf-8") as stream:
            try:
                document = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid camera debug YAML {config_path}: {exc}") from exc
        if not isinstance(document, dict) or not isinstance(document.get("robot"), dict):
            raise ValueError("Camera debug configuration must contain a robot mapping.")
        parameters = _camera_parameters(document["robot"])
        parameters = {
            name: settings for name, settings in parameters.items()
            if settings.get("type") == camera_type
        }
    else:
        devices = [camera_id] if camera_id is not None else [
            info["id"] for info in discover_cameras(backend)
        ]
        parameters = {}
        for index, device in enumerate(devices):
            if backend == "opencv":
                if isinstance(device, str) and device.lstrip("-").isdecimal():
                    device = int(device)
                if isinstance(device, bool) or (isinstance(device, int) and device < 0):
                    raise ValueError("OpenCV camera index must be a nonnegative integer.")
            else:
                device = str(device)
            parameters[f"camera_{index}"] = {id_field: device}

    if not parameters:
        raise ValueError(f"No {backend} cameras found in the configuration or connected devices.")

    configs = {}
    for name, settings in parameters.items():
        try:
            native = draccus.decode(config_class, {
                key: value for key, value in settings.items() if key != "type"
            })
            changes = {
                key: value for key, value in (("rotation", rotation), ("width", width),
                                             ("height", height), ("fps", fps))
                if value is not None
            }
            # Preserve capture dimensions when only the debugging rotation changes.
            if rotation is not None and width is None and (
                (abs(int(native.rotation)) == 90) != (abs(rotation) == 90)
            ):
                changes.update(width=native.height, height=native.width)
            native = replace(native, **changes)
            for field in ("width", "height", "fps"):
                _positive_int(f"{name}.{field}", getattr(native, field))
            if (native.width is None) != (native.height is None):
                raise ValueError("Camera width and height must both be set or both be omitted.")
            if getattr(native, "use_rgb", True) is False:
                raise ValueError("Camera debug preview requires use_rgb: true.")
            configs[name] = native
        except Exception as exc:
            raise ValueError(f"Invalid camera settings for {name!r}: {exc}") from exc
    return configs


def make_cameras_from_configs(configs):
    """Load LeRobot's camera factory only when an actual preview is requested."""
    from lerobot.cameras.utils import make_cameras_from_configs as factory

    return factory(configs)


def debug_cameras(
    backend: str = "opencv",
    *,
    config_path: str | Path | None = None,
    camera_id: str | int | None = None,
    rotation: int | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
) -> dict[str, Any]:
    """Show selected cameras in one grid; q, Escape, close, or Ctrl+C stops."""
    preview = CameraPreview("WROBO camera debug")
    configs = _load_camera_configs(
        backend, config_path=config_path, camera_id=camera_id,
        rotation=rotation, width=width, height=height, fps=fps,
    )
    cameras = make_cameras_from_configs(configs)
    attempted = []
    descriptions = {}
    try:
        for name, camera in cameras.items():
            attempted.append(camera)
            camera.connect()
            native = configs[name]
            device = getattr(native, "index_or_path", getattr(native, "serial_number_or_name", ""))
            descriptions[name] = {
                "device": str(device), "rotation": int(native.rotation), "fps": camera.fps,
                "color_mode": getattr(native.color_mode, "value", native.color_mode),
            }
            print(f"{name} | {device} | rotation={int(native.rotation)}: "
                  f"{camera.width}x{camera.height}, fps={camera.fps}", flush=True)
        print("Camera debug: q / Esc / close the window / Ctrl+C to exit.", flush=True)
        while True:
            started = time.monotonic()
            frames = {}
            for name, camera in cameras.items():
                frame = camera.async_read(timeout_ms=1000)
                frames[name] = frame
                descriptions[name]["shape"] = list(frame.shape)
            if not preview.show(frames, descriptions):
                break
            time.sleep(max(0, 1 / 30 - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        for camera in reversed(attempted):
            try:
                camera.disconnect()
            except Exception:
                logger.warning("Could not disconnect a debug camera.", exc_info=True)
        try:
            preview.close()
        except Exception:
            logger.warning("Could not close OpenCV preview windows.", exc_info=True)
    return {"status": "stopped", "cameras": descriptions}
