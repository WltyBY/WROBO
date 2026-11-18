"""Discover hardware identifiers without selecting a particular robot driver."""

from collections.abc import Callable
from pathlib import Path
from time import sleep as _sleep
from typing import Any


_SERIAL_LINK_DIRS = (Path("/dev/serial/by-id"), Path("/dev/serial/by-path"))
_CAMERA_LINK_DIRS = (Path("/dev/v4l/by-id"), Path("/dev/v4l/by-path"))


def _stable_paths(device: str, directories: tuple[Path, ...]) -> list[str]:
    """Return existing persistent symlinks pointing to this device."""
    try:
        target = Path(device).resolve()
    except (OSError, RuntimeError):
        return []

    matches = []
    for directory in directories:
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink() and entry.resolve(strict=True) == target:
                    matches.append(str(entry))
            except (OSError, RuntimeError):
                # A device can disappear while its links are being enumerated.
                continue
    return sorted(set(matches))


def discover_ports() -> list[dict[str, Any]]:
    """List serial ports and their USB metadata without opening a connection.

    ``stable_paths`` contains Linux ``/dev/serial/by-id`` and ``by-path``
    aliases when present. Metadata absent from pyserial is represented by None.
    """
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise ImportError("Serial port discovery requires pyserial.") from exc

    ports = []
    for port in list_ports.comports():
        device = port.device
        ports.append(
            {
                "device": device,
                "description": getattr(port, "description", None),
                "hwid": getattr(port, "hwid", None),
                "serial_number": getattr(port, "serial_number", None),
                "manufacturer": getattr(port, "manufacturer", None),
                "vid": getattr(port, "vid", None),
                "pid": getattr(port, "pid", None),
                "stable_paths": _stable_paths(device, _SERIAL_LINK_DIRS),
            }
        )
    return sorted(ports, key=lambda port: port["device"])


def identify_port(
    *,
    prompt: Callable[[str], Any] | None = None,
    sleep: Callable[[float], Any] | None = None,
    discover: Callable[[], list[dict[str, Any]]] | None = None,
) -> str:
    """Identify one USB serial port by asking the user to unplug its cable.

    The cable must be connected before this function is called. The function
    returns the device path when exactly one port disappears. Inject the three
    callbacks to use the same comparison in a GUI or in hardware-free tests.
    """
    prompt = input if prompt is None else prompt
    sleep = _sleep if sleep is None else sleep
    discover = discover_ports if discover is None else discover

    before = {port["device"] for port in discover()}
    prompt("Unplug the USB cable of the device to identify, then press Enter: ")
    sleep(0.5)
    after = {port["device"] for port in discover()}
    removed = before - after
    if len(removed) != 1:
        raise RuntimeError(
            "Expected exactly one serial port to disappear; "
            f"found {len(removed)}: {sorted(removed)}."
        )
    return removed.pop()


def discover_cameras(backend: str = "opencv") -> list[dict[str, Any]]:
    """Query LeRobot camera discovery for OpenCV or RealSense.

    Optional camera dependencies are imported only for the selected backend.
    OpenCV discovery briefly opens and releases candidate cameras to inspect
    their default profiles; it does not leave a camera connected.
    """
    if backend == "opencv":
        from lerobot.cameras.opencv import OpenCVCamera

        cameras = []
        for info in OpenCVCamera.find_cameras():
            camera = dict(info)
            device = camera.get("id")
            camera["stable_paths"] = (
                _stable_paths(device, _CAMERA_LINK_DIRS) if isinstance(device, str) else []
            )
            cameras.append(camera)
        return cameras
    if backend == "realsense":
        from lerobot.cameras.realsense import RealSenseCamera

        return [dict(info) for info in RealSenseCamera.find_cameras()]
    raise ValueError("Camera backend must be 'opencv' or 'realsense'.")
