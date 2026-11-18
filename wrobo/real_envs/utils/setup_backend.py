"""LeRobot adapter and hardware-independent initialization orchestration."""

from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import importlib
from inspect import getattr_static
import logging
from pathlib import Path
from typing import Any, Callable, Protocol

from .setup_config import HardwareSetupConfig


logger = logging.getLogger(__name__)

# Only this adapter knows LeRobot's names. The CLI and saved settings use wrobo
# names, while connection, motor setup, and calibration stay in LeRobot drivers.
_DRIVERS = {
    "robot": {
        "so100": ("so100_follower", "lerobot.robots.so_follower"),
        "so101": ("so101_follower", "lerobot.robots.so_follower"),
        "koch": ("koch_follower", "lerobot.robots.koch_follower"),
        "omx": ("omx_follower", "lerobot.robots.omx_follower"),
        "bi_so": ("bi_so_follower", "lerobot.robots.bi_so_follower"),
        "openarm": ("openarm_follower", "lerobot.robots.openarm_follower"),
        "bi_openarm": ("bi_openarm_follower", "lerobot.robots.bi_openarm_follower"),
    },
    "teleop": {
        "so100": ("so100_leader", "lerobot.teleoperators.so_leader"),
        "so101": ("so101_leader", "lerobot.teleoperators.so_leader"),
        "koch": ("koch_leader", "lerobot.teleoperators.koch_leader"),
        "omx": ("omx_leader", "lerobot.teleoperators.omx_leader"),
        "bi_so": ("bi_so_leader", "lerobot.teleoperators.bi_so_leader"),
        "openarm": ("openarm_leader", "lerobot.teleoperators.openarm_leader"),
        "bi_openarm": ("bi_openarm_leader", "lerobot.teleoperators.bi_openarm_leader"),
    },
}

_CAMERA_MODULES = {
    "opencv": "lerobot.cameras.opencv",
    "intelrealsense": "lerobot.cameras.realsense",
    "reachy2_camera": "lerobot.cameras.reachy2_camera",
    "zmq": "lerobot.cameras.zmq",
}


def available_device_types() -> list[dict[str, str]]:
    """List supported names without importing SDKs or inspecting hardware."""
    return [
        {"kind": kind, "type": name, "driver_type": driver_type}
        for kind, drivers in _DRIVERS.items()
        for name, (driver_type, _) in drivers.items()
    ]


def _resolve_driver(config: HardwareSetupConfig) -> tuple[str, str]:
    for name, (driver_type, module) in _DRIVERS[config.kind].items():
        if config.device_type in (name, driver_type):
            return driver_type, config.module or module
    # Read old settings or use an explicitly configured third-party adapter.
    if config.module is not None:
        return config.device_type, config.module
    raise ValueError(
        f"Unsupported {config.kind} type {config.device_type!r}. "
        "Run 'python -m wrobo.real_envs.setup types' to list supported names."
    )


class SetupBackend(Protocol):
    """Drivers implement these operations without exposing motor protocols."""

    def calibration_files(self) -> list[Path]: ...
    def connect(self) -> None: ...
    def calibrate(self) -> None: ...
    def setup_motors(self) -> None: ...
    def describe(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


def _import_camera_configs(parameters: dict[str, Any]) -> None:
    """Register only camera types actually mentioned in device parameters."""
    for key, value in parameters.items():
        if key == "cameras" and isinstance(value, dict):
            for camera in value.values():
                if isinstance(camera, dict):
                    module = _CAMERA_MODULES.get(camera.get("type"))
                    if module:
                        importlib.import_module(module)
        elif isinstance(value, dict):
            _import_camera_configs(value)


def _load_native_config(config: HardwareSetupConfig):
    """Decode registered robot/teleoperator parameters without creating hardware."""
    driver_type, driver_module = _resolve_driver(config)
    try:
        draccus = importlib.import_module("draccus")
        importlib.import_module(driver_module)
        _import_camera_configs(config.parameters)
        if config.kind == "robot":
            registry_module = importlib.import_module("lerobot.robots.config")
            registry = registry_module.RobotConfig
        else:
            registry_module = importlib.import_module("lerobot.teleoperators.config")
            registry = registry_module.TeleoperatorConfig
    except ImportError as exc:
        raise ImportError(
            "Initialization requires LeRobot, draccus, and the dependencies of "
            f"{driver_module}. Install the matching LeRobot hardware extra."
        ) from exc

    try:
        registry.get_choice_class(driver_type)
    except KeyError as exc:
        raise ValueError(
            f"{driver_type!r} is not registered after importing {driver_module!r}. "
            "Check that the installed LeRobot version supports this device."
        ) from exc
    try:
        native = draccus.decode(registry, {"type": driver_type, **config.parameters})
    except Exception as exc:
        raise ValueError(f"Invalid parameters for {config.device_type!r}: {exc}") from exc
    return draccus, native


def normalize_setup_config(config: HardwareSetupConfig) -> HardwareSetupConfig:
    """Validate native parameters and include driver defaults in the saved YAML."""
    draccus, native = _load_native_config(config)
    parameters = dict(draccus.encode(native))
    # Some LeRobot classes have several registered names; preserve the chosen one.
    parameters.pop("type", None)
    return HardwareSetupConfig(
        module=config.module,
        device_type=config.device_type,
        parameters=parameters,
        kind=config.kind,
        extra_sections=config.extra_sections,
    )


def _resources(root):
    """Visit driver-owned disconnectable resources, including composite devices."""
    pending = [root]
    visited = set()
    while pending:
        resource = pending.pop()
        if id(resource) in visited:
            continue
        visited.add(id(resource))
        yield resource
        for value in getattr(resource, "__dict__", {}).values():
            values = value.values() if isinstance(value, dict) else (
                value if isinstance(value, (list, tuple)) else (value,)
            )
            for child in values:
                if callable(getattr_static(child, "disconnect", None)):
                    pending.append(child)


def calibration_diagnostics(device) -> list[str]:
    """Read driver-owned buses and report calibration differences; never write."""
    details = []
    for bus in _resources(device):
        if not callable(getattr_static(bus, "read_calibration", None)):
            continue
        read_calibration = getattr(bus, "read_calibration", None)
        expected = getattr(bus, "calibration", None)
        if not callable(read_calibration) or not isinstance(expected, Mapping):
            continue
        port = getattr(bus, "port", "not specified")
        protocol = getattr(bus, "protocol_version", None)
        details.append(f"Motor calibration read-back: port={port!r}, protocol={protocol!r}")
        try:
            actual = read_calibration()
        except Exception as exc:
            details.append(f"  Could not read motor calibration: {exc}")
            continue
        if not isinstance(actual, Mapping):
            details.append("  Driver did not return a motor calibration mapping.")
            continue
        missing = set(actual) - set(expected)
        extra = set(expected) - set(actual)
        if missing:
            details.append(f"  JSON missing motor entries: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"  JSON has unexpected motor entries: {', '.join(sorted(extra))}")
        fields_to_compare = ("range_min", "range_max")
        if protocol != 1:
            fields_to_compare += ("homing_offset",)
        differences = set(missing)
        compared_fields = 0
        absent = object()
        for motor in sorted(set(expected) & set(actual)):
            for name in fields_to_compare:
                def value(record):
                    return record.get(name, absent) if isinstance(record, Mapping) else getattr(record, name, absent)

                wanted, found = value(expected[motor]), value(actual[motor])
                if wanted is absent or found is absent:
                    continue
                compared_fields += 1
                if wanted != found:
                    differences.add(motor)
                    details.append(f"  {motor}.{name}: JSON={wanted!r}, motor={found!r}")
        if not differences and not extra:
            if compared_fields:
                details.append("  Compared limits and offsets match on read-back; the driver reported uncalibrated.")
            else:
                details.append("  Driver exposes no comparable limit/offset fields for read-back diagnostics.")
        # State registers help diagnose a failed write. Read only fields that
        # the driver's model control table actually supports.
        motors = getattr(bus, "motors", {})
        control_tables = getattr(bus, "model_ctrl_table", {})
        read = getattr(bus, "read", None)
        if not callable(read) or not isinstance(motors, Mapping) or not isinstance(control_tables, Mapping):
            continue
        for motor in sorted(differences):
            definition = motors.get(motor)
            model = getattr(definition, "model", None)
            table = control_tables.get(model, {})
            if not isinstance(table, Mapping):
                continue
            state = []
            for register in ("Torque_Enable", "Lock", "Phase", "Angular_Resolution"):
                if register in table:
                    try:
                        state.append(f"{register}={read(register, motor, normalize=False)!r}")
                    except Exception as exc:
                        state.append(f"{register}=unavailable ({exc})")
            if state:
                details.append(f"  {motor} state: {', '.join(state)}")
    return details


def _feature_description(value):
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, dict):
        return {str(key): _feature_description(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_feature_description(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _disable_cameras(config) -> None:
    """Keep arm calibration independent of camera availability.

    Operate only on the newly decoded runtime configuration. The wrobo settings
    retain all camera parameters for export and later deployment.
    """
    if not is_dataclass(config):
        return
    for entry in fields(config):
        value = getattr(config, entry.name)
        if entry.name == "cameras" and isinstance(value, dict):
            setattr(config, entry.name, {})
        elif is_dataclass(value):
            _disable_cameras(value)
        elif isinstance(value, dict):
            for child in value.values():
                _disable_cameras(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                _disable_cameras(child)


class LeRobotSetupBackend:
    """Adapt LeRobot's public lifecycle; motor details stay inside its driver."""

    def __init__(self, config: HardwareSetupConfig, *, enable_cameras: bool = False):
        self.config = config
        _, self.native_config = _load_native_config(config)
        if not enable_cameras:
            _disable_cameras(self.native_config)
        self._device = None

    @property
    def device(self):
        if self._device is None:
            if self.config.kind == "robot":
                module = importlib.import_module("lerobot.robots.utils")
                self._device = module.make_robot_from_config(self.native_config)
            else:
                module = importlib.import_module("lerobot.teleoperators.utils")
                self._device = module.make_teleoperator_from_config(self.native_config)
        return self._device

    def calibration_files(self) -> list[Path]:
        paths = {self.config.calibration_file}
        for resource in _resources(self.device):
            path = getattr(resource, "calibration_fpath", None)
            if path is not None:
                paths.add(Path(path))
        return sorted(paths, key=str)

    def connect(self) -> None:
        # Let the driver configure its motors, but leave calibration to the next
        # explicit call. Cameras were removed only from this runtime config.
        self.device.connect(calibrate=False)

    def calibrate(self) -> None:
        device = self.device
        # SO follower connect/configure re-enables torque and locks EEPROM.
        # Its existing-JSON branch does not unlock before writing. Explicit
        # calibration must keep native motor buses disabled/unlocked, including
        # child buses of composite devices, rather than restoring torque here.
        for resource in _resources(device):
            if all(callable(getattr_static(resource, name, None)) for name in (
                "read_calibration", "write_calibration", "disable_torque",
            )):
                resource.disable_torque()
        device.calibrate()

    def setup_motors(self) -> None:
        setup = getattr(self.device, "setup_motors", None)
        if not callable(setup):
            raise ValueError(f"{self.config.device_type!r} does not support setup_motors().")
        # Drivers perform their own per-motor connections; do not connect the whole robot.
        setup()

    def describe(self) -> dict[str, Any]:
        device = self.device
        return {
            "driver": f"{type(device).__module__}.{type(device).__name__}",
            "name": getattr(device, "name", self.config.device_type),
            "id": device.id,
            "action_features": _feature_description(getattr(device, "action_features", {})),
            "observation_features": _feature_description(
                getattr(device, "observation_features", {})
            ),
            "is_calibrated": bool(device.is_calibrated),
            "calibration_files": [
                str(path) for path in self.calibration_files() if path.is_file()
            ],
        }

    def close(self) -> None:
        if self._device is None:
            return
        # A failed camera connection can leave a motor bus open while the overall
        # device reports False. Disconnect the device, then any remaining children.
        resources = list(_resources(self._device))
        errors = []
        recovered_buses = set()
        for resource in resources:
            try:
                if getattr(resource, "is_connected", False):
                    resource.disconnect()
            except Exception as exc:
                from .feetech_safety import (
                    is_supported_feetech_bus,
                    verify_torque_disabled,
                )

                # SDK error text can be empty for an unrecognized reply. During
                # shutdown, independently attempt torque-off and verify strict
                # raw replies instead of inferring success from that text.
                if is_supported_feetech_bus(resource):
                    try:
                        faults = verify_torque_disabled(resource, num_retry=5)
                    except Exception as recovery_error:
                        # Keep the native diagnostic as well as numeric raw
                        # verification details, even if the SDK text is empty.
                        errors.append((resource, exc))
                        errors.append((resource, recovery_error))
                    else:
                        recovered_buses.add(id(resource))
                        logger.warning(
                            "Native disconnect failed on %r (%s: %s); torque off "
                            "and EEPROM unlock verified for every motor (faults: %s). "
                            "Protection settings were not changed.",
                            getattr(resource, "port", "motor bus"), type(exc).__name__, exc, faults,
                        )
                    finally:
                        # Even failed verification must not leak the serial
                        # handle. Keep its torque-off error fatal after closing.
                        try:
                            resource.disconnect(disable_torque=False)
                        except Exception as disconnect_error:
                            errors.append((resource, disconnect_error))
                else:
                    errors.append((resource, exc))
        remaining_errors = []
        for resource, exc in errors:
            descendants = list(_resources(resource))
            # A composite driver's disconnect can propagate the same bus
            # overload before its children get cleaned up individually. Accept
            # that parent error only after a descendant bus was verified off
            # and every owned connection has actually closed.
            recovered_child = any(
                child is not resource and id(child) in recovered_buses
                for child in descendants
            )
            if not (
                _is_overload_error(exc) and recovered_child
                and all(_resource_is_closed(child) for child in descendants)
            ):
                remaining_errors.append((resource, exc))
        errors = remaining_errors
        if errors:
            details = "; ".join(
                f"{type(resource).__name__}"
                + (f" (port={getattr(resource, 'port')!r})" if hasattr(resource, "port") else "")
                + f": {type(exc).__name__}: {exc}"
                for resource, exc in errors
            )
            raise RuntimeError(f"Failed to disconnect hardware resources: {details}") from errors[0][1]


def _is_overload_error(error) -> bool:
    return isinstance(error, RuntimeError) and str(error).endswith("[RxPacketError] Overload error!")


def _resource_is_closed(resource) -> bool:
    try:
        return not getattr(resource, "is_connected", False)
    except Exception:
        return False


@contextmanager
def _managed_backend(backend: SetupBackend):
    try:
        yield backend
    except BaseException:
        try:
            backend.close()
        except Exception:
            logger.exception("Resource cleanup also failed after operation failed.")
        raise
    else:
        backend.close()


def calibrate_hardware(
    config: HardwareSetupConfig,
    *,
    allow_existing: bool = False,
    backend_factory: Callable[[HardwareSetupConfig], SetupBackend] | None = None,
) -> dict[str, Any]:
    """Run driver calibration and return metadata, always closing its resources."""
    if config.calibration_file.exists() and not allow_existing:
        raise FileExistsError(
            f"Calibration exists: {config.calibration_file}. "
            "Use --allow-existing to let the driver reuse or replace it."
        )
    backend = (backend_factory or LeRobotSetupBackend)(config)
    with _managed_backend(backend):
        existing = [path for path in backend.calibration_files() if path.exists()]
        if existing and not allow_existing:
            raise FileExistsError(
                f"Calibration exists: {existing[0]}. "
                "Use --allow-existing to let the driver reuse or replace it."
            )
        backend.connect()
        backend.calibrate()
        description = backend.describe()
        if not description.get("is_calibrated", False):
            details = calibration_diagnostics(backend.device) if hasattr(backend, "device") else []
            raise RuntimeError(
                "Calibration write-back verification failed: is_calibrated is false. "
                "The calibration has not been accepted; initialization metadata was not updated."
                + ("\n" + "\n".join(details) if details else "")
            )
        return description


def setup_hardware_motors(
    config: HardwareSetupConfig,
    *,
    backend_factory: Callable[[HardwareSetupConfig], SetupBackend] | None = None,
) -> None:
    """Delegate motor ID/baudrate setup to a capable driver and close afterwards."""
    backend = (backend_factory or LeRobotSetupBackend)(config)
    with _managed_backend(backend):
        backend.setup_motors()
