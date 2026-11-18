"""Release held SO follower groups when a sustained manual push is inferred.

Feetech's load register reports signed motor effort, not measured external
force. A stationary baseline and debounce reduce false triggers, but callers
must let the arm settle before touching it. Arm joints and the gripper have
independent baselines and release independently.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from inspect import getattr_static
import logging
from math import isfinite
from numbers import Real
from statistics import median

from .feetech_safety import disable_torque_verified, is_supported_feetech_bus, verify_motor_faults
from .setup_backend import _resources


_SUPPORTED_MODELS = frozenset({"sts3215", "sts3250", "sts_series", "sms_series", "sm8512bl"})
_REQUIRED_REGISTERS = (
    "Present_Load", "Present_Position", "Goal_Position", "Torque_Enable", "Lock", "Status",
    "Min_Position_Limit", "Max_Position_Limit",
)
_BUS_METHODS = (
    "sync_read", "sync_write", "disable_torque", "enable_torque",
    "_read", "_write", "_is_comm_success",
)
_TIME_TOLERANCE = 1e-9
_POSITION_MAGNITUDE_MAX = (1 << 15) - 1
_MAX_RESUME_CORRECTION = 20
logger = logging.getLogger(__name__)
_FOLLOWER_MOTORS = frozenset({
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
})


class ManualResetOutOfRange(RuntimeError):
    """A released group must be moved back into range before resuming."""


@dataclass
class _Group:
    bus: object
    names: tuple[str, ...]
    label: str
    load_threshold: float
    released: bool = False
    samples: deque = field(default_factory=lambda: deque(maxlen=2048))
    baseline_load: dict[str, float] | None = None
    baseline_position: dict[str, float] | None = None
    triggered_since: dict[str, float] = field(default_factory=dict)

    def clear(self):
        self.released = False
        self.samples.clear()
        self.baseline_load = None
        self.baseline_position = None
        self.triggered_since.clear()


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


class TouchReset:
    """Manage independent manual reset of each native arm and gripper.

    Construction only inspects driver metadata. ``hold`` and ``poll`` are used
    after a deployment round stops; ``resume`` must complete before another
    round sends actions. Thresholds are raw servo units: load full scale is
    1000, and SO101 position uses 4096 ticks per revolution.
    """

    def __init__(
        self,
        robot,
        *,
        arm_load_threshold=100.0,
        gripper_load_threshold=100.0,
        load_duration=0.15,
        position_threshold=20.0,
        settle_time=0.5,
        baseline_time=0.5,
    ):
        arm_load_threshold = _number(arm_load_threshold, "arm_load_threshold", positive=True)
        gripper_load_threshold = _number(gripper_load_threshold, "gripper_load_threshold", positive=True)
        self.load_duration = _number(load_duration, "load_duration")
        self.position_threshold = _number(position_threshold, "position_threshold", positive=True)
        self.settle_time = _number(settle_time, "settle_time")
        self.baseline_time = _number(baseline_time, "baseline_time", positive=True)
        self._holding = False
        self._hold_started = None
        self._last_poll = None
        self._groups = []
        for resource in _resources(robot):
            motors = getattr(resource, "motors", None)
            if not isinstance(motors, Mapping) or not motors:
                continue
            self._check_bus(resource, motors)
            port = str(getattr(resource, "port", f"bus {len(self._groups) + 1}"))
            arm_names = tuple(name for name in motors if name != "gripper")
            if arm_names:
                self._groups.append(_Group(resource, arm_names, f"Arm ({port})", arm_load_threshold))
            if "gripper" in motors:
                self._groups.append(_Group(resource, ("gripper",), f"Gripper ({port})", gripper_load_threshold))
        if not self._groups:
            raise ValueError("Touch reset requires a supported native Feetech motor bus.")

    @staticmethod
    def _check_bus(bus, motors):
        if set(motors) != _FOLLOWER_MOTORS:
            raise ValueError(
                "Touch reset supports only SO100/SO101 follower motor layouts: "
                "shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, and gripper."
            )
        if not all(callable(getattr_static(bus, name, None)) for name in _BUS_METHODS):
            raise ValueError("Touch reset requires native motor read/write and torque control methods.")
        if getattr(bus, "protocol_version", None) != 0:
            raise ValueError("Touch reset supports only Feetech STS/SMS protocol 0 motor buses.")
        if not is_supported_feetech_bus(bus):
            raise ValueError("Touch reset requires native STS/SMS motor IDs and torque/status registers.")
        tables = getattr(bus, "model_ctrl_table", None)
        encodings = getattr(bus, "model_encoding_table", None)
        if not isinstance(tables, Mapping) or not isinstance(encodings, Mapping):
            raise ValueError("Touch reset requires native motor control and sign-encoding tables.")
        addresses = {}
        for name, motor in motors.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Touch reset requires named motors.")
            model = getattr(motor, "model", None)
            if not isinstance(model, str) or model not in _SUPPORTED_MODELS:
                raise ValueError(f"Touch reset does not support motor {name!r} with model {model!r}.")
            table = tables.get(model)
            encoding = encodings.get(model)
            if not isinstance(table, Mapping):
                raise ValueError(f"Touch reset does not support motor {name!r} with model {model!r}.")
            if not isinstance(encoding, Mapping) or encoding.get("Present_Load") != 10:
                raise ValueError(f"Touch reset requires signed STS/SMS load decoding for motor {name!r}.")
            if encoding.get("Goal_Position") != 15:
                raise ValueError(f"Touch reset requires signed STS/SMS goal position encoding for motor {name!r}.")
            for register in _REQUIRED_REGISTERS:
                address = table.get(register)
                if (
                    not isinstance(address, (tuple, list)) or len(address) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in address)
                    or address[0] < 0 or address[1] <= 0
                ):
                    raise ValueError(f"Touch reset requires a valid {register} register for motor {name!r}.")
                address = tuple(address)
                if register in addresses and addresses[register] != address:
                    raise ValueError(f"Touch reset requires matching {register} registers on each motor bus.")
                addresses[register] = address

    @staticmethod
    def _read(bus, register, names):
        try:
            values = bus.sync_read(register, list(names), normalize=False)
        except Exception as exc:
            raise RuntimeError(f"Touch reset could not read {register} on {getattr(bus, 'port', 'motor bus')!r}.") from exc
        if not isinstance(values, Mapping):
            raise ValueError(f"Touch reset received invalid {register} data.")
        result = {}
        for name in names:
            value = values.get(name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
                raise ValueError(f"Touch reset requires a finite {register} value for motor {name!r}.")
            result[name] = float(value)
        return result

    @staticmethod
    def _time(now):
        if isinstance(now, bool) or not isinstance(now, Real) or not isfinite(float(now)):
            raise ValueError("Touch reset time must be finite.")
        return float(now)

    @classmethod
    def _verify_targets(cls, group, positions):
        # Native sync_write is a broadcast with no per-motor acknowledgement.
        # Reading the goal register avoids enabling torque against a stale goal
        # if a target packet was dropped. Native raw writes convert to int.
        goals = cls._read(group.bus, "Goal_Position", group.names)
        for name, position in positions.items():
            if goals[name] != int(position):
                raise RuntimeError(
                    f"Touch reset target read-back mismatch for {group.label}, motor {name!r}: "
                    f"expected {int(position)}, got {goals[name]}."
                )

    def hold(self, now):
        """Freeze fresh arm positions without changing the gripper's grasp."""
        now = self._time(now)
        if any(group.released for group in self._groups):
            raise RuntimeError("Resume touch reset before holding a released arm or gripper again.")
        # Validate every read before changing any target, including on dual arms.
        targets = []
        for group in self._groups:
            if group.names != ("gripper",):
                targets.append((group, self._read(group.bus, "Present_Position", group.names)))
        for group, positions in targets:
            group.bus.sync_write("Goal_Position", positions, normalize=False)
        for group, positions in targets:
            self._verify_targets(group, positions)
        for group in self._groups:
            group.clear()
        self._holding = True
        self._hold_started = now
        self._last_poll = now

    def _baseline(self, group, now, loads, positions):
        group.samples.append((now, loads, positions))
        cutoff = now - self.baseline_time
        # Keep one sample at/before the boundary so a sampled window can cover
        # the full duration even when polls do not align with its endpoints.
        while len(group.samples) > 2 and group.samples[1][0] <= cutoff:
            group.samples.popleft()
        if len(group.samples) < 3 or now - group.samples[0][0] + _TIME_TOLERANCE < self.baseline_time:
            return False
        for name in group.names:
            load_values = [sample[1][name] for sample in group.samples]
            position_values = [sample[2][name] for sample in group.samples]
            if (
                max(load_values) - min(load_values) > group.load_threshold / 2
                or max(position_values) - min(position_values) > self.position_threshold / 2
            ):
                return False
        group.baseline_load = {
            name: median(sample[1][name] for sample in group.samples) for name in group.names
        }
        group.baseline_position = {
            name: median(sample[2][name] for sample in group.samples) for name in group.names
        }
        group.samples.clear()
        return True

    def poll(self, now):
        """Return status messages; independently release sustained touched groups."""
        if not self._holding:
            return []
        now = self._time(now)
        if now < self._last_poll:
            raise ValueError("Touch reset time must not move backwards.")
        self._last_poll = now
        if now - self._hold_started + _TIME_TOLERANCE < self.settle_time:
            return []
        # Read and validate all still-held groups before releasing any group.
        readings = []
        for group in self._groups:
            if not group.released:
                loads = self._read(group.bus, "Present_Load", group.names)
                positions = self._read(group.bus, "Present_Position", group.names)
                readings.append((group, loads, positions))
        messages = []
        for group, loads, positions in readings:
            if group.baseline_load is None:
                if self._baseline(group, now, loads, positions):
                    messages.append(f"{group.label} touch reset ready; a sustained push unlocks this group.")
                continue
            for name in group.names:
                load_change = abs(loads[name] - group.baseline_load[name])
                position_change = abs(positions[name] - group.baseline_position[name])
                if load_change >= group.load_threshold or position_change >= self.position_threshold:
                    since = group.triggered_since.setdefault(name, now)
                    if now - since + _TIME_TOLERANCE >= self.load_duration:
                        faults = disable_torque_verified(group.bus, list(group.names))
                        group.released = True
                        group.triggered_since.clear()
                        detail = "; overload reported, torque-off verified" if faults else ""
                        messages.append(f"{group.label} unlocked for manual reset ({name}{detail}).")
                        break
                else:
                    group.triggered_since.pop(name, None)
        return messages

    def resume(self):
        """Stage fresh targets for every released group, then enable its torque."""
        readings = []
        for group in self._groups:
            if group.released:
                readings.append((
                    group,
                    self._read(group.bus, "Present_Position", group.names),
                    self._read(group.bus, "Min_Position_Limit", group.names),
                    self._read(group.bus, "Max_Position_Limit", group.names),
                ))
        validated = []
        for group, positions, minimums, maximums in readings:
            raw_positions, raw_minimums, raw_maximums = {}, {}, {}
            for name in group.names:
                position, minimum, maximum = positions[name], minimums[name], maximums[name]
                if any(value != int(value) for value in (position, minimum, maximum)):
                    raise ValueError(f"Touch reset requires integral raw positions and limits for motor {name!r}.")
                position, minimum, maximum = int(position), int(minimum), int(maximum)
                if not -_POSITION_MAGNITUDE_MAX <= position <= _POSITION_MAGNITUDE_MAX:
                    raise ValueError(f"Touch reset received an invalid raw position for motor {name!r}: {position}.")
                if not 0 <= minimum < maximum <= _POSITION_MAGNITUDE_MAX:
                    raise ValueError(
                        f"Touch reset received invalid position limits for motor {name!r}: "
                        f"minimum={minimum}, maximum={maximum}."
                    )
                raw_positions[name], raw_minimums[name], raw_maximums[name] = position, minimum, maximum
            validated.append((group, raw_positions, raw_minimums, raw_maximums))
        targets = []
        corrections = []
        for group, positions, minimums, maximums in validated:
            bounded = {}
            for name in group.names:
                position, minimum, maximum = positions[name], minimums[name], maximums[name]
                target = min(maximum, max(minimum, position))
                correction = abs(target - position)
                if correction > _MAX_RESUME_CORRECTION:
                    raise ManualResetOutOfRange(
                        f"Cannot resume {group.label}, motor {name!r}: current position {position} "
                        f"is outside limits [{minimum}, {maximum}] by {correction} raw ticks. "
                        "Torque remains off. Manually move the released group into its safe range, "
                        "then press Right to retry."
                    )
                bounded[name] = target
                if correction:
                    corrections.append((group, name, position, target, correction))
            targets.append((group, bounded))
        # Every released group gets its current target before any torque is
        # enabled. A failed read/write therefore cannot enable another group.
        for group, positions in targets:
            group.bus.sync_write("Goal_Position", positions, normalize=False)
        for group, positions in targets:
            self._verify_targets(group, positions)
        for group, name, position, target, correction in corrections:
            logger.warning(
                "Touch reset adjusted %s, motor %r, from %s to hardware limit %s "
                "(%s raw ticks); the target read-back was verified before torque enabling.",
                group.label, name, position, target, correction,
            )
        # A fresh target can clear a latched overload. Check each status packet
        # after staging; sync_read alone discards native servo error bits.
        # A fault on either group must prevent torque on every released group.
        for group, _ in targets:
            verify_motor_faults(group.bus, list(group.names))
        for group, _ in targets:
            group.bus.enable_torque(list(group.names))
        for group in self._groups:
            group.clear()
        self._holding = False
        self._hold_started = None
        self._last_poll = None
