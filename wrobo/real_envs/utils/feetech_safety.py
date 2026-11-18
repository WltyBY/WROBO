"""Verify torque-off without treating an overload status as a lost packet.

LeRobot's acknowledged writes raise for any servo alarm, including an overload
reported after a successful torque-off write. Strict raw verification allows
only the overload bit, and only while writing and verifying torque-off. Shutdown
can request that verification without relying on the SDK's error text.
Torque enabling, position targets, and protection settings are never changed.
"""

from collections.abc import Mapping, Sequence
from inspect import getattr_static
import logging
from numbers import Integral
import re


logger = logging.getLogger(__name__)
_OVERLOAD = 0x20
_OVERLOAD_MESSAGE = "[RxPacketError] Overload error!"
_MODELS = frozenset({"sts3215", "sts3250", "sts_series", "sms_series", "sm8512bl"})
_REGISTERS = {"Torque_Enable": (40, 1), "Lock": (55, 1), "Status": (65, 1)}
_RAW_METHODS = ("_read", "_write", "_is_comm_success")


def _metadata(bus):
    """Inspect a known STS/SMS layout without opening or reading its port."""
    if getattr(bus, "protocol_version", None) != 0:
        raise ValueError("Verified torque-off requires a Feetech STS/SMS protocol 0 bus.")
    if not all(callable(getattr_static(bus, name, None)) for name in _RAW_METHODS):
        raise ValueError("Verified torque-off requires native raw motor read/write methods.")
    motors = getattr(bus, "motors", None)
    tables = getattr(bus, "model_ctrl_table", None)
    if not isinstance(motors, Mapping) or not motors or not isinstance(tables, Mapping):
        raise ValueError("Verified torque-off requires native motor metadata and control tables.")
    ids = set()
    for name, motor in motors.items():
        motor_id = getattr(motor, "id", None)
        model = getattr(motor, "model", None)
        if (
            not isinstance(name, str) or not name
            or isinstance(motor_id, bool) or not isinstance(motor_id, Integral)
            or not 1 <= motor_id <= 253 or motor_id in ids
            or not isinstance(model, str) or model not in _MODELS
        ):
            raise ValueError("Verified torque-off requires named STS/SMS motors with unique IDs.")
        ids.add(motor_id)
        table = tables.get(model)
        if not isinstance(table, Mapping):
            raise ValueError(f"Verified torque-off requires a control table for motor {name!r}.")
        for register, expected in _REGISTERS.items():
            address = table.get(register)
            if (
                not isinstance(address, (tuple, list)) or len(address) != 2
                or any(isinstance(value, bool) or not isinstance(value, Integral) for value in address)
                or tuple(address) != expected
            ):
                raise ValueError(f"Verified torque-off requires the STS/SMS {register} register.")
    return motors


def is_supported_feetech_bus(bus) -> bool:
    """Return whether metadata supports strict raw verification; perform no I/O."""
    try:
        _metadata(bus)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _selected(bus, motors):
    known = _metadata(bus)
    if motors is None:
        requested = list(known)
    elif isinstance(motors, (str, Integral)) and not isinstance(motors, bool):
        requested = [motors]
    elif isinstance(motors, Sequence) and not isinstance(motors, (str, bytes, bytearray)):
        requested = list(motors)
    else:
        raise ValueError("Motors must be names, IDs, or a sequence of names/IDs.")
    names = []
    for item in requested:
        if isinstance(item, str) and item in known:
            name = item
        elif isinstance(item, Integral) and not isinstance(item, bool):
            name = next((name for name, motor in known.items() if motor.id == item), None)
            if name is None:
                raise ValueError(f"Unknown motor ID {item!r}.")
        else:
            raise ValueError(f"Unknown motor {item!r}.")
        if name not in names:
            names.append(name)
    return names


def _retries(num_retry):
    if isinstance(num_retry, bool) or not isinstance(num_retry, Integral) or num_retry < 0:
        raise ValueError("num_retry must be a non-negative integer.")


def _location(bus, name):
    return f"motor {name!r} (id={bus.motors[name].id}, port={getattr(bus, 'port', 'motor bus')!r})"


def _check_reply(bus, name, register, comm, status, *, allow_overload):
    location = _location(bus, name)
    if isinstance(comm, bool) or not isinstance(comm, Integral) or not bus._is_comm_success(comm):
        raise ConnectionError(f"Could not verify {register} for {location}: communication result {comm!r}.")
    allowed = (0, _OVERLOAD) if allow_overload else (0,)
    if isinstance(status, bool) or not isinstance(status, Integral) or status not in allowed:
        raise RuntimeError(f"Motor fault while verifying {register} for {location}: packet status {status!r}.")
    return int(status)


def verify_torque_disabled(bus, motors=None, *, num_retry=0) -> dict[str, int]:
    """Write torque-off and verify both off registers using strict raw replies.

    All selected motors are attempted even if one cannot be verified. Such a
    failure remains fatal; callers may then close the port without claiming the
    unverified motor was released. Only successful communication and status 0
    or the pure overload bit are accepted; unknown and combined faults remain
    fatal. Returned overloads do not mean that a motor is healthy to re-enable.
    """
    _retries(num_retry)
    names = _selected(bus, motors)
    if not names:
        raise ValueError("Cannot verify torque-off without selecting a motor.")
    overloads = {}
    failures = []
    for name in names:
        motor_id = bus.motors[name].id
        try:
            # Verify torque-off before unlocking EEPROM writes with Lock=0.
            for register in ("Torque_Enable", "Lock"):
                address, length = _REGISTERS[register]
                comm, status = bus._write(
                    address, length, motor_id, 0, num_retry=num_retry, raise_on_error=False,
                )
                flag = _check_reply(bus, name, register, comm, status, allow_overload=True)
                if flag:
                    overloads[name] = flag
                value, comm, status = bus._read(
                    address, length, motor_id, num_retry=num_retry, raise_on_error=False,
                )
                flag = _check_reply(bus, name, register, comm, status, allow_overload=True)
                if flag:
                    overloads[name] = flag
                if isinstance(value, bool) or not isinstance(value, Integral) or value != 0:
                    raise RuntimeError(
                        f"Torque-off read-back mismatch for {_location(bus, name)}, {register}: "
                        f"expected 0, got {value!r}. The motor has not been verified as released."
                    )
        except Exception as exc:
            failures.append(exc)
    if failures:
        raise failures[0]
    return overloads


def recover_torque_disable(bus, error, motors=None, *, num_retry=0) -> dict[str, int]:
    """Recover only a native overload acknowledgement after strict verification."""
    if not isinstance(error, RuntimeError) or not str(error).endswith(_OVERLOAD_MESSAGE):
        raise error
    overloads = verify_torque_disabled(bus, motors, num_retry=num_retry)
    names = _selected(bus, motors)
    match = re.search(r"\bid_=(\d+)\b", str(error))
    if match is not None:
        original_id = int(match.group(1))
        overloads.update({name: _OVERLOAD for name in names if bus.motors[name].id == original_id})
    logger.warning(
        "Feetech reported overload while disabling torque on %s; Torque_Enable=0 and Lock=0 "
        "were verified for motors %s. Protection settings were not changed.",
        getattr(bus, "port", "motor bus"), ", ".join(names),
    )
    return overloads


def disable_torque_verified(bus, motors=None, *, num_retry=0) -> dict[str, int]:
    """Use native torque-off, recovering only a verified overload acknowledgement."""
    _retries(num_retry)
    try:
        bus.disable_torque(motors, num_retry=num_retry)
    except RuntimeError as exc:
        return recover_torque_disable(bus, exc, motors, num_retry=num_retry)
    return {}


def verify_motor_faults(bus, motors) -> None:
    """Reject any outstanding motor fault before a caller enables torque."""
    names = _selected(bus, motors)
    address, length = _REGISTERS["Status"]
    for name in names:
        value, comm, status = bus._read(address, length, bus.motors[name].id, raise_on_error=False)
        try:
            _check_reply(bus, name, "Status", comm, status, allow_overload=False)
            if isinstance(value, bool) or not isinstance(value, Integral) or value != 0:
                raise RuntimeError(f"Status register still reports a fault: {value!r}.")
        except (ConnectionError, RuntimeError) as exc:
            raise RuntimeError(
                f"Cannot enable torque for {_location(bus, name)}: {exc} "
                "Remove any obstruction or manual force and inspect the motor before retrying. "
                "Torque remains off; protection settings were not changed."
            ) from exc
