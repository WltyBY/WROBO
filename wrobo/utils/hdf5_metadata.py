"""Shared HDF5 attribute parsing and validation for recording and tools."""

import json

import h5py
import numpy as np

from wrobo.training.dataset.instruction import parse_instruction_segments, format_instruction_metadata


def _normalize_instruction(instruction):
    if isinstance(instruction, dict):
        instruction = json.dumps(instruction, ensure_ascii=False)
    segments = parse_instruction_segments(instruction)
    episode = next((ref for start, _, ref in segments if start is None), None)
    frames = [
        dict(start=start, end=end, **(ref if isinstance(ref, dict) else {"task_id": ref}))
        for start, end, ref in segments if start is not None
    ]
    return format_instruction_metadata(episode, frames)

def _attribute_value(value):
    if value is None:
        raise ValueError("HDF5 attributes cannot store null; use a quoted string if needed.")
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.dtype.kind == "U":
            if not all(isinstance(item, str) for item in np.asarray(value, dtype=object).flat):
                raise ValueError("Attribute arrays must contain only numbers or only strings.")
            return np.asarray(value, dtype=h5py.string_dtype(encoding="utf-8"))
        if array.dtype.kind == "O":
            raise ValueError("Attribute arrays must contain only numbers or only strings.")
        return array
    return value

def _validate_attribute_name(name, *, allow_shell_quotes=False):
    if not isinstance(name, str) or not name.strip() or "\x00" in name:
        raise ValueError("Attribute names must be nonempty strings without NUL characters.")
    stripped = name.strip()
    if not allow_shell_quotes and (stripped[0] in "\"'" or stripped[-1] in "\"'"):
        raise ValueError(
            f"Attribute name {name!r} contains a literal shell quote. "
            "Windows CMD does not use single quotes to quote arguments; "
            "use double quotes around NAME=VALUE and escape JSON double quotes "
            r"with \". See wrobo/documents/teleoperation.md."
        )

def _prepare_attributes(attributes: dict) -> dict:
    """Validate all values in an in-memory HDF5 file before changing real files."""
    if not isinstance(attributes, dict) or not attributes:
        raise ValueError("Provide at least one attribute to update.")
    prepared = {}
    with h5py.File("attribute-validation.h5", "w", driver="core", backing_store=False) as scratch:
        for name, value in attributes.items():
            _validate_attribute_name(name)
            try:
                value = _normalize_instruction(value) if name == "instruction" else _attribute_value(value)
                scratch.attrs[name] = value
            except (TypeError, ValueError, OverflowError, OSError) as exc:
                raise ValueError(f"Invalid value for attribute {name!r}: {exc}") from exc
            prepared[name] = value
    return prepared

def _parse_assignment(assignment):
    name, separator, text = assignment.partition("=")
    if not separator:
        raise ValueError("--attr requires NAME=VALUE.")
    _validate_attribute_name(name)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text
    return name, value
