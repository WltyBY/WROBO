"""Portable configuration files for real hardware setup.

The user-facing file records a robot or teleoperator type and its parameters.
Driver selection is internal to wrobo. The calibration directory and ID follow
LeRobot's JSON file convention; motor calibration belongs to the driver.
"""

from __future__ import annotations

import copy
import keyword
import os
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


_RESERVED_SECTIONS = frozenset({"backend", "robot", "teleop"})


def _yaml_value(value: Any) -> Any:
    """Copy values into types accepted by YAML's safe serializer."""
    if isinstance(value, Enum):
        return _yaml_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {_yaml_value(key): _yaml_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_value(item) for item in value]
    return copy.deepcopy(value)


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Hardware setup configuration requires PyYAML.") from exc
    return yaml


@dataclass(kw_only=True)
class HardwareSetupConfig:
    """wrobo settings used to construct one hardware device.

    ``parameters`` contains the selected driver's configuration fields. Only
    ``id`` and ``calibration_dir`` are shared by this setup workflow. A new
    configuration can retain a relative calibration directory; loading a file
    resolves that directory relative to the file rather than the working
    directory.

    ``module`` is an internal compatibility override for older setup files.
    It is optional and never exported to the user-facing configuration.
    """

    device_type: str
    parameters: dict[str, Any]
    kind: str = "robot"
    module: str | None = None
    extra_sections: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.module is not None and (
            not isinstance(self.module, str)
            or not self.module
            or any(
                not component.isidentifier() or keyword.iskeyword(component)
                for component in self.module.split(".")
            )
        ):
            raise ValueError("Internal module override must be a valid dotted Python module name.")
        if (
            not isinstance(self.device_type, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.device_type) is None
        ):
            raise ValueError("Device type must be a nonempty name suitable for a directory.")
        if self.kind not in ("robot", "teleop"):
            raise ValueError("Device kind must be 'robot' or 'teleop'.")
        if not isinstance(self.parameters, dict) or any(
            not isinstance(key, str) for key in self.parameters
        ):
            raise ValueError("Device parameters must be a dictionary with string keys.")
        if "type" in self.parameters:
            raise ValueError("Device type must be supplied separately from parameters.")
        if not isinstance(self.extra_sections, dict) or any(
            not isinstance(key, str) for key in self.extra_sections
        ):
            raise ValueError("Extra configuration sections must have string keys.")
        if _RESERVED_SECTIONS.intersection(self.extra_sections):
            raise ValueError("Extra sections cannot replace backend, robot, or teleop.")

        self.parameters = copy.deepcopy(self.parameters)
        self.extra_sections = copy.deepcopy(self.extra_sections)
        identifier = self.parameters.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in (".", "..")
            or any(character in identifier for character in "/\\\0")
        ):
            raise ValueError("Device id must be a nonempty name without path separators.")
        directory = self.parameters.get("calibration_dir")
        if not isinstance(directory, (str, Path)) or not str(directory).strip() or "\0" in str(directory):
            raise ValueError("Device calibration_dir must be a nonempty directory path.")
        self.parameters["calibration_dir"] = Path(directory).expanduser()

    @property
    def calibration_file(self) -> Path:
        """Calibration destination used by the selected device backend."""
        return Path(self.parameters["calibration_dir"]) / f"{self.parameters['id']}.json"

    @property
    def device_key(self) -> str:
        return self.kind

    def to_dict(self) -> dict[str, Any]:
        """Return a separate, YAML-compatible copy of this configuration."""
        return _yaml_value(
            {
                self.device_key: {"type": self.device_type, **self.parameters},
                **self.extra_sections,
            }
        )

    def save(self, path: str | Path, *, force: bool = False) -> Path:
        """Atomically write YAML, requiring ``force`` to replace an existing file.

        Serialize the complete document before creating files. Relative paths
        in parameters are preserved; callers creating a new configuration can
        choose an absolute calibration directory to make its meaning explicit.
        """
        yaml = _yaml_module()
        payload = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        destination = Path(path).expanduser().absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if force:
                os.replace(temporary, destination)
            else:
                # Creating a hard link fails atomically if the destination
                # already exists, including a dangling symbolic link.
                os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination


def load_setup_config(path: str | Path) -> HardwareSetupConfig:
    """Read and validate a setup document without importing its hardware backend."""
    yaml = _yaml_module()
    source = Path(path).expanduser().absolute()
    with source.open(encoding="utf-8") as stream:
        try:
            document = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML configuration {source}: {exc}") from exc
    if not isinstance(document, dict) or any(not isinstance(key, str) for key in document):
        raise ValueError("Hardware setup configuration must be a mapping with string keys.")
    device_keys = [key for key in ("robot", "teleop") if key in document]
    if len(device_keys) != 1:
        raise ValueError("Hardware setup configuration must contain exactly one robot or teleop.")
    device_key = device_keys[0]
    module = None
    if "backend" in document:
        backend = document["backend"]
        if not isinstance(backend, dict) or set(backend) != {"module", "kind"}:
            raise ValueError("Legacy backend must contain exactly module and kind.")
        if backend["kind"] != device_key:
            raise ValueError("backend.kind must match the robot or teleop section.")
        module = backend["module"]
    device = document[device_key]
    if not isinstance(device, dict) or "type" not in device:
        raise ValueError("Device configuration must be a mapping containing type.")
    config = HardwareSetupConfig(
        module=module,
        kind=device_key,
        device_type=device["type"],
        parameters={key: value for key, value in device.items() if key != "type"},
        extra_sections={
            key: value for key, value in document.items() if key not in _RESERVED_SECTIONS
        },
    )
    directory = config.parameters["calibration_dir"]
    if not directory.is_absolute():
        config.parameters["calibration_dir"] = (source.parent / directory).resolve()
    return config
