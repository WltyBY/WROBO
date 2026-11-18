"""Strict, dependency-free local and remote compute-device selection."""

from dataclasses import dataclass
import ipaddress
import re


@dataclass(frozen=True)
class TargetDevice:
    compute_device: str
    address: str | None = None

    @property
    def is_remote(self):
        return self.address is not None


def canonical_compute_device(value):
    """Canonicalize an explicit device without inspecting this machine's hardware."""
    value = str(value).strip().lower()
    if value in ("cpu", "mps"):
        return value
    if value == "cuda":
        return "cuda:0"
    if re.fullmatch(r"cuda:[0-9]+", value):
        return f"cuda:{int(value.split(':')[1])}"
    raise ValueError(f"Invalid compute device {value!r}; use cpu, mps, cuda or cuda:N")


def parse_device(value):
    """Accept cpu/cuda[:N]/mps/auto or HOST:PORT:cpu/cuda[:N]/mps.

    Bracket IPv6 addresses, e.g. [::1]:8080:cuda:0. Remote auto is
    deliberately unsupported: the handshake verifies the requested device.
    """
    if not isinstance(value, str):
        raise ValueError("Device must be a command-line string")
    value = value.strip()

    if value.lower() == "auto":
        return TargetDevice(compute_device="auto")

    try:
        return TargetDevice(canonical_compute_device(value))
    except ValueError:
        pass

    match = re.fullmatch(r"(\[[^\]]+\]|[A-Za-z0-9_.-]+):([0-9]+):(.+)", value)
    if match is None:
        raise ValueError(
            "Use a local device or HOST:PORT:DEVICE (e.g. 192.168.1.10:8080:cuda)"
        )

    host, port, compute = match.groups()
    if host.startswith("["):
        try:
            ipaddress.IPv6Address(host[1:-1])
        except ValueError as exc:
            raise ValueError("Invalid bracketed IPv6 address") from exc

    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError("Remote port must be between 1 and 65535")

    return TargetDevice(
        compute_device=canonical_compute_device(compute), address=f"{host}:{port}"
    )


def resolve_local_device(value):
    """Resolve auto and reject unavailable explicitly selected accelerators."""
    import torch

    target = parse_device(str(value))
    if target.is_remote:
        raise ValueError("A local policy runner cannot use a remote device address")
    
    name = target.compute_device
    mps = getattr(torch.backends, "mps", None)
    if name == "auto":
        name = (
            "cuda:0"
            if torch.cuda.is_available()
            else ("mps" if mps is not None and mps.is_available() else "cpu")
        )

    device = torch.device(name)
    if device.type == "cuda" and (
        not torch.cuda.is_available() or device.index >= torch.cuda.device_count()
    ):
        raise ValueError(f"CUDA device {name} requested but unavailable")
    if device.type == "mps" and (mps is None or not mps.is_available()):
        raise ValueError("MPS requested but unavailable")
    
    return device
