"""Local utilities used by the standalone SmolVLA port."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


_transformers_available = importlib.util.find_spec("transformers") is not None


def require_package(pkg_name: str, extra: str | None = None, import_name: str | None = None):
    """Raise a useful error when an optional runtime dependency is unavailable."""
    check_name = import_name or pkg_name
    if importlib.util.find_spec(check_name) is None:
        suffix = f" (the {extra!r} extra)" if extra else ""
        raise ImportError(f"Package {pkg_name!r}{suffix} is required by SmolVLA")


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Pad the final dimension with zeros, following the official implementation."""
    if vector.shape[-1] > new_dim:
        raise ValueError(
            f"Input dimension {vector.shape[-1]} exceeds configured maximum {new_dim}. "
            "Increase max_state_dim/max_action_dim instead of silently truncating data."
        )
    if vector.shape[-1] == new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad(img: Tensor, height: int, width: int, *, pad_value: float) -> Tensor:
    """Aspect-preserving resize with SmolVLA's left/top padding convention."""
    squeeze_batch = img.ndim == 3
    if squeeze_batch:
        img = img.unsqueeze(0)
    elif img.ndim != 4:
        raise ValueError(f"Expected image tensor (C,H,W) or (B,C,H,W), got {tuple(img.shape)}")
    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img.squeeze(0) if squeeze_batch else img

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )
    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    padded = F.pad(resized, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded.squeeze(0) if squeeze_batch else padded


def create_sinusoidal_pos_embedding(
    time: Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> Tensor:
    """Create the openpi/SmolVLA sine-cosine timestep embedding."""
    if dimension % 2:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim not in (1, 2):
        raise ValueError("time must have shape (batch,) or (batch, action_horizon)")

    dtype = torch.float32 if device.type in {"mps", "xpu"} else torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    sin_input = time[..., None] * (2 * math.pi / period)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Build block-causal attention masks from token validity and block markers."""
    if pad_masks.ndim != 2 or att_masks.ndim != 2:
        raise ValueError(
            f"pad_masks and att_masks must be rank 2, got {pad_masks.ndim} and {att_masks.ndim}"
        )
    cumsum = torch.cumsum(att_masks, dim=1)
    attention = cumsum[:, None, :] <= cumsum[:, :, None]
    validity = pad_masks[:, None, :] & pad_masks[:, :, None]
    return attention & validity


def sample_noise(shape, device) -> Tensor:
    return torch.normal(0.0, 1.0, size=shape, dtype=torch.float32, device=device)


def sample_time_beta(
    bsize: int,
    device,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    scale: float = 0.999,
    offset: float = 0.001,
) -> Tensor:
    # Beta/Dirichlet sampling is not implemented on MPS, so sample on CPU.
    distribution = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32), torch.tensor(beta, dtype=torch.float32)
    )
    return (distribution.sample((bsize,)) * scale + offset).to(device=device, dtype=torch.float32)


def euler_integrate(denoise_fn: Callable[[Tensor, Tensor], Tensor], noise: Tensor, num_steps: int) -> Tensor:
    """Integrate the learned velocity from noise at t=1 to actions at t=0."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    batch_size = noise.shape[0]
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        timestep = torch.full(
            (batch_size,), 1.0 + step * dt, dtype=torch.float32, device=noise.device
        )
        x_t = x_t + dt * denoise_fn(x_t, timestep)
    return x_t


def pad_sequence(tensor: Tensor, max_len: int, pad_value: int | float = 0) -> Tensor:
    """Pad sequence dimension 1 to ``max_len`` without changing other dimensions."""
    if tensor.shape[1] >= max_len:
        return tensor
    result = torch.full(
        (tensor.shape[0], max_len, *tensor.shape[2:]),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    result[:, : tensor.shape[1]] = tensor
    return result
