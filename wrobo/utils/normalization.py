"""Affine normalization parameters derived from real training-set statistics."""

from collections.abc import Mapping

import numpy as np


MIN_SCALE = 1e-8


def normalization_parameters(entry, mode="standard"):
    """Return float64 ``(center, scale)`` vectors without clipping samples.

    ``standard`` uses actual mean/std. ``quantile`` maps q01/q99 to -1/+1
    whenever their half-width exceeds MIN_SCALE; narrower dimensions use the
    same center and a scale floor. A constant dimension therefore maps to zero.
    Legacy quantiles_affine entries with pseudo mean/std are intentionally rejected.
    """
    if mode not in ("standard", "quantile"):
        raise ValueError("normalization mode must be 'standard' or 'quantile'")
    if not isinstance(entry, Mapping):
        raise ValueError("normalization statistics must be a mapping")
    if entry.get("normalization") == "quantiles_affine":
        raise ValueError("legacy quantiles_affine statistics are unsupported; recompute real mean/std/q01/q99")
    keys = ("mean", "std") if mode == "standard" else ("q01", "q99")
    missing = [key for key in keys if key not in entry]
    if missing:
        raise ValueError(f"normalization statistics are missing {missing} for mode {mode!r}")
    try:
        arrays = [np.asarray(entry[key]) for key in keys]
        if any(array.dtype.kind not in "fiu" for array in arrays):
            raise ValueError("statistics must contain real numbers, not booleans or strings")
        first, second = (array.astype(np.float64, copy=False) for array in arrays)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"normalization statistics {keys} must be numeric vectors") from error
    if first.ndim != 1 or first.size == 0 or second.shape != first.shape:
        raise ValueError(f"normalization statistics {keys} must be nonempty matching 1-D vectors")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError(f"normalization statistics {keys} must be finite")
    if mode == "standard":
        if not (second > 0).all():
            raise ValueError("normalization std must be positive")
        return first, second
    if (second < first).any():
        raise ValueError("normalization q99 must be greater than or equal to q01")
    # Half each bound first to avoid overflow for finite, large endpoints.
    center = first / 2 + second / 2
    scale = np.maximum(second / 2 - first / 2, MIN_SCALE)
    return center, scale
