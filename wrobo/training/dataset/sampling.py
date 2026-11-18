"""Configuration and random draws for per-dataset timestep sampling."""

import math
from numbers import Real
from typing import List, Literal, Union

import numpy as np


SamplingMode = Literal["uniform", "early", "late"]
SAMPLING_MODES = ("uniform", "early", "late")


def normalize_biased_samples(
    biased_sample: Union[SamplingMode, List[SamplingMode]], count: int
) -> List[SamplingMode]:
    """Expand a shared mode or validate modes in dataset_dir order."""
    if count < 1:
        raise ValueError("dataset_dir must contain at least one dataset")
    if isinstance(biased_sample, str):
        modes = [biased_sample]
    elif isinstance(biased_sample, list):
        modes = list(biased_sample)
    else:
        raise ValueError("biased_sample must be a string or a list of sampling modes")

    if len(modes) not in (1, count):
        raise ValueError(
            f"biased_sample must contain one shared mode or one mode per dataset_dir "
            f"({count} entries), got {len(modes)}"
        )
    for index, mode in enumerate(modes):
        if mode not in SAMPLING_MODES:
            raise ValueError(
                f"Invalid biased_sample value at index {index}: {mode!r}. "
                f"Must be one of: {SAMPLING_MODES}"
            )

    return modes * count if len(modes) == 1 else modes


def _validate_uniform_sample_prob(probability):
    if (
        isinstance(probability, bool)
        or not isinstance(probability, Real)
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError("uniform_sample_prob must be a finite number in [0, 1]")
    return float(probability)


def normalize_uniform_sample_probs(
    uniform_sample_prob: Union[float, List[float]], count: int
) -> List[float]:
    """Expand a shared mixture probability or validate one per dataset."""
    if count < 1:
        raise ValueError("dataset_dir must contain at least one dataset")
    if isinstance(uniform_sample_prob, list):
        probabilities = uniform_sample_prob
    else:
        probabilities = [uniform_sample_prob]
    if len(probabilities) not in (1, count):
        raise ValueError(
            "uniform_sample_prob must contain one shared probability or one "
            f"probability per dataset_dir ({count} entries), got {len(probabilities)}"
        )
    probabilities = [_validate_uniform_sample_prob(value) for value in probabilities]
    return probabilities * count if len(probabilities) == 1 else probabilities


def sample_timestep(
    episode_len: int,
    sampling_mode: SamplingMode,
    uniform_sample_prob: float = 0.0,
) -> int:
    """Choose uniform or Beta sampling afresh for this timestep draw.

    For early/late modes, use uniform sampling with uniform_sample_prob and
    the configured Beta distribution otherwise. Uniform mode always remains
    uniform. A zero probability preserves the legacy random-number sequence.
    """
    if episode_len < 1:
        raise ValueError("episode_len must be at least 1")
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"Unknown sampling mode: {sampling_mode!r}")
    probability = _validate_uniform_sample_prob(uniform_sample_prob)
    if sampling_mode == "uniform" or probability == 1.0:
        return int(np.random.randint(episode_len))
    if probability > 0.0 and np.random.random() < probability:
        return int(np.random.randint(episode_len))

    a, b = (1.0, 2.0) if sampling_mode == "early" else (2.0, 1.0) #early: a=1, b=2; late: a=2, b=1
    u = np.random.beta(a=a, b=b)
    # Guard the u == 1.0 endpoint when mapping to discrete timestep indices.
    return min(int(u * episode_len), episode_len - 1)
