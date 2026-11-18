"""Collect numerical and image statistics from explicitly selected episodes."""

import os
from collections import Counter
from multiprocessing import cpu_count, get_context
from numbers import Integral
from time import perf_counter

import h5py
import numpy as np
from tqdm import tqdm

from wrobo.utils.image_codec import decode_frame, get_encoding, is_encoded
from wrobo.utils.normalization import MIN_SCALE


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _process_single_file(arguments):
    """Open one episode once; return sampled vectors and image accumulators."""
    file_path, num_sample, img_num_sample, seed = arguments
    rng = np.random.default_rng(seed)
    vectors, images = {}, {}
    with h5py.File(file_path, "r") as root:
        vector_datasets, image_datasets = {}, {}
        observations = root.get("observations")
        if observations is not None:
            if not isinstance(observations, h5py.Group):
                raise ValueError(f"{file_path}: observations must be an HDF5 group")
            for key, dataset in observations.items():
                if key.startswith("proprio_"):
                    vector_datasets[key] = dataset
                elif key.startswith("image_"):
                    image_datasets[key] = dataset
        for key, dataset in root.items():
            if key.startswith("action_"):
                vector_datasets[key] = dataset
        datasets = {**vector_datasets, **image_datasets}
        if not datasets:
            raise ValueError(f"{file_path}: no proprio_*, action_* or image_* datasets")
        for key, dataset in datasets.items():
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim < 1:
                raise ValueError(f"{file_path}: {key} must be a time-indexed HDF5 dataset")
        length = root.attrs.get("seq_len", next(iter(datasets.values())).shape[0])
        length = _positive_integer(length, f"{file_path}: seq_len")
        for key, dataset in datasets.items():
            if dataset.shape[0] != length:
                raise ValueError(
                    f"{file_path}: {key} length {dataset.shape[0]} does not match seq_len {length}"
                )
        if length > num_sample:
            indices = np.sort(rng.choice(length, size=num_sample, replace=False))
        else:
            indices = np.arange(length)

        for key, dataset in vector_datasets.items():
            if dataset.ndim != 2 or dataset.shape[1] < 1 or dataset.dtype.kind not in "fiu":
                raise ValueError(f"{file_path}: {key} must be a real numeric [T,D] array")
            # Low-dimensional vectors are cheap to validate in full. This catches
            # invalid rows even when they are not selected for statistics sampling.
            values = np.asarray(dataset[:], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{file_path}: {key} contains non-finite values")
            vectors[key] = values[indices]

        # Use a uniform subset of the same sampled time indices, rather than the
        # earliest sorted indices, which would bias image statistics toward episode starts.
        image_indices = indices
        if len(indices) > img_num_sample:
            image_indices = np.sort(rng.choice(indices, size=img_num_sample, replace=False))
        for key, dataset in image_datasets.items():
            encoding = get_encoding(dataset)
            if encoding not in ("raw", "gzip", "png", "jpeg"):
                raise ValueError(f"{file_path}: {key} has unsupported image encoding {encoding!r}")
            encoded = is_encoded(encoding)
            if dataset.ndim != (1 if encoded else 4):
                raise ValueError(f"{file_path}: {key} has invalid {encoding} image layout")
            accumulator = None
            for index in image_indices:
                try:
                    frame = decode_frame(dataset[int(index)]) if encoded else dataset[int(index)]
                except Exception as error:
                    raise ValueError(f"{file_path}: cannot decode {key} frame {index}") from error
                if (frame.ndim != 3 or frame.shape[0] not in (1, 3, 4)
                        or min(frame.shape) < 1 or frame.dtype != np.uint8):
                    raise ValueError(f"{file_path}: {key} frame {index} must be uint8 CHW")
                pixels = np.asarray(frame, dtype=np.float64) / 255.0
                channels = pixels.shape[0]
                if accumulator is None:
                    accumulator = {"sum": np.zeros(channels), "sumsq": np.zeros(channels), "count": 0}
                elif accumulator["sum"].shape != (channels,):
                    raise ValueError(f"{file_path}: {key} changes channel count between frames")
                accumulator["sum"] += pixels.sum(axis=(1, 2))
                accumulator["sumsq"] += np.square(pixels).sum(axis=(1, 2))
                accumulator["count"] += pixels.shape[1] * pixels.shape[2]
            images[key] = accumulator
    return vectors, images


def get_norm_stats(
    episode_paths, num_sample=1000, img_num_sample=200, seed=319, num_workers=None,
    *, show_progress=True, log_fn=None,
):
    """Compute one set of statistics over the supplied HDF5 episode paths.

    No directory discovery, split selection or task filtering happens here. Callers
    supply exactly the train episodes to include, in a stable order. Combining
    datasets means supplying their combined paths, not averaging precomputed quantiles.

    Direct observations/proprio_* and root action_* fields return true per-dimension
    mean/std/q01/q99 arrays. Numerical aggregation uses float64, with population
    standard deviation floored at MIN_SCALE. All vector rows are checked for
    non-finite values, while up to num_sample shared time indices per file contribute
    to the statistics.

    Direct observations/image_* fields return per-channel mean/std of RGB pixels
    divided by 255. Up to img_num_sample uniformly selected frames from those same
    indices are decoded. PNG/JPEG and dense raw/gzip layouts are supported;
    unselected image frames are not decoded. Images are accumulated by pixel count.

    Errors propagate instead of silently omitting files. num_workers 0 or 1 runs
    serially; larger values use worker processes. Results and sampling are identical
    for serial and parallel execution with the same path order and seed.

    The calling process logs collection/computation stages and updates one episode
    progress bar. log_fn can persist stage messages with the Trainer's logger;
    show_progress only controls the bar. Workers never receive either callback.
    """
    if isinstance(episode_paths, (str, bytes, os.PathLike)):
        raise ValueError("episode_paths must be an explicit list of HDF5 file paths")
    try:
        paths = [os.fspath(path) for path in episode_paths]
    except (TypeError, ValueError) as error:
        raise ValueError("episode_paths must be an explicit list of HDF5 file paths") from error
    if not paths:
        raise ValueError("episode_paths must contain at least one HDF5 file")
    canonical_paths = [os.path.realpath(path) for path in paths]
    if len(set(canonical_paths)) != len(paths):
        raise ValueError("episode_paths must not contain duplicate files")
    num_sample = _positive_integer(num_sample, "num_sample")
    img_num_sample = _positive_integer(img_num_sample, "img_num_sample")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    elif isinstance(num_workers, bool) or not isinstance(num_workers, Integral) or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer or None")
    workers = min(max(1, int(num_workers)), len(paths))
    if log_fn is not None and not callable(log_fn):
        raise ValueError("log_fn must be callable or None")
    log = log_fn if log_fn is not None else lambda message: print(message, flush=True)
    started = perf_counter()
    sources = []
    for path in paths:
        parent = os.path.dirname(os.path.abspath(os.fsdecode(path)))
        sources.append(os.path.dirname(parent) if os.path.basename(parent) == "episodes" else parent)
    totals = Counter(sources)
    completed = Counter()
    log(
        f"Normalization statistics: collecting samples from {len(paths)} training episodes "
        f"across {len(totals)} datasets (workers={workers})"
    )
    for source, count in totals.items():
        log(f"Statistics source {source}: {count} training episodes")
    arguments = [(path, num_sample, img_num_sample, int(seed) + index) for index, path in enumerate(paths)]
    with tqdm(total=len(paths), desc="Collecting normalization samples", unit="episode",
              dynamic_ncols=True, disable=not show_progress) as progress:
        def collect(iterator):
            results = []
            iterator = iter(iterator)
            previous_source = None
            for source in sources:
                if not completed[source]:
                    with tqdm.external_write_mode():
                        log(f"Collecting statistics samples: dataset={source} (0/{totals[source]} episodes)")
                if source != previous_source:
                    progress.set_postfix(dataset=os.path.basename(source) or source,
                                         completed=f"{completed[source]}/{totals[source]}", refresh=True)
                    previous_source = source
                # Consume in path order to preserve the original aggregation and
                # sampling exactly; only completed episodes advance the bar.
                results.append(next(iterator))
                completed[source] += 1
                progress.set_postfix(dataset=os.path.basename(source) or source,
                                     completed=f"{completed[source]}/{totals[source]}", refresh=False)
                progress.update(1)
                if completed[source] == totals[source]:
                    with tqdm.external_write_mode():
                        log(f"Collected statistics samples: dataset={source} "
                            f"({completed[source]}/{totals[source]} episodes)")
            return results

        if workers > 1:
            with get_context("spawn").Pool(workers) as pool:
                results = collect(pool.imap(_process_single_file, arguments, chunksize=1))
        else:
            results = collect(map(_process_single_file, arguments))

    vector_buffers, image_accumulators = {}, {}
    log("Aggregating collected numerical samples and image moments")
    for vectors, images in results:
        for key, values in vectors.items():
            buffers = vector_buffers.setdefault(key, [])
            if buffers and values.shape[1:] != buffers[0].shape[1:]:
                raise ValueError(f"{key}: vector dimensions differ between selected episodes")
            buffers.append(values)
        for key, values in images.items():
            if key not in image_accumulators:
                image_accumulators[key] = values
                continue
            total = image_accumulators[key]
            if total["sum"].shape != values["sum"].shape:
                raise ValueError(f"{key}: image channel counts differ between selected episodes")
            total["sum"] += values["sum"]
            total["sumsq"] += values["sumsq"]
            total["count"] += values["count"]

    result = {}
    for key, buffers in vector_buffers.items():
        values = np.concatenate(buffers, axis=0)
        log(f"Computing numerical mean/std/q01/q99: key={key}, "
            f"samples={values.shape[0]}, dimensions={values.shape[1]}")
        with np.errstate(over="raise", invalid="raise"):
            try:
                mean, std = values.mean(axis=0), values.std(axis=0)
                q01, q99 = np.quantile(values, [0.01, 0.99], axis=0)
            except FloatingPointError as error:
                raise ValueError(f"{key}: numerical overflow while computing statistics") from error
        result[key] = {"mean": mean, "std": std.clip(min=MIN_SCALE), "q01": q01, "q99": q99}
    for key, values in image_accumulators.items():
        log(f"Computing image mean/std: key={key}, pixels_per_channel={values['count']}")
        mean = values["sum"] / values["count"]
        variance = values["sumsq"] / values["count"] - np.square(mean)
        result[key] = {"mean": mean, "std": np.sqrt(variance.clip(min=0.0)).clip(min=MIN_SCALE)}
    log(f"Normalization statistics complete: {len(paths)} episodes, {len(result)} fields, "
        f"elapsed={perf_counter() - started:.1f}s")
    return result
