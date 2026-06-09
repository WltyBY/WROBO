import os
import h5py
import numpy as np

from typing import Dict, List

from wrobo.utils.image_codec import get_encoding, decode_frame, RAW


def get_norm_stats(
    dataset_dir, case_lst=None, num_sample=1000, img_num_sample=200, seed=319
):
    """
    Compute normalization stats across the given episodes, used to normalize model
    inputs/outputs.

    What is collected:
        - every dataset under "observations" whose name contains "proprio"
          (e.g. proprio_state, proprio_vel)  -> per-dimension mean/std
        - every top-level dataset whose name contains "action"
          (e.g. action_abs, action_rel)      -> per-dimension mean/std
        - every dataset under "observations" whose name contains "image"
          (e.g. image_top, image_angle)      -> per-CHANNEL mean/std, computed on
          pixel values divided by 255.0 (to mirror torchvision's ToTensor + Normalize,
          which also operates on [0, 1] per-channel values)

    Time-step sampling: for each episode we draw up to `num_sample` time indices from
    that episode's time axis T and reuse the SAME indices for all collected keys, so
    different signals stay time-aligned. For images we use the first `img_num_sample`
    of those same indices (images are far larger, so a smaller per-episode cap keeps
    memory and IO bounded). proprio / action / image are collected independently: a
    file missing one still contributes the others.

    Image stats are accumulated incrementally (sum / sum-of-squares / count) with
    chunked reads, so peak memory stays bounded regardless of frame size or count.

    Returns: {key: {"mean": np.ndarray, "std": np.ndarray}}
        - proprio/action: arrays of shape (dim,)
        - image:          arrays of shape (C,)  (per channel)
    """
    if case_lst is None:
        file_paths = [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith(".hdf5")
        ]
    else:
        file_paths = [
            os.path.join(dataset_dir, f if f.endswith(".hdf5") else f + ".hdf5")
            for f in case_lst
        ]

    if not file_paths:
        raise ValueError(f"No episode files found in {dataset_dir}")

    rng = np.random.default_rng(seed)

    # per-dimension buffers for proprio/action (small, buffered directly)
    data_buffers: Dict[str, List[np.ndarray]] = {}
    # per-channel running accumulators for images (large, accumulated incrementally)
    img_accum: Dict[str, Dict[str, np.ndarray]] = {}

    for file_path in file_paths:
        with h5py.File(file_path, "r") as root:
            # gather the keys: proprio + image under observations, actions at root
            vec_keys: Dict[str, h5py.Dataset] = {}  # proprio/action -> per-dim stats
            img_keys: Dict[str, h5py.Dataset] = {}  # image -> per-channel stats

            obs_group = root.get("observations")
            if obs_group is not None:
                for key in obs_group.keys():
                    if "image" in key.lower():
                        img_keys[key] = obs_group[key]
                    elif "proprio" in key.lower():
                        vec_keys[key] = obs_group[key]

            for key in root.keys():
                if key == "observations":
                    continue
                if "action" in key.lower() and isinstance(root[key], h5py.Dataset):
                    vec_keys[key] = root[key]

            if not vec_keys and not img_keys:
                continue

            # establish the episode time length T (prefer the recorded seq_len, fall
            # back to the shortest collected sequence) and sample indices once per file.
            seq_len_attr = dict(root.attrs).get("seq_len")
            lengths = [
                int(ds.shape[0]) for ds in (*vec_keys.values(), *img_keys.values())
            ]
            T = int(seq_len_attr) if seq_len_attr is not None else min(lengths)
            # never index past the shortest available sequence
            T = min([T] + lengths)
            if T <= 0:
                continue

            if T > num_sample:
                sample_indices = rng.choice(T, size=num_sample, replace=False)
                sample_indices.sort()
            else:
                sample_indices = np.arange(T)

            # proprio / action: buffer sampled rows directly (small per-dim vectors)
            for key, ds in vec_keys.items():
                sampled = ds[sample_indices]  # (len(sample_indices), dim)
                data_buffers.setdefault(key, []).append(sampled)

            # images: reuse the first img_num_sample of the SAME time indices, and
            # accumulate per-channel sum / sum-of-squares / count over [0, 1] pixels.
            # NOTE: for "raw" datasets, frames are gzip-compressed with one frame per
            # chunk, so h5py fancy indexing (ds[list]) is pathologically slow (per-point
            # selection); reading one frame at a time (ds[i]) is ~80x faster. Encoded
            # (png/jpg) datasets are 1-D byte strings, decoded per frame to (C, H, W).
            img_indices = sample_indices[:img_num_sample]
            for key, ds in img_keys.items():
                encoding = get_encoding(ds)
                for i in img_indices:
                    raw = ds[int(i)]
                    frame = (
                        raw if encoding == RAW else decode_frame(raw)
                    )  # (C, H, W) uint8
                    frame = np.asarray(frame, dtype=np.float64) / 255.0
                    C = frame.shape[0]
                    acc = img_accum.setdefault(
                        key,
                        {
                            "sum": np.zeros(C, dtype=np.float64),
                            "sumsq": np.zeros(C, dtype=np.float64),
                            "count": 0,  # number of pixels per channel
                        },
                    )
                    acc["sum"] += frame.sum(axis=(1, 2))
                    acc["sumsq"] += (frame**2).sum(axis=(1, 2))
                    acc["count"] += frame.shape[1] * frame.shape[2]

    if not data_buffers and not img_accum:
        raise ValueError(
            f"No proprio/action/image data found in any episode under {dataset_dir}. "
            f"Cannot compute normalization stats."
        )

    norm_stats = {}
    # per-dimension stats for proprio/action
    for key, samples in data_buffers.items():
        if not samples:
            continue
        stacked = np.concatenate(samples, axis=0)  # (sum_of_samples, dim)
        mean = stacked.mean(axis=0)  # (dim,)
        std = stacked.std(axis=0).clip(min=1e-8)  # floor std to avoid div-by-zero
        norm_stats[key] = {"mean": mean, "std": std}

    # per-channel stats for images (from the running accumulators)
    for key, acc in img_accum.items():
        count = acc["count"]
        if count <= 0:
            continue
        mean = acc["sum"] / count  # (C,)
        var = acc["sumsq"] / count - mean**2
        std = np.sqrt(np.clip(var, 0.0, None)).clip(min=1e-8)  # (C,)
        norm_stats[key] = {"mean": mean, "std": std}

    return norm_stats
