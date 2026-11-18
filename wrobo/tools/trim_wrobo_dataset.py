#!/usr/bin/env python
"""Trim static episode boundaries in an existing wrobo HDF5 dataset.

Writes a NEW dataset directory; never edits source episodes. PNG/JPEG byte
streams are sliced directly, without decoding or lossy recompression.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wrobo.training.dataset.instruction import parse_instruction_segments


def _compute_action_velocity(action, window_size):
    """Compute the velocity signal used for both estimation and trimming."""
    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}")

    action_velocity = np.linalg.norm(np.diff(action, axis=0), axis=1)
    if window_size > 1 and len(action_velocity) > 0:
        effective_window = min(window_size, len(action_velocity))
        kernel = np.ones(effective_window) / effective_window
        action_velocity = np.convolve(action_velocity, kernel, mode='same')
    return action_velocity


def trim_static_frames(
    action,
    threshold=0.01,
    min_frames=5,
    window_size=3,
):
    """
    Detect and trim static frames at the beginning and end of an episode.

    The function computes action velocity (L2 norm of action differences) and identifies
    the first and last regions with significant movement. Frames before the first movement
    and after the last movement are trimmed.

    Args:
        action: Action array of shape (T, action_dim)
        threshold: Threshold for action change magnitude (L2 norm per frame).
                   Typical values: 0.001-0.01 for normalized actions, 0.01-0.1 for joint angles
        min_frames: Minimum number of frames to keep even if all are static
        window_size: Size of sliding window for smoothing velocity signal.
                     Helps avoid trimming due to noise or small jitters.

    Returns:
        start_idx: Index of first non-static frame
        end_idx: Index after last non-static frame (python slice convention)
        trimmed: Whether any frames were trimmed
        stats: Dict with statistics about the trimming:
               - original_length: Original episode length
               - trimmed_length: Length after trimming
               - trimmed_start: Number of frames trimmed from start
               - trimmed_end: Number of frames trimmed from end
               - max_velocity: Maximum velocity in the episode
               - mean_velocity: Mean velocity in the episode
    """
    original_length = len(action)

    if original_length <= min_frames:
        stats = {
            'original_length': original_length,
            'trimmed_length': original_length,
            'trimmed_start': 0,
            'trimmed_end': 0,
            'max_velocity': 0.0,
            'mean_velocity': 0.0,
        }
        return 0, original_length, False, stats

    # Calculate and smooth action change magnitude (velocity).
    action_velocity = _compute_action_velocity(action, window_size)

    # Statistics
    max_velocity = float(np.max(action_velocity)) if len(action_velocity) > 0 else 0.0
    mean_velocity = float(np.mean(action_velocity)) if len(action_velocity) > 0 else 0.0

    # Find frames with significant movement
    moving_frames = action_velocity > threshold

    if not np.any(moving_frames):
        # All frames are static, keep original
        stats = {
            'original_length': original_length,
            'trimmed_length': original_length,
            'trimmed_start': 0,
            'trimmed_end': 0,
            'max_velocity': max_velocity,
            'mean_velocity': mean_velocity,
        }
        return 0, original_length, False, stats

    # Find first and last frames with movement
    moving_indices = np.where(moving_frames)[0]
    first_moving = moving_indices[0]
    last_moving = moving_indices[-1]

    # Include one frame before and after movement for context
    # Note: action_diff has length T-1, so index i corresponds to movement between frame i and i+1
    start_idx = max(0, first_moving)  # The frame where movement starts
    end_idx = min(original_length, last_moving + 2)  # +2: last_moving is in diff space, +1 to include the frame, +1 for slice convention

    # Ensure minimum length
    if end_idx - start_idx < min_frames:
        # Center around the movement
        center = (start_idx + end_idx) // 2
        start_idx = max(0, center - min_frames // 2)
        end_idx = min(original_length, start_idx + min_frames)
        start_idx = max(0, end_idx - min_frames)

    trimmed = start_idx > 0 or end_idx < original_length
    trimmed_length = end_idx - start_idx

    stats = {
        'original_length': original_length,
        'trimmed_length': trimmed_length,
        'trimmed_start': start_idx,
        'trimmed_end': original_length - end_idx,
        'max_velocity': max_velocity,
        'mean_velocity': mean_velocity,
    }

    return start_idx, end_idx, trimmed, stats


def trim_episode_data(
    action,
    proprio_state=None,
    video_frames=None,
    threshold=0.01,
    min_frames=5,
    window_size=3,
):
    """
    Trim static frames from all episode data (action, proprio, and video frames).

    This is a convenience function that applies trim_static_frames() to the action
    and then slices all other data accordingly.

    Args:
        action: Action array of shape (T, action_dim)
        proprio_state: Optional proprioceptive state array of shape (T, state_dim)
        video_frames: Optional dict of {camera_name: list/array of frames}
        threshold: Threshold for action change magnitude
        min_frames: Minimum number of frames to keep
        window_size: Size of sliding window for smoothing velocity signal

    Returns:
        trimmed_data: Dict containing:
            - action: Trimmed action array
            - proprio_state: Trimmed proprio state (if provided)
            - video_frames: Trimmed video frames (if provided)
            - start_idx: Start index of trim
            - end_idx: End index of trim
            - trimmed: Whether any frames were trimmed
            - stats: Trimming statistics
    """
    # Detect trim indices
    start_idx, end_idx, trimmed, stats = trim_static_frames(
        action,
        threshold=threshold,
        min_frames=min_frames,
        window_size=window_size,
    )

    # Trim all data
    trimmed_data = {
        'action': action[start_idx:end_idx],
        'start_idx': start_idx,
        'end_idx': end_idx,
        'trimmed': trimmed,
        'stats': stats,
    }

    if proprio_state is not None:
        trimmed_data['proprio_state'] = proprio_state[start_idx:end_idx]

    if video_frames is not None:
        trimmed_frames = {}
        for cam_name, frames in video_frames.items():
            if isinstance(frames, list):
                trimmed_frames[cam_name] = frames[start_idx:end_idx]
            else:
                trimmed_frames[cam_name] = frames[start_idx:end_idx]
        trimmed_data['video_frames'] = trimmed_frames

    return trimmed_data


def estimate_threshold_from_dataset(actions_list, percentile=5, window_size=3):
    """
    Estimate a good threshold value from a list of action arrays.

    This computes the velocity distribution across all episodes and returns
    a threshold at the specified percentile. The velocity calculation mirrors
    trim_static_frames(), including its optional moving-average smoothing.
    Zero velocities are excluded when estimating the percentile so that a
    dataset with many static frames does not produce a zero threshold.

    Args:
        actions_list: List of action arrays, each of shape (T_i, action_dim)
        percentile: Percentile of velocity distribution to use as threshold (default: 5)
        window_size: Moving-average window used by trim_static_frames (default: 3)

    Returns:
        threshold: Estimated threshold value
        stats: Dict with velocity statistics:
               - min: Minimum velocity
               - max: Maximum velocity
               - mean: Mean velocity
               - median: Median velocity
               - percentile_X: Value at requested percentile
    """
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}")

    all_velocities = []

    for action in actions_list:
        if len(action) > 1:
            action_velocity = _compute_action_velocity(action, window_size)
            action_velocity = action_velocity[np.isfinite(action_velocity)]
            # Static frames should not force auto to choose a zero threshold.
            action_velocity = action_velocity[action_velocity > 0]
            if len(action_velocity) > 0:
                all_velocities.append(action_velocity)

    if not all_velocities:
        return 0.0, {}

    all_velocities = np.concatenate(all_velocities)

    percentile_value = float(np.percentile(all_velocities, percentile))
    # trim_static_frames uses a strict > comparison. Moving one float below the
    # percentile keeps quantized velocities equal to the cutoff classified as movement.
    threshold = float(np.nextafter(percentile_value, -np.inf))

    stats = {
        'min': float(np.min(all_velocities)),
        'max': float(np.max(all_velocities)),
        'mean': float(np.mean(all_velocities)),
        'median': float(np.median(all_velocities)),
        f'percentile_{percentile}': percentile_value,
    }

    return threshold, stats


def _read_action(path):
    with h5py.File(path, "r") as root:
        action = np.asarray(root["action_abs"][:], dtype=np.float64)
        length = root.attrs.get("seq_len")
        if (not isinstance(length, (int, np.integer)) or isinstance(length, (bool, np.bool_))
                or length < 1 or action.ndim != 2 or action.shape[1] < 1
                or len(action) != length or not np.isfinite(action).all()):
            raise ValueError(f"{path}: expected finite action_abs[T,D] matching positive integer seq_len")
        return action


def _temporal(dataset, length, static_keys):
    name = dataset.name.lstrip("/")
    if any(name == key or name.startswith(key + "/") for key in static_keys):
        return False
    if dataset.shape is None or dataset.ndim == 0:
        return False
    required = (name.startswith(("observations/", "metrics/")) or
                name in ("action_abs", "action_delta", "action_rel", "prediction_abs", "target_abs",
                         "reward", "rewards", "done", "timestamp"))
    if required and dataset.shape[0] != length:
        raise ValueError(f"{dataset.name}: first dimension {dataset.shape[0]} != seq_len {length}; "
                         "use --static-keys only if this is genuinely non-temporal metadata")
    return dataset.shape[0] == length


def _walk(group, seen=None):
    """Reject aliases/external data instead of silently changing link semantics."""
    seen = set() if seen is None else seen
    address = h5py.h5o.get_info(group.id).addr
    if address in seen:
        raise ValueError(f"HDF5 hard-link aliases/cycles are not supported: {group.name}")
    seen.add(address)
    for name in group:
        if not isinstance(group.get(name, getlink=True), h5py.HardLink):
            raise ValueError(f"External/soft links are not supported: {group.name}/{name}")
        obj = group[name]
        yield obj
        if isinstance(obj, h5py.Group):
            yield from _walk(obj, seen)
        else:
            address = h5py.h5o.get_info(obj.id).addr
            if address in seen:
                raise ValueError(f"HDF5 hard-link aliases are not supported: {obj.name}")
            seen.add(address)


def trim_instruction(value, start, end, original_length):
    """Preserve literal/episode text; intersect inclusive frame annotations."""
    start, end = int(start), int(end)
    if start == 0 and end == original_length:
        return value
    text = value.decode("utf-8") if isinstance(value, bytes) else value
    if not isinstance(text, str):
        return value
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError:
        return value
    if not isinstance(metadata, dict) or "frames" not in metadata:
        return value
    parse_instruction_segments(text)
    frames = []
    for frame in metadata["frames"]:
        lo, hi = max(frame["start"], start), min(frame["end"], end - 1)
        if lo <= hi:
            frames.append({**frame, "start": lo - start, "end": hi - start})
    metadata["frames"] = frames
    if not frames and metadata.get("episode") is None:
        raise ValueError("Trimming removes all frame-only instruction annotations; "
                         "adjust threshold or annotations before processing")
    result = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    parse_instruction_segments(result)
    return result


def _plan(path, threshold, min_frames, window_size, static_keys):
    action = _read_action(path)
    start, end, trimmed, stats = trim_static_frames(
        action, threshold=threshold, min_frames=min_frames, window_size=window_size)
    with h5py.File(path, "r") as root:
        for obj in _walk(root):
            if isinstance(obj, h5py.Dataset):
                if obj.is_virtual or obj.external or h5py.check_dtype(ref=obj.dtype) is not None:
                    raise ValueError(f"Unsupported virtual/external/reference dataset: {obj.name}")
                _temporal(obj, len(action), static_keys)
        if "instruction" in root.attrs:
            trim_instruction(root.attrs["instruction"], start, end, len(action))
    return {"episode": path.name, "start": int(start), "end": int(end),
            "trimmed": bool(trimmed),
            **{key: value.item() if isinstance(value, np.generic) else value
               for key, value in stats.items()}}


def _copy_dataset(source, parent, name, start, end, temporal, block_frames):
    if not temporal:
        source.file.copy(source, parent, name=name)
        return
    shape = (end - start, *source.shape[1:])
    options = {}
    if source.chunks is not None:
        options["chunks"] = tuple(max(1, min(c, n)) for c, n in zip(source.chunks, shape))
        options["maxshape"] = source.maxshape
    if source.compression is not None:
        options.update(compression=source.compression, compression_opts=source.compression_opts)
    if source.shuffle:
        options["shuffle"] = True
    if source.fletcher32:
        options["fletcher32"] = True
    if source.scaleoffset is not None:
        options["scaleoffset"] = source.scaleoffset
    target = parent.create_dataset(name, shape=shape, dtype=source.dtype, **options)
    target.attrs.update(dict(source.attrs))
    # Vlen encoded frames are copied one-by-one: avoids h5py broadcasting a
    # same-length object array as a dense image matrix. Raw images are chunked.
    step = 1 if h5py.check_dtype(vlen=source.dtype) is not None else block_frames
    for offset in range(0, end - start, step):
        if step == 1:
            target[offset] = source[start + offset]
        else:
            count = min(step, end - start - offset)
            target[offset:offset + count] = source[start + offset:start + offset + count]


def write_episode(source_path, destination, plan, static_keys=(), block_frames=16):
    """Write an episode to a temporary file, then publish without overwriting."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    fd, temporary_name = tempfile.mkstemp(prefix=".trim_", suffix=".hdf5", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with h5py.File(source_path, "r") as source, h5py.File(temporary, "w") as target:
            length, start, end = plan["original_length"], plan["start"], plan["end"]
            if int(source.attrs["seq_len"]) != length:
                raise ValueError(f"Source changed since planning: {source_path}")
            target.attrs.update(dict(source.attrs))
            for obj in _walk(source):
                name = obj.name.lstrip("/")
                if isinstance(obj, h5py.Group):
                    target.require_group(name).attrs.update(dict(obj.attrs))
                else:
                    parent_name, _, base = name.rpartition("/")
                    parent = target.require_group(parent_name) if parent_name else target
                    _copy_dataset(obj, parent, base, start, end, _temporal(obj, length, static_keys), block_frames)
            target.attrs["seq_len"] = end - start
            if "instruction" in source.attrs:
                target.attrs["instruction"] = trim_instruction(source.attrs["instruction"], start, end, length)
        os.link(temporary, destination)  # Atomic, and refuses overwrite.
    finally:
        temporary.unlink(missing_ok=True)


def trim_dataset(src, dst=None, *, threshold="auto", min_frames=5, window_size=3,
                 percentile=5, block_frames=16, static_keys=(), dry_run=False):
    if any(type(value) is not int or value < 1 for value in (min_frames, window_size, block_frames)):
        raise ValueError("min_frames, window_size and block_frames must be positive integers")
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    src = Path(src).expanduser().resolve()
    dst = Path(dst).expanduser().resolve() if dst is not None else None
    if not dry_run and dst is None:
        raise ValueError("--dst is required unless --dry-run is used")
    if dst is not None and (dst == src or src in dst.parents or dst in src.parents):
        raise ValueError("Source and destination must be separate, non-nested directories")
    if dst is not None and dst.exists():
        raise FileExistsError(f"Destination must not exist: {dst}")
    static_keys = tuple(key.strip("/") for key in static_keys)
    if any(not key or key == "action_abs" for key in static_keys):
        raise ValueError("static_keys must be nonempty paths and cannot include action_abs")
    files = list((src / "episodes").glob("*.hdf5"))
    files.sort(key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.name)])
    if not files:
        raise ValueError(f"No episodes/*.hdf5 in {src}")
    if isinstance(threshold, str) and threshold.lower() == "auto":
        threshold, _ = estimate_threshold_from_dataset(
            (_read_action(path) for path in files), percentile=percentile, window_size=window_size)
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a nonnegative finite number or auto")
    plans = [_plan(path, threshold, min_frames, window_size, static_keys) for path in files]
    report = {"source": str(src), "threshold": threshold, "min_frames": min_frames,
              "window_size": window_size, "static_keys": static_keys, "episodes": plans}
    print(f"Threshold: {threshold:.6g}; episodes: {len(plans)}")
    for plan in plans:
        print(f"{plan['episode']}: {plan['original_length']} -> {plan['trimmed_length']} "
              f"(remove start={plan['trimmed_start']}, end={plan['trimmed_end']})")
    if dry_run:
        return report
    dst.mkdir(parents=True, exist_ok=False)
    try:
        for path, plan in zip(files, plans):
            write_episode(path, dst / "episodes" / path.name, plan, static_keys, block_frames)
        # Episode names do not change; these sidecars stay valid. Cached norm
        # stats and other derived metadata are deliberately NOT propagated.
        for name in ("dataset_split.json", "instruction_library.json"):
            if (src / name).is_file():
                with (src / name).open("rb") as source, (dst / name).open("xb") as target:
                    shutil.copyfileobj(source, target)
        with (dst / "trim_report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
    except BaseException:
        print(f"Processing interrupted; original data unchanged. Partial output retained at {dst}", file=sys.stderr)
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Existing dataset directory containing episodes/")
    parser.add_argument("--dst", help="New output dataset directory (must not exist)")
    parser.add_argument("--trim-threshold", "--trim_threshold", default="auto", help="Action L2 change threshold or auto")
    parser.add_argument("--trim-min-frames", "--trim_min_frames", type=int, default=5)
    parser.add_argument("--trim-window-size", "--trim_window_size", type=int, default=3)
    parser.add_argument("--percentile", type=float, default=5, help="Dataset-wide percentile for auto threshold")
    parser.add_argument("--block-frames", type=int, default=16, help="Raw frames copied per memory block")
    parser.add_argument("--static-keys", default="", help="Comma-separated non-temporal HDF5 datasets/groups")
    parser.add_argument("--dry-run", action="store_true", help="Print crop ranges without writing anything")
    args = parser.parse_args(argv)
    trim_dataset(args.src, args.dst, threshold=args.trim_threshold, min_frames=args.trim_min_frames,
                 window_size=args.trim_window_size, percentile=args.percentile, block_frames=args.block_frames,
                 static_keys=tuple(key.strip() for key in args.static_keys.split(",") if key.strip()),
                 dry_run=args.dry_run)


if __name__ == "__main__":
    main()
