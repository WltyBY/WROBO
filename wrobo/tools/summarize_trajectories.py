#!/usr/bin/env python
"""Summarize episode frame counts and durations in trajectory_info.json.

Duration is the span from the first recorded frame to the last, in seconds.
Prefer actual timestamps; without timestamps, estimate (frame_count - 1) / FPS.
Only dataset/episodes/*.hdf5 and *.h5 are counted. Image data is never decoded.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np


def _frame_count(file):
    lengths = {}
    for name in ("action_abs", "observations/proprio_state", "timestamp"):
        if name in file:
            dataset = file[name]
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim == 0:
                raise ValueError(f"{name} must be a dataset with a frame dimension")
            lengths[name] = dataset.shape[0]
    observations = file.get("observations")
    if isinstance(observations, h5py.Group):
        for name, dataset in observations.items():
            if name.startswith("image_") and isinstance(dataset, h5py.Dataset):
                if dataset.ndim == 0:
                    raise ValueError(f"observations/{name} has no frame dimension")
                lengths[f"observations/{name}"] = dataset.shape[0]
    if "seq_len" in file.attrs:
        value = file.attrs["seq_len"]
        if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
            raise ValueError("seq_len must be a nonnegative integer")
        count = int(value)
        if count < 0 or count != float(value):
            raise ValueError("seq_len must be a nonnegative integer")
        lengths["seq_len"] = count
    if not lengths:
        raise ValueError("No frame datasets or seq_len attribute found")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent frame counts: {lengths}")
    return int(next(iter(lengths.values())))


def read_episode_info(path, fallback_fps=None):
    """Read one HDF5 episode without loading its state, action or image arrays."""
    path = Path(path)
    try:
        with h5py.File(path, "r") as file:
            frame_count = _frame_count(file)
            if "timestamp" in file:
                dataset = file["timestamp"]
                if dataset.ndim != 1:
                    raise ValueError("timestamp must be one-dimensional")
                unit = dataset.attrs.get("unit", "seconds")
                if isinstance(unit, bytes):
                    unit = unit.decode("utf-8")
                if str(unit).lower() not in ("seconds", "second", "s"):
                    raise ValueError(f"Unsupported timestamp unit: {unit!r}; expected seconds")
                timestamps = np.asarray(dataset[:], dtype=np.float64)
                if not np.all(np.isfinite(timestamps)) or np.any(timestamps < 0):
                    raise ValueError("timestamp must contain finite nonnegative seconds")
                if np.any(np.diff(timestamps) < 0):
                    raise ValueError("timestamp must be nondecreasing")
                duration = float(timestamps[-1] - timestamps[0]) if frame_count > 1 else 0.0
                source = "timestamp_span"
            else:
                source = "fps_estimate"
                duration = 0.0
                if frame_count > 1:
                    fps = file.attrs.get("FPS", file.attrs.get("fps", fallback_fps))
                    if fps is None:
                        raise ValueError("No timestamp or FPS found; provide --fps to estimate duration")
                    if isinstance(fps, (bool, np.bool_)) or np.ndim(fps) != 0:
                        raise ValueError("FPS must be a finite positive number")
                    fps = float(fps)
                    if not math.isfinite(fps) or fps <= 0:
                        raise ValueError("FPS must be a finite positive number")
                    duration = (frame_count - 1) / fps
            if not math.isfinite(duration):
                raise ValueError("Computed duration is not finite")
            return {
                "frame_count": frame_count,
                "duration_seconds": duration,
                "duration_source": source,
            }
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _natural_key(path):
    return [(1, int(part)) if part.isdigit() else (0, part.lower())
            for part in re.split(r"(\d+)", path.name)]


def _write_statistics(path, statistics):
    existing = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if not isinstance(existing, dict):
            raise ValueError(f"{path} must contain a JSON object")
    existing.update(statistics)
    # Replacing the complete file keeps old results intact if writing fails.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(existing, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def summarize_trajectories(target, fallback_fps=None, dry_run=False):
    """Return per-episode and total statistics, merging them into the dataset JSON.

    target accepts a dataset root or its episodes/ directory. Existing unrelated
    top-level JSON fields are preserved; summary and episodes are recomputed.
    A bad episode aborts the operation before the output is changed.
    """
    if fallback_fps is not None:
        if isinstance(fallback_fps, (bool, np.bool_)) or np.ndim(fallback_fps) != 0:
            raise ValueError("--fps must be a finite positive number")
        fallback_fps = float(fallback_fps)
        if not math.isfinite(fallback_fps) or fallback_fps <= 0:
            raise ValueError("--fps must be a finite positive number")
    target = Path(target).expanduser().resolve()
    dataset_dir = target.parent if target.name == "episodes" else target
    episodes_dir = dataset_dir / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"Episodes directory does not exist: {episodes_dir}")
    paths = sorted(
        (path for path in episodes_dir.iterdir()
         if path.is_file() and path.suffix.lower() in (".hdf5", ".h5")),
        key=_natural_key,
    )
    episodes = {path.name: read_episode_info(path, fallback_fps) for path in paths}
    statistics = {
        "summary": {
            "episode_count": len(episodes),
            "total_frames": sum(info["frame_count"] for info in episodes.values()),
            "total_duration_seconds": math.fsum(info["duration_seconds"] for info in episodes.values()),
        },
        "episodes": episodes,
    }
    if not dry_run:
        _write_statistics(dataset_dir / "trajectory_info.json", statistics)
    return statistics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="Dataset root or its episodes/ directory")
    parser.add_argument("--fps", type=float, default=None,
                        help="Fallback FPS when an episode has neither timestamps nor FPS metadata")
    parser.add_argument("--dry-run", action="store_true", help="Print statistics without writing JSON")
    args = parser.parse_args(argv)
    try:
        statistics = summarize_trajectories(args.target, args.fps, args.dry_run)
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    for name, info in statistics["episodes"].items():
        print(f"{name}: {info['frame_count']} frames, {info['duration_seconds']:.3f} s")
    summary = statistics["summary"]
    print(f"Total: {summary['episode_count']} episodes, {summary['total_frames']} frames, "
          f"{summary['total_duration_seconds']:.3f} s")
    target = Path(args.target).expanduser().resolve()
    output = (target.parent if target.name == "episodes" else target) / "trajectory_info.json"
    print(f"{'[DRY RUN] Would update' if args.dry_run else 'Updated'} {output}")


if __name__ == "__main__":
    main()
