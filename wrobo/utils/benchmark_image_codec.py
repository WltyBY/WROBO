"""
Benchmark decode performance of the three image storage layouts used by episode
HDF5 files, across sequential and random frame access:

    raw  (gzip)  -> Dataset/ACT_wrobo/sim_transfer_stack_cube_raw
    png          -> Dataset/ACT_wrobo/sim_transfer_stack_cube
    jpg (Q=95)   -> Dataset/ACT_wrobo/sim_transfer_stack_cube_JPG

What is timed
-------------
For every image dataset in every episode we measure the wall-clock cost of turning
stored data into a usable (C, H, W) uint8 RGB frame -- the exact thing the training
dataloader does:
  - raw : ds[i]                       (h5py read + gzip/shuffle decompress)
  - png : decode_frame(ds[i])         (h5py read of byte string + cv2.imdecode)
  - jpg : decode_frame(ds[i])         (h5py read of byte string + cv2.imdecode)
so the reported number is the apples-to-apples "time to get one decoded frame",
not just the codec call. Reads and decode are not split apart because for raw the
two are fused inside h5py and cannot be separated.

Access patterns
---------------
  - sequential : frames visited in index order 0..T-1 (best case for chunk/page cache)
  - random     : frames visited in a fixed shuffled order (worst case; what a shuffled
                 sampler actually does)
Both are measured per file, so the comparison is identical work in a different order.

Cache handling
--------------
By default each file's bytes are read into the OS page cache before timing ("warm"),
so the measurement isolates decode/access CPU cost from disk-IO jitter -- which also
matches steady-state training where files are re-read across epochs. Pass --cold to
drop the page cache before each file instead (needs `vmtouch` or root via
/proc/sys/vm/drop_caches; falls back to a warning if neither is available).

Usage
-----
    python -m wrobo.utils.benchmark_image_codec \
        --root Dataset/ACT_wrobo \
        --episodes 10 --frames-per-episode 200

Run with no limits to benchmark the whole dataset (slow).
"""

import argparse
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import h5py
import numpy as np

from wrobo.utils.image_codec import get_encoding, decode_frame, RAW


# (label, directory name) for the three layouts under --root
LAYOUTS = [
    ("raw (gzip)", "sim_transfer_stack_cube_raw"),
    ("png", "sim_transfer_stack_cube"),
    ("jpg (Q=95)", "sim_transfer_stack_cube_JPG"),
]


def _list_episodes(directory: str, limit: Optional[int]) -> List[str]:
    files = sorted(
        f for f in os.listdir(directory) if f.endswith(".hdf5")
    )
    paths = [os.path.join(directory, f) for f in files]
    if limit is not None:
        paths = paths[:limit]
    return paths


def _warm_cache(path: str) -> None:
    """Read the whole file once so its bytes live in the OS page cache."""
    with open(path, "rb") as fh:
        while fh.read(1 << 24):  # 16 MiB chunks
            pass


def _drop_cache(path: str) -> bool:
    """
    Best-effort eviction of `path` from the OS page cache. Returns True on success.

    Order of preference:
      1. os.posix_fadvise(POSIX_FADV_DONTNEED) -- drops just this file's pages, needs
         no privileges and no external tool. This is the portable, unprivileged path.
      2. vmtouch -e <path>                     -- same effect via an external tool.
      3. echo 1 > /proc/sys/vm/drop_caches     -- system-wide, needs root.
    Note: posix_fadvise only evicts pages already flushed to disk; for a read-only
    benchmark file that's always the case, so no fsync is needed.
    """
    if hasattr(os, "posix_fadvise"):
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                return True
            finally:
                os.close(fd)
        except OSError:
            pass
    if os.system(f"vmtouch -e '{path}' >/dev/null 2>&1") == 0:
        return True
    try:
        with open("/proc/sys/vm/drop_caches", "w") as fh:
            fh.write("1")
        return True
    except OSError:
        return False


class Accumulator:
    """Running totals for one (layout, access-pattern) combination, split per camera."""

    def __init__(self):
        # per image-key: total decode seconds, frame count, total decoded MB
        self.sec: Dict[str, float] = defaultdict(float)
        self.frames: Dict[str, int] = defaultdict(int)
        self.mb: Dict[str, float] = defaultdict(float)

    def add(self, key: str, seconds: float, n_frames: int, n_bytes: int):
        self.sec[key] += seconds
        self.frames[key] += n_frames
        self.mb[key] += n_bytes / 1e6

    def totals(self):
        sec = sum(self.sec.values())
        frames = sum(self.frames.values())
        mb = sum(self.mb.values())
        return sec, frames, mb


def _decode_all(ds: h5py.Dataset, encoding: str, indices: np.ndarray):
    """Decode the given frame indices in order; return (elapsed_s, n_frames, n_bytes)."""
    n_bytes = 0
    t0 = time.perf_counter()
    for i in indices:
        raw = ds[int(i)]
        frame = raw if encoding == RAW else decode_frame(raw)
        # touch the array so lazy work (if any) is forced and the frame is "used"
        n_bytes += frame.nbytes
    elapsed = time.perf_counter() - t0
    return elapsed, len(indices), n_bytes


def benchmark_layout(
    directory: str,
    episodes: Optional[int],
    frames_per_episode: Optional[int],
    cold: bool,
    seed: int,
) -> Dict[str, Accumulator]:
    """Benchmark one layout; returns {"sequential": Accumulator, "random": Accumulator}."""
    paths = _list_episodes(directory, episodes)
    if not paths:
        raise ValueError(f"No .hdf5 episodes found in {directory}")

    rng = np.random.default_rng(seed)
    out = {"sequential": Accumulator(), "random": Accumulator()}
    warned_cold = False

    for path in paths:
        if not cold:
            _warm_cache(path)

        with h5py.File(path, "r") as root:
            obs = root.get("observations")
            if obs is None:
                continue
            img_keys = [k for k in obs.keys() if "image" in k.lower()]
            for key in img_keys:
                ds = obs[key]
                encoding = get_encoding(ds)
                T = int(ds.shape[0])
                n = T if frames_per_episode is None else min(frames_per_episode, T)

                # same n frames, two orders: identical work, different access pattern.
                base = rng.choice(T, size=n, replace=False)
                seq_idx = np.sort(base)
                rnd_idx = base.copy()
                rng.shuffle(rnd_idx)

                for pattern, idx in (("sequential", seq_idx), ("random", rnd_idx)):
                    # In cold mode, evict the file before EACH pass so no pass is warmed
                    # by a previous one (sequential must not warm the file for random,
                    # nor one camera for the next). h5py's own chunk cache is per-open
                    # file, but the dominant cost here is the OS page cache.
                    if cold:
                        if not _drop_cache(path) and not warned_cold:
                            print("  [warn] could not drop page cache "
                                  "(posix_fadvise/vmtouch/root all unavailable); "
                                  "results will be warm-ish")
                            warned_cold = True
                    elapsed, nf, nb = _decode_all(ds, encoding, idx)
                    out[pattern].add(key, elapsed, nf, nb)

    return out


def _fmt_overall(label: str, acc: Dict[str, Accumulator]):
    """One summary line per access pattern: throughput in frames/s and MB/s."""
    lines = []
    for pattern in ("sequential", "random"):
        sec, frames, mb = acc[pattern].totals()
        if frames == 0 or sec == 0:
            lines.append(f"    {pattern:11s}: no data")
            continue
        fps = frames / sec
        mbps = mb / sec
        us = sec / frames * 1e6
        lines.append(
            f"    {pattern:11s}: {fps:9.1f} frames/s   {us:8.1f} us/frame   "
            f"{mbps:8.1f} MB/s decoded   ({frames} frames, {sec:.2f}s)"
        )
    return f"  {label}\n" + "\n".join(lines)


def _fmt_per_camera(acc: Dict[str, Accumulator]):
    """Per-camera us/frame, sequential vs random, so resolution effects are visible."""
    keys = sorted(set(acc["sequential"].sec) | set(acc["random"].sec))
    rows = []
    rows.append(f"    {'camera':28s} {'seq us/frame':>14s} {'rnd us/frame':>14s}")
    for key in keys:
        seq = acc["sequential"]
        rnd = acc["random"]
        seq_us = seq.sec[key] / seq.frames[key] * 1e6 if seq.frames[key] else 0.0
        rnd_us = rnd.sec[key] / rnd.frames[key] * 1e6 if rnd.frames[key] else 0.0
        rows.append(f"    {key:28s} {seq_us:14.1f} {rnd_us:14.1f}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="Dataset/ACT_wrobo",
                    help="parent dir holding the three layout subdirs")
    ap.add_argument("--episodes", type=int, default=None,
                    help="limit number of episodes per layout (default: all)")
    ap.add_argument("--frames-per-episode", type=int, default=None,
                    help="frames sampled per camera per episode (default: all)")
    ap.add_argument("--cold", action="store_true",
                    help="drop page cache before each file (cold-read timing)")
    ap.add_argument("--seed", type=int, default=319)
    ap.add_argument("--per-camera", action="store_true",
                    help="also print a per-camera (resolution) breakdown")
    args = ap.parse_args()

    print(f"Image codec decode benchmark  (cache={'cold' if args.cold else 'warm'}, "
          f"episodes={'all' if args.episodes is None else args.episodes}, "
          f"frames/ep={'all' if args.frames_per_episode is None else args.frames_per_episode})")
    print("=" * 78)

    results = {}
    for label, subdir in LAYOUTS:
        directory = os.path.join(args.root, subdir)
        if not os.path.isdir(directory):
            print(f"  [skip] {label}: missing {directory}")
            continue
        print(f"\nbenchmarking {label}  ({directory}) ...", flush=True)
        t0 = time.perf_counter()
        acc = benchmark_layout(
            directory, args.episodes, args.frames_per_episode, args.cold, args.seed
        )
        results[label] = acc
        print(f"  done in {time.perf_counter() - t0:.1f}s")

    print("\n" + "=" * 78)
    print("RESULTS (time to obtain one decoded (C,H,W) uint8 frame)")
    print("=" * 78)
    for label, _ in LAYOUTS:
        if label in results:
            print(_fmt_overall(label, results[label]))
            if args.per_camera:
                print(_fmt_per_camera(results[label]))
            print()


if __name__ == "__main__":
    # Just a toy benchmark to test the relative decode performance of the three image storage layouts
    main()
    
