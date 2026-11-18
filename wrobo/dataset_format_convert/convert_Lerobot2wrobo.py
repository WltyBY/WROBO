#!/usr/bin/env python
"""
Convert LeRobot dataset format to wcode (HDF5) format.

LeRobot format:
- Parquet files for data (action, observation.state, metadata)
- MP4 videos for images
- JSON metadata files

wcode format:
- HDF5 files with structure:
  - /observations/image_{camera_name}: encoded images per frame
  - /observations/proprio_state: joint positions (T, D)
  - /action_abs: actions (T, D)
  - Attributes: dataset, success, comment, seq_len
"""

import os
import sys
import json
import re
from pathlib import Path
import contextlib
from multiprocessing import Pool, cpu_count
from functools import partial

# Suppress FFmpeg warnings before importing cv2
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

# Video decoding
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: cv2 not found. Will try pyav for video decoding.")

try:
    import av
    av.logging.set_level(av.logging.PANIC)  # Suppress all PyAV warnings
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

from wrobo.utils.image_codec import create_image_dataset
from wrobo.tools.trim_wrobo_dataset import (
    estimate_threshold_from_dataset,
    trim_episode_data,
)


@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    null_fd = os.open(os.devnull, os.O_RDWR)
    save_stderr = os.dup(2)
    os.dup2(null_fd, 2)
    try:
        yield
    finally:
        os.dup2(save_stderr, 2)
        os.close(null_fd)
        os.close(save_stderr)


def load_lerobot_metadata(dataset_path):
    """Load lerobot metadata from info.json."""
    info_path = Path(dataset_path) / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    return info


def load_episode_metadata(dataset_path):
    """Load episode metadata from parquet files."""
    episodes_dir = Path(dataset_path) / "meta" / "episodes"

    # Find all episode metadata files
    episode_files = sorted(episodes_dir.rglob("*.parquet"))

    if not episode_files:
        raise FileNotFoundError(f"No episode metadata found in {episodes_dir}")

    # Load all episode metadata
    tables = [pq.read_table(f) for f in episode_files]
    episodes_table = tables[0] if len(tables) == 1 else pq.concat_tables(tables)

    # Convert to list of dicts
    episodes = []
    for i in range(len(episodes_table)):
        ep_dict = {col: episodes_table[col][i].as_py() for col in episodes_table.column_names}
        episodes.append(ep_dict)

    return episodes


def load_data_parquet(dataset_path):
    """Load all data from parquet files."""
    data_dir = Path(dataset_path) / "data"

    # Find all data files
    data_files = sorted(data_dir.rglob("*.parquet"))

    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    # Load all data
    tables = [pq.read_table(f) for f in data_files]
    data_table = tables[0] if len(tables) == 1 else pq.concat_tables(tables)

    return data_table


def decode_video_opencv(video_path, start_idx=0, end_idx=None):
    """Decode video frames using OpenCV."""
    if not HAS_CV2:
        raise ImportError("OpenCV (cv2) is required for video decoding")

    with suppress_stderr():
        cap = cv2.VideoCapture(str(video_path))

        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        frame_idx = 0

        # Seek to start position
        if start_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            frame_idx = start_idx

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if end_idx is not None and frame_idx >= end_idx:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_idx += 1

        cap.release()

    return frames


def decode_video_pyav(video_path, start_idx=0, end_idx=None):
    """Decode video frames using PyAV."""
    if not HAS_PYAV:
        raise ImportError("PyAV is required for video decoding")

    container = av.open(str(video_path))
    stream = container.streams.video[0]

    frames = []
    frame_idx = 0
    try:
        for frame in container.decode(stream):
            if frame_idx >= start_idx:
                if end_idx is not None and frame_idx >= end_idx:
                    break
                # Convert to numpy array (RGB)
                try:
                    img = frame.to_ndarray(format='rgb24')
                    frames.append(img)
                except Exception:
                    pass  # Silently skip failed frames
            frame_idx += 1
    except av.error.InvalidDataError:
        pass  # Silently handle decode errors
    finally:
        container.close()

    return frames


def decode_video_frames(video_path, start_idx=0, end_idx=None):
    """Decode video frames using available backend."""
    # Try OpenCV first (better AV1 support)
    if HAS_CV2:
        try:
            frames = decode_video_opencv(video_path, start_idx, end_idx)
            if len(frames) > 0:
                return frames
        except Exception as e:
            print(f"OpenCV decoding failed: {e}, trying PyAV...")

    # Fallback to PyAV
    if HAS_PYAV:
        try:
            return decode_video_pyav(video_path, start_idx, end_idx)
        except Exception as e:
            print(f"PyAV decoding failed: {e}")
            return []

    raise ImportError("Either OpenCV or PyAV is required for video decoding")


def convert_episode_worker(args):
    """
    Worker function for multiprocessing.
    Loads data_table inside the worker to avoid pickling large objects.

    Args:
        args: Tuple of (dataset_path, episode_idx, episode_meta, output_path, kwargs)

    Returns:
        Tuple of (episode_idx, success, error_msg, stats_dict)
    """
    dataset_path, episode_idx, episode_meta, output_path, kwargs = args

    try:
        # Load data table in worker process
        data_table = load_data_parquet(dataset_path)

        # Call the actual conversion function
        stats = convert_episode(
            dataset_path,
            episode_idx,
            episode_meta,
            data_table,
            output_path,
            **kwargs
        )
        return (episode_idx, True, None, stats)
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return (episode_idx, False, error_msg, None)


def convert_episode(
    dataset_path,
    episode_idx,
    episode_meta,
    data_table,
    output_path,
    dataset_name="LeRobot",
    success=True,
    comment="",
    img_encoding="png",
    jpeg_quality=95,
    camera_name_mapping=None,
    trim_static=False,
    trim_threshold=0.01,
    trim_min_frames=5,
    trim_window_size=3,
    overwrite=True,
):
    """
    Convert a single episode from LeRobot format to wcode HDF5 format.

    Args:
        dataset_path: Path to LeRobot dataset root
        episode_idx: Episode index
        episode_meta: Episode metadata dict
        data_table: Full data table (parquet)
        output_path: Output HDF5 file path
        dataset_name: Dataset name attribute
        success: Success flag attribute
        comment: Comment attribute
        img_encoding: Image encoding format ('png', 'jpeg', 'gzip')
        jpeg_quality: JPEG quality if using jpeg encoding
        camera_name_mapping: Dict to map camera names (e.g., {'fixed': 'wrist', 'wrist': 'fixed'})
        trim_static: Whether to trim static frames at beginning and end
        trim_threshold: Threshold for action change magnitude (L2 norm), or "auto"
                           to estimate one from the selected dataset episodes
        trim_min_frames: Minimum number of frames to keep even if all static
        trim_window_size: Size of sliding window for smoothing velocity signal
        overwrite: Whether an existing output file may be overwritten

    Returns:
        Dict with conversion statistics
    """
    dataset_path = Path(dataset_path)

    # Get episode data range
    start_idx = episode_meta['dataset_from_index']
    end_idx = episode_meta['dataset_to_index']
    seq_len = episode_meta['length']

    # Statistics to return
    stats = {
        'episode_idx': episode_idx,
        'original_length': seq_len,
        'final_length': seq_len,
        'cameras': [],
        'trimmed': False,
        'trim_start': 0,
        'trim_end': 0,
    }

    # Extract episode data from parquet
    episode_data = {}
    for col in data_table.column_names:
        col_data = data_table[col][start_idx:end_idx]
        if hasattr(col_data, 'to_pylist'):
            episode_data[col] = col_data.to_pylist()
        else:
            episode_data[col] = [col_data[i].as_py() for i in range(len(col_data))]

    # Convert lists to numpy arrays
    action = np.array(episode_data['action'], dtype=np.float32)  # (T, action_dim)
    proprio_state = np.array(episode_data['observation.state'], dtype=np.float32)  # (T, state_dim)

    # Direct callers may pass auto as well; the dataset-level conversion path
    # resolves it once for all episodes before workers are launched.
    if isinstance(trim_threshold, str):
        if trim_threshold.strip().lower() == "auto":
            trim_threshold, _ = estimate_threshold_from_dataset(
                [action], window_size=trim_window_size
            )
        else:
            try:
                trim_threshold = float(trim_threshold)
            except ValueError as exc:
                raise ValueError(
                    "trim_threshold must be a number or 'auto'"
                ) from exc

    # Load video frames for each camera
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)

    # Find camera keys
    camera_keys = [k for k in info['features'].keys() if k.startswith('observation.images.')]

    video_frames = {}
    for cam_key in camera_keys:
        # Get video file path
        video_chunk_idx = episode_meta[f'videos/{cam_key}/chunk_index']
        video_file_idx = episode_meta[f'videos/{cam_key}/file_index']

        video_path = dataset_path / "videos" / cam_key / f"chunk-{video_chunk_idx:03d}" / f"file-{video_file_idx:03d}.mp4"

        if not video_path.exists():
            print(f"Warning: Video file not found: {video_path}")
            continue

        # Calculate frame range within the video file
        # The video file might contain multiple episodes
        from_ts = episode_meta[f'videos/{cam_key}/from_timestamp']
        to_ts = episode_meta[f'videos/{cam_key}/to_timestamp']
        fps = info['fps']

        # Calculate frame indices more accurately
        # from_timestamp is relative to the start of the video file
        video_start_frame = int(round(from_ts * fps))
        video_end_frame = int(round(to_ts * fps))

        # Ensure we get exactly seq_len frames
        expected_frames = video_end_frame - video_start_frame
        if expected_frames != seq_len:
            # Adjust end frame to match seq_len
            video_end_frame = video_start_frame + seq_len

        # Decode video frames
        frames = decode_video_frames(video_path, video_start_frame, video_end_frame)

        # Ensure we got the right number of frames
        if len(frames) != seq_len:
            # Pad or trim as needed
            if len(frames) < seq_len:
                # Pad with last frame
                while len(frames) < seq_len:
                    frames.append(frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8))
            else:
                frames = frames[:seq_len]

        # Extract camera name from key (e.g., observation.images.fixed -> fixed)
        cam_name = cam_key.split('.')[-1]

        # Apply camera name mapping if provided
        if camera_name_mapping and cam_name in camera_name_mapping:
            cam_name = camera_name_mapping[cam_name]

        stats['cameras'].append(cam_name)
        video_frames[cam_name] = frames

    # Trim static frames if enabled
    if trim_static:
        trimmed = trim_episode_data(
            action=action,
            proprio_state=proprio_state,
            video_frames=video_frames,
            threshold=trim_threshold,
            min_frames=trim_min_frames,
            window_size=trim_window_size,
        )

        action = trimmed['action']
        proprio_state = trimmed['proprio_state']
        video_frames = trimmed['video_frames']
        seq_len = len(action)

        if trimmed['trimmed']:
            trim_stats = trimmed['stats']
            stats['trimmed'] = True
            stats['trim_start'] = trim_stats['trimmed_start']
            stats['trim_end'] = trim_stats['trimmed_end']
            stats['final_length'] = trim_stats['trimmed_length']

    # Create HDF5 file
    with h5py.File(output_path, "w" if overwrite else "x", rdcc_nbytes=1024**2 * 2) as f:
        # Set attributes
        f.attrs["dataset"] = dataset_name
        f.attrs["success"] = success
        f.attrs["comment"] = comment
        f.attrs["seq_len"] = seq_len

        # Create observations group
        obs_grp = f.create_group("observations")

        # Save images for each camera
        for cam_name, frames in video_frames.items():
            # Convert frames to (C, H, W) format
            frames_chw = [np.transpose(frame, (2, 0, 1)) for frame in frames]
            create_image_dataset(
                obs_grp,
                f"image_{cam_name}",
                frames_chw,
                encoding=img_encoding,
                jpeg_quality=jpeg_quality,
            )

        # Save proprio_state
        obs_grp.create_dataset(
            "proprio_state",
            data=proprio_state,
            dtype="float32",
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            chunks=(min(50, seq_len), proprio_state.shape[1]),
        )

        # Save action
        f.create_dataset(
            "action_abs",
            data=action,
            dtype="float32",
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            chunks=(min(50, seq_len), action.shape[1]),
        )

    return stats


def convert_lerobot_dataset(
    src_path,
    dst_path,
    comment="lerobot_converted",
    dataset_name="LeRobot",
    success=True,
    img_encoding="png",
    jpeg_quality=95,
    episode_indices=None,
    camera_name_mapping=None,
    trim_static=False,
    trim_threshold=0.01,
    trim_min_frames=5,
    trim_window_size=3,
    num_workers=None,
    append=False,
):
    """
    Convert entire LeRobot dataset to wcode format using multiprocessing.

    Args:
        src_path: Path to LeRobot dataset root
        dst_path: Output directory for HDF5 files
        comment: Comment to add to HDF5 files
        dataset_name: Dataset name attribute
        success: Success flag for all episodes
        img_encoding: Image encoding format ('png', 'jpeg', 'gzip')
        jpeg_quality: JPEG quality if using jpeg encoding
        episode_indices: List of episode indices to convert (None = all)
        camera_name_mapping: Dict to map camera names (e.g., {'fixed': 'wrist', 'wrist': 'fixed'})
        trim_static: Whether to trim static frames at beginning and end
        trim_threshold: Threshold for action change magnitude (L2 norm), or "auto"
                           to estimate one from the selected dataset episodes
        trim_min_frames: Minimum number of frames to keep even if all static
        trim_window_size: Size of sliding window for smoothing velocity signal
        num_workers: Number of worker processes (None = auto-detect)
        append: Write new episodes consecutively after the highest existing
                episode number instead of overwriting files with source indices

    Returns:
        List of successfully converted HDF5 file paths, in metadata order
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)

    # Create episodes subdirectory for trajectory files
    episodes_dir = dst_path / "episodes"
    episodes_dir.mkdir(exist_ok=True)

    next_episode_idx = 0
    if append:
        for path in episodes_dir.glob("episode_*.hdf5"):
            match = re.fullmatch(r"episode_(\d+)\.hdf5", path.name)
            if match:
                next_episode_idx = max(next_episode_idx, int(match.group(1)) + 1)
        print(f"Append mode: new files start at episode_{next_episode_idx}.hdf5")

    # Load metadata
    print(f"Loading metadata from {src_path}...")
    info = load_lerobot_metadata(src_path)
    episodes = load_episode_metadata(src_path)

    print(f"Dataset info:")
    print(f"  Total episodes: {info['total_episodes']}")
    print(f"  Total frames: {info['total_frames']}")
    print(f"  FPS: {info['fps']}")
    print(f"  Robot type: {info.get('robot_type', 'N/A')}")

    # Filter episodes if specified
    if episode_indices is not None:
        episodes = [ep for ep in episodes if ep['episode_index'] in episode_indices]
        print(f"Converting {len(episodes)} episodes (filtered)")
    else:
        print(f"Converting all {len(episodes)} episodes")

    if not episodes:
        raise ValueError("No episodes selected for conversion")

    # Resolve the dataset-level automatic threshold before starting workers so
    # every episode uses the same movement criterion.
    if isinstance(trim_threshold, str):
        if trim_threshold.strip().lower() != "auto":
            try:
                trim_threshold = float(trim_threshold)
            except ValueError as exc:
                raise ValueError(
                    "trim_threshold must be a number or 'auto'"
                ) from exc
        else:
            data_table = load_data_parquet(src_path)
            actions = []
            for episode_meta in episodes:
                start = episode_meta['dataset_from_index']
                end = episode_meta['dataset_to_index']
                action_data = data_table['action'][start:end].to_pylist()
                actions.append(np.asarray(action_data, dtype=np.float32))
            trim_threshold, threshold_stats = estimate_threshold_from_dataset(
                actions,
                window_size=trim_window_size,
            )
            print(
                f"Auto trim threshold: {trim_threshold:.6g} "
                f"(p5, min={threshold_stats.get('min', 0.0):.6g}, "
                f"median={threshold_stats.get('median', 0.0):.6g}, "
                f"max={threshold_stats.get('max', 0.0):.6g})"
            )

    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    print(f"Using {num_workers} worker processes")

    # Prepare worker arguments
    kwargs = {
        'dataset_name': dataset_name,
        'success': success,
        'comment': comment,
        'img_encoding': img_encoding,
        'jpeg_quality': jpeg_quality,
        'camera_name_mapping': camera_name_mapping,
        'trim_static': trim_static,
        'trim_threshold': trim_threshold,
        'trim_min_frames': trim_min_frames,
        'trim_window_size': trim_window_size,
        'overwrite': not append,
    }

    worker_args = []
    for offset, episode_meta in enumerate(episodes):
        episode_idx = episode_meta['episode_index']
        output_idx = next_episode_idx + offset if append else episode_idx
        output_file = episodes_dir / f"episode_{output_idx}.hdf5"
        worker_args.append((src_path, episode_idx, episode_meta, output_file, kwargs))

    # Process episodes in parallel
    failed_episodes = []
    error_details = {}  # Store error messages for failed episodes
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(worker_args), desc="Converting episodes") as pbar:
            for episode_idx, success_flag, error_msg, ep_stats in pool.imap_unordered(convert_episode_worker, worker_args):
                if success_flag and ep_stats:
                    # Build postfix string with stats
                    postfix_parts = []
                    if ep_stats['trimmed']:
                        postfix_parts.append(f"trim:{ep_stats['trim_start']}+{ep_stats['trim_end']}")
                        postfix_parts.append(f"len:{ep_stats['original_length']}->{ep_stats['final_length']}")
                    else:
                        postfix_parts.append(f"len:{ep_stats['final_length']}")
                    postfix_parts.append(f"cam:{','.join(ep_stats['cameras'])}")
                    pbar.set_postfix_str(' | '.join(postfix_parts))
                elif not success_flag:
                    failed_episodes.append(episode_idx)
                    error_details[episode_idx] = error_msg
                    pbar.set_postfix_str(f"FAILED ep_{episode_idx}")
                pbar.update(1)

    # Check for failures and print errors
    if failed_episodes:
        print("\nErrors occurred during conversion:")
        for episode_idx in failed_episodes:
            print(f"\nEpisode {episode_idx}:")
            print(error_details[episode_idx])

    # Summary
    print(f"\nConversion complete!")
    print(f"  Successful: {len(worker_args) - len(failed_episodes)}/{len(worker_args)}")
    if failed_episodes:
        print(f"  Failed episodes: {failed_episodes}")
    print(f"  Files saved to {episodes_dir}")

    return [
        output_file
        for _, episode_idx, _, output_file, _ in worker_args
        if episode_idx not in error_details
    ]



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to wrobo HDF5 format")
    parser.add_argument("--src", type=str, default="./Dataset/human/mode1_human_suffix_930_3",
                        help="Source LeRobot dataset path")
    parser.add_argument("--dst", type=str, default="./Dataset/SO101/blue_cube_pick_place_blue_paper_tray_human_intervention",
                        help="Destination directory for HDF5 files")
    parser.add_argument("--append", action="store_true",
                        help="Append after the highest existing episode number without overwriting files")
    parser.add_argument("--comment", type=str, default="so101_pick_and_place_blue_cube_human_intervention",
                        help="Comment to add to HDF5 files")
    parser.add_argument("--dataset_name", type=str, default="LeRobot_SO101",
                        help="Dataset name attribute")
    parser.add_argument("--encoding", type=str, default="jpeg", choices=["png", "jpeg", "gzip"],
                        help="Image encoding format")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="JPEG quality (if using jpeg encoding)")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Comma-separated episode indices to convert (e.g., '0,1,2'). Default: all")
    parser.add_argument("--camera_mapping", type=str, default="fixed:angle,wrist:wrist",
                        help="Camera name mapping in format 'src1:dst1,src2:dst2' (e.g., 'fixed:wrist,wrist:fixed')")
    parser.add_argument("--trim_static", action="store_true",
                        help="Enable trimming of static frames at beginning and end of episodes")
    parser.add_argument("--trim_threshold", type=str, default="auto",
                        help="Threshold for action change magnitude, or 'auto' to estimate it from the dataset")
    parser.add_argument("--trim_min_frames", type=int, default=5,
                        help="Minimum number of frames to keep even if all are static")
    parser.add_argument("--trim_window_size", type=int, default=3,
                        help="Size of sliding window for smoothing velocity signal (helps reduce noise)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker processes (default: 8, can be None for auto-detect)")
    args = parser.parse_args()

    # Parse episode indices
    episode_indices = None
    if args.episodes:
        episode_indices = [int(x.strip()) for x in args.episodes.split(',')]

    # Parse camera name mapping
    camera_name_mapping = None
    if args.camera_mapping:
        camera_name_mapping = {}
        for pair in args.camera_mapping.split(','):
            src, dst = pair.strip().split(':')
            camera_name_mapping[src.strip()] = dst.strip()
        print(f"Camera name mapping: {camera_name_mapping}")

    # Convert dataset
    output_files = convert_lerobot_dataset(
        args.src,
        args.dst,
        comment=args.comment,
        dataset_name=args.dataset_name,
        img_encoding=args.encoding,
        jpeg_quality=args.jpeg_quality,
        episode_indices=episode_indices,
        camera_name_mapping=camera_name_mapping,
        trim_static=args.trim_static,
        trim_threshold=args.trim_threshold,
        trim_min_frames=args.trim_min_frames,
        trim_window_size=args.trim_window_size,
        num_workers=args.num_workers,
        append=args.append,
    )

    # Verify first converted file
    if output_files:
        first_file = output_files[0]
        print("\nVerifying first episode...")
        from wrobo.tools.vis_trajectory import print_h5_structure
        print_h5_structure(str(first_file))
