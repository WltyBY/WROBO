"""Resolve complete episode membership independently of language rendering."""

import json
from pathlib import Path
import re


def _validated_task_ids(task_ids):
    if task_ids is None:
        return None
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError("task_ids must be None or a nonempty list of nonnegative integers")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in task_ids):
        raise ValueError("task_ids must contain only nonnegative integers")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must not contain duplicate task IDs")
    return sorted(task_ids)


def normalize_task_ids(tokens, count):
    """Parse one CLI token per dataset: 'all', '0', or '0,2'."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("dataset_dir must contain at least one dataset")
    if tokens is None:
        return [None] * count
    if not isinstance(tokens, list) or len(tokens) != count:
        raise ValueError(f"task_ids must contain one token per dataset_dir ({count} entries)")
    selections = []
    for index, token in enumerate(tokens):
        if not isinstance(token, str):
            raise ValueError(f"task_ids token at index {index} must be a string")
        token = token.strip()
        if token == "all":
            selections.append(None)
            continue
        values = [value.strip() for value in token.split(",")]
        if not all(re.fullmatch(r"[0-9]+", value) for value in values):
            raise ValueError(f"Invalid task_ids token at index {index}: {token!r}; use 'all' or comma-separated nonnegative integers")
        selections.append(_validated_task_ids([int(value) for value in values]))
    return selections


def _read_object(path):
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _episode_list(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of episode IDs")
    for episode_id in value:
        if (not isinstance(episode_id, str) or not episode_id.strip()
                or episode_id in (".", "..") or "/" in episode_id or "\\" in episode_id
                or episode_id.endswith(".hdf5")):
            raise ValueError(f"{label} contains invalid episode ID {episode_id!r}; use filenames without .hdf5")
    return list(value)


def _require_episode_files(directory, episode_ids, label):
    for episode_id in dict.fromkeys(episode_ids):
        path = directory / "episodes" / (episode_id + ".hdf5")
        if not path.is_file():
            raise FileNotFoundError(f"{label} references an episode file that does not exist: {path}")


def resolve_episode_ids(dataset_dir, fold, split, task_ids=None):
    """Keep split order while selecting complete trajectories from task mappings.

    All-task selection does not open the library. No selection opens HDF5 files
    or interprets episode/frame annotations; it only checks file existence.
    """
    task_ids = _validated_task_ids(task_ids)
    if split not in ("train", "val", "test"):
        raise ValueError("split must be 'train', 'val' or 'test'")
    directory = Path(dataset_dir)
    split_path = directory / "dataset_split.json"
    membership = _read_object(split_path)
    fold = str(fold)
    try:
        if split == "test":
            ids = _episode_list(membership["test"], f"{split_path}:test")
        elif fold == "all":
            ids = _episode_list(membership["0"]["train"], f"{split_path}:0/train")
            ids += _episode_list(membership["0"]["val"], f"{split_path}:0/val")
            ids = list(dict.fromkeys(ids))
        else:
            ids = _episode_list(membership[fold][split], f"{split_path}:{fold}/{split}")
    except (KeyError, TypeError) as error:
        raise ValueError(f"{split_path} has no valid {split!r} list for fold {fold!r}") from error

    if task_ids is not None:
        library_path = directory / "instruction_library.json"
        library = _read_object(library_path)
        selected = set()
        for task_id in task_ids:
            task = library.get(str(task_id))
            if not isinstance(task, dict):
                raise ValueError(f"Selected task {task_id} is missing or invalid in {library_path}")
            episodes = _episode_list(task.get("episodes"), f"{library_path}:task {task_id}.episodes")
            if not episodes:
                raise ValueError(f"Selected task {task_id} has an empty episodes list in {library_path}")
            _require_episode_files(directory, episodes, f"Selected task {task_id} in {library_path}")
            selected.update(episodes)
        ids = list(dict.fromkeys(episode_id for episode_id in ids if episode_id in selected))
    _require_episode_files(directory, ids, f"{split_path}:{fold}/{split}")
    if not ids and split in ("train", "val"):
        selection = "all tasks" if task_ids is None else f"task_ids={task_ids}"
        raise ValueError(f"No episodes remain for {directory}, fold={fold!r}, split={split!r}, {selection}")
    return ids
