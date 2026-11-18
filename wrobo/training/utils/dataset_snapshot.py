"""Save the actual per-fold dataset membership used by a training run."""

import json
import os

from wrobo.utils.file_operations import save_json


def _episode_groups(dataset):
    groups = [[] for _ in dataset.dataset_dirs]
    for episode_id in dataset.episode_ids:
        if dataset.is_multi_dataset:
            prefix, episode_id = episode_id.split("_", 1)
            index = int(prefix[len("dataset"):])
        else:
            index = 0
        groups[index].append(episode_id)
    return groups


def save_dataset_snapshot(train_dataset, val_dataset, output_dir):
    """Save resolved membership and byte-for-byte copies of source libraries."""
    if train_dataset.dataset_dirs != val_dataset.dataset_dirs:
        raise ValueError("Train/val dataset order differs")
    if train_dataset.fold != val_dataset.fold:
        raise ValueError("Train/val folds differ")
    if train_dataset.use_instruction != val_dataset.use_instruction:
        raise ValueError("Train/val instruction settings differ")
    if train_dataset.instruction_modes != val_dataset.instruction_modes:
        raise ValueError("Train/val instruction modes differ")
    if train_dataset.task_ids != val_dataset.task_ids:
        raise ValueError("Train/val task selections differ")
    if train_dataset.normalization_mode != val_dataset.normalization_mode:
        raise ValueError("Train/val normalization modes differ")
    train_groups = _episode_groups(train_dataset)
    val_groups = _episode_groups(val_dataset)
    snapshot = {"fold": str(train_dataset.fold),
                "normalization_mode": train_dataset.normalization_mode, "datasets": []}
    libraries = {}
    for index, source in enumerate(train_dataset.dataset_dirs):
        dataset_id = f"dataset{index}"
        library_name = None
        mode = None
        if train_dataset.use_instruction:
            mode = train_dataset.instruction_modes[index]
            library_name = f"{dataset_id}_instruction_library.json"
            provider = train_dataset.instruction_providers[source]
            library = provider.library
            if library != val_dataset.instruction_providers[source].library:
                raise ValueError(f"Train/val instruction libraries differ for {source}")
            with open(provider.library_path, "rb") as file:
                source_bytes = file.read()
            if json.loads(source_bytes.decode("utf-8")) != library:
                raise ValueError(f"Instruction library changed since it was loaded: {provider.library_path}")
            libraries[library_name] = source_bytes
        snapshot["datasets"].append({
            "dataset_id": dataset_id,
            "source": os.path.abspath(source),
            # Preserve the exact norm_stats.json key independently of host/cwd.
            "norm_stats_key": os.fspath(source),
            "task_ids": train_dataset.task_ids[index],
            "instruction_mode": mode,
            "instruction_library": library_name,
            "train": train_groups[index],
            "val": val_groups[index],
        })
    os.makedirs(output_dir, exist_ok=True)
    for filename, source_bytes in libraries.items():
        with open(os.path.join(output_dir, filename), "wb") as file:
            file.write(source_bytes)
    save_json(snapshot, os.path.join(output_dir, "dataset_split.json"))
    return snapshot


def resolve_eval_instruction_library(artifact_dir):
    """Select a unique library from a run snapshot; never guess among datasets."""
    manifest_path = os.path.join(artifact_dir, "dataset_split.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)
        if "datasets" in manifest:
            names = [entry["instruction_library"] for entry in manifest["datasets"]
                     if entry.get("instruction_library") is not None]
            if len(names) > 1:
                raise ValueError(
                    "This run has multiple dataset instruction libraries. "
                    "Provide --instruction explicitly for evaluation."
                )
            return os.path.join(artifact_dir, names[0]) if names else None
    # Existing run directories may predate per-dataset snapshot filenames.
    legacy_path = os.path.join(artifact_dir, "instruction_library.json")
    return legacy_path if os.path.isfile(legacy_path) else None
