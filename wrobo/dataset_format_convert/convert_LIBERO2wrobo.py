"""Replay LIBERO demonstrations and convert regenerated observations to wrobo.

Images are rendered at 256x256 by default. Failure and no-op filtering are both
opt-in. Both rendered camera views are rotated 180 degrees for the OpenVLA
image convention. All original demo data is archived alongside pre-action
observations and source action indices. Splitting remains a training concern.
Run in the LIBERO environment; --dry-run requires only HDF5/image dependencies.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import h5py
import numpy as np
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wrobo.utils.image_codec import create_image_dataset
from wrobo.dataset_format_convert.libero_replay import replay_demo

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10")
CAMERAS = {"agentview_rgb": "image_top", "eye_in_hand_rgb": "image_wrist"}
VECTOR_WIDTHS = {
    "actions": 7, "robot_states": 9, "obs/joint_states": 7,
    "obs/gripper_states": 2, "obs/ee_states": 6,
    "obs/ee_pos": 3, "obs/ee_ori": 3,
}
KNOWN_DATASETS = set(VECTOR_WIDTHS) | {
    "states", "rewards", "dones", *[f"obs/{name}" for name in CAMERAS],
}
ACTION_SEMANTICS = "native_LIBERO_OSC_POSE_delta_commands_with_gripper"
INDEX_POLICY = "pre_action_replay_after_10_settling_steps_with_explicit_source_indices"
IMAGE_ORIENTATION = "rotate_180_from_renderer"
METADATA_UPDATE_INTERVAL_SECONDS = 30.0
FIELD_MAPPING = {
    "obs/agentview_rgb": {"target": "observations/image_top", "transform": "rotate_180_then_HWC_to_CHW"},
    "obs/eye_in_hand_rgb": {"target": "observations/image_wrist", "transform": "rotate_180_then_HWC_to_CHW"},
    "obs/joint_states": {"target": "observations/proprio_joint", "columns": [0, 7]},
    "obs/gripper_states": {"target": "observations/proprio_joint", "columns": [7, 9]},
    "obs/ee_states": {"target": "observations/proprio_ee", "columns": [0, 6]},
    "obs/ee_pos": {"target": "observations/proprio_ee", "columns": [0, 3]},
    "obs/ee_ori": {"target": "observations/proprio_ee", "columns": [3, 6]},
    "robot_states": {"target": "observations/proprio_states"},
    "actions": {"target": "action_rel"}, "states": {"target": "libero/states"},
    "rewards": {"target": "rewards"}, "dones": {"target": "dones"},
}
PROPRIO_COMPONENTS = {
    "proprio_joint": ["joint_positions[7]", "gripper_positions[2]"],
    "proprio_ee": ["ee_position[3]", "ee_rotation_vector[3]", "gripper_positions[2]"],
    "proprio_states": ["gripper_positions[2]", "ee_position[3]", "ee_quaternion_xyzw[4]"],
}


def _json_value(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _attributes(group):
    """Keep source JSON attributes as their original strings."""
    return {name: _json_value(value) for name, value in group.attrs.items()}


def _attribute_dtypes(group):
    result = {}
    for name, value in group.attrs.items():
        dtype = group.attrs.get_id(name).dtype
        entry = {"dtype": str(dtype), "shape": list(np.shape(value))}
        string = h5py.check_string_dtype(dtype)
        if string is not None:
            entry.update(string_encoding=string.encoding, string_length=string.length)
        result[name] = entry
    return result


def _parsed_json(attrs, name):
    value = attrs.get(name, {})
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return parsed


def _language(attrs, source):
    value = _parsed_json(attrs, "problem_info").get("language_instruction")
    if isinstance(value, list) and all(isinstance(word, str) for word in value):
        value = " ".join(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: missing problem_info.language_instruction")
    return value


def _discover_files(source, suites):
    if suites is not None:
        if not suites or any(suite not in SUITES for suite in suites):
            raise ValueError(f"suites must be selected from {SUITES}")
        if len(suites) != len(set(suites)):
            raise ValueError("Duplicate suites are not allowed")
    if source.is_file():
        paths = [source] if source.suffix == ".hdf5" else []
    elif source.is_dir():
        paths = list(source.glob("*_demo.hdf5"))
        if not paths:
            for suite in suites or SUITES:
                folder = source / suite
                found = list(folder.glob("*_demo.hdf5"))
                if suites and not found:
                    raise FileNotFoundError(f"No LIBERO task files in {folder}")
                paths.extend(found)
    else:
        raise FileNotFoundError(f"Source does not exist: {source}")
    if suites:
        paths = [path for path in paths if path.parent.name in suites]
    if not paths:
        raise FileNotFoundError(f"No selected LIBERO *_demo.hdf5 files in {source}")
    return sorted(paths, key=lambda path: (
        SUITES.index(path.parent.name) if path.parent.name in SUITES else len(SUITES),
        path.parent.name, path.name,
    ))


def _dataset(demo, name):
    if name not in demo or not isinstance(demo[name], h5py.Dataset):
        raise ValueError(f"{demo.file.filename}:{demo.name}: missing dataset {name}")
    return demo[name]


def _validate_demo(demo):
    """Validate mapped headers; unknown objects are archived without reshaping."""
    label = f"{demo.file.filename}:{demo.name}"
    actions = _dataset(demo, "actions")
    if actions.ndim != 2 or actions.shape[1] != 7 or actions.shape[0] < 1:
        raise ValueError(f"{label}: actions must have shape (T, 7), T > 0")
    length = actions.shape[0]
    samples = demo.attrs.get("num_samples")
    if (not isinstance(samples, (int, np.integer)) or isinstance(samples, (bool, np.bool_))
            or int(samples) != length):
        raise ValueError(f"{label}: num_samples must be an integer matching actions length")
    for name, width in VECTOR_WIDTHS.items():
        field = _dataset(demo, name)
        if field.shape != (length, width) or not np.issubdtype(field.dtype, np.floating):
            raise ValueError(f"{label}: {name} must be floating-point ({length}, {width})")
    states = _dataset(demo, "states")
    if (states.ndim != 2 or states.shape[0] != length or states.shape[1] < 1
            or not np.issubdtype(states.dtype, np.floating)):
        raise ValueError(f"{label}: states must be floating-point (T, S)")
    for name in ("rewards", "dones"):
        if _dataset(demo, name).shape != (length,):
            raise ValueError(f"{label}: {name} must have shape ({length},)")
    for name in CAMERAS:
        camera = _dataset(demo, f"obs/{name}")
        if (camera.ndim != 4 or camera.shape[0] != length or camera.shape[-1] != 3
                or min(camera.shape[1:3]) < 1 or camera.dtype != np.dtype("uint8")):
            raise ValueError(f"{label}: obs/{name} must be uint8 (T, H, W, 3)")
    if "init_state" not in demo.attrs or np.shape(demo.attrs["init_state"]) != (states.shape[1],):
        raise ValueError(f"{label}: init_state must have shape ({states.shape[1]},)")
    model = demo.attrs.get("model_file")
    if not isinstance(model, (str, bytes, np.str_, np.bytes_)) or not len(model):
        raise ValueError(f"{label}: missing model_file XML attribute")
    return length


def _revision(path):
    for metadata in (
        path.parent / ".cache/huggingface/download" / (path.name + ".metadata"),
        path.parent.parent / ".cache/huggingface/download" / path.parent.name / (path.name + ".metadata"),
    ):
        if metadata.is_file():
            lines = metadata.read_text(encoding="utf-8").splitlines()
            if lines and re.fullmatch(r"[0-9a-f]{40}", lines[0]):
                return lines[0]
    return None


def _validate_destinations(destination, suites):
    """Protect selected outputs without rejecting unrelated existing data."""
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"Destination parent must be a directory: {destination}")
    for suite in sorted(suites):
        folder = destination / suite
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            raise FileExistsError(
                f"Target suite directory must be absent or empty: {folder}. "
                "Existing output is preserved; use a different --dst to convert this suite again."
            )


def _plan(paths):
    tasks, episodes, task_counts, episode_counts = [], [], {}, {}
    for path in tqdm(paths, desc="Inspecting tasks",
                     unit="task", dynamic_ncols=True):
        suite = path.parent.name
        with h5py.File(path, "r") as src:
            if "data" not in src or not isinstance(src["data"], h5py.Group):
                raise ValueError(f"{path}: missing data group")
            if set(src) != {"data"}:
                raise ValueError(f"{path}: unsupported objects outside data; refusing to omit them")
            data = src["data"]
            names = sorted((name for name in data if re.fullmatch(r"demo_\d+", name)),
                           key=lambda name: int(name[5:]))
            if not names or set(names) != set(data):
                raise ValueError(f"{path}: data must contain demo_N groups only; refusing to omit other objects")
            attrs = _attributes(data)
            task_id = task_counts.get(suite, 0)
            task = {"task_id": task_id, "suite": suite,
                    "name": re.sub(r"_demo$", "", path.stem), "source_file": str(path),
                    "source_revision": _revision(path), "source_data_attributes": attrs,
                    "source_data_attribute_dtypes": _attribute_dtypes(data),
                    "source_root_attributes": _attributes(src),
                    "source_root_attribute_dtypes": _attribute_dtypes(src),
                    "parsed_environment": _parsed_json(attrs, "env_args"),
                    "parsed_problem_info": _parsed_json(attrs, "problem_info"),
                    "language_instruction": _language(attrs, path)}
            for name in names:
                demo = data[name]
                length = _validate_demo(demo)
                episode_id = f"episode_{episode_counts.get(suite, 0)}"
                episode = {"episode_id": episode_id, "suite": suite, "task_id": task_id,
                           "source_demo": name, "seq_len": length, "source_seq_len": length,
                           "source_attribute_dtypes": _attribute_dtypes(demo),
                           "source_dataset_dtypes": {key: str(demo[key].dtype) for key in sorted(KNOWN_DATASETS)}}
                episodes.append(episode)
                episode_counts[suite] = episode_counts.get(suite, 0) + 1
            tasks.append(task)
            task_counts[suite] = task_id + 1
    return tasks, episodes


def _read_numeric(dataset):
    values = dataset[:]
    if not (np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_)):
        raise ValueError(f"{dataset.name}: expected numeric data")
    if not np.isfinite(values).all():
        raise ValueError(f"{dataset.name}: non-finite values")
    return values


def _numeric(group, name, values):
    """Lossless compression without dtype conversion."""
    return group.create_dataset(name, data=values, compression="gzip", compression_opts=1,
                                shuffle=True, chunks=(min(50, len(values)), *values.shape[1:]))


def _copy_attributes(source, target, skip=()):
    for name, value in source.attrs.items():
        if name not in skip:
            target.attrs.create(name, value, dtype=source.attrs.get_id(name).dtype)


def _archive_source(source, archive):
    """Preserve the complete original demo, including arrays replaced by replay."""
    _copy_attributes(source, archive)
    for name in source:
        source.copy(name, archive, name=name)


def _validate_replay(replayed, demo, image_size):
    arrays = replayed["arrays"]
    indices = np.asarray(replayed["source_indices"])
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= demo["actions"].shape[0])
            or np.any(np.diff(indices) <= 0)):
        raise ValueError(f"{demo.name}: invalid replay source_indices")
    length = len(indices)
    for name, width in VECTOR_WIDTHS.items():
        values = np.asarray(arrays[name])
        if values.shape != (length, width) or not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"{demo.name}: replay {name} must be floating-point ({length}, {width})")
        if not np.isfinite(values).all():
            raise ValueError(f"{demo.name}: non-finite replay {name}")
    actions = demo["actions"][:]
    if not np.isfinite(actions).all():
        raise ValueError(f"{demo.name}: non-finite source actions")
    if arrays["actions"].dtype != actions.dtype or not np.array_equal(arrays["actions"], actions[indices]):
        raise ValueError(f"{demo.name}: replay must preserve original retained action values and dtype")
    states = np.asarray(arrays["states"])
    if states.ndim != 2 or states.shape[0] != length or states.shape[1] < 1 or not np.isfinite(states).all():
        raise ValueError(f"{demo.name}: invalid replay simulator states")
    for name in ("rewards", "dones"):
        values = np.asarray(arrays[name])
        if values.shape != (length,) or not np.isfinite(values).all():
            raise ValueError(f"{demo.name}: invalid replay {name}")
    for name in CAMERAS:
        images = np.asarray(arrays[f"obs/{name}"])
        if images.shape != (length, image_size, image_size, 3) or images.dtype != np.uint8:
            raise ValueError(f"{demo.name}: replay images must be uint8 (T, {image_size}, {image_size}, 3)")
    if length:
        ee = np.concatenate((arrays["obs/ee_pos"], arrays["obs/ee_ori"]), axis=1)
        if not np.array_equal(ee, arrays["obs/ee_states"]):
            raise ValueError(f"{demo.name}: replay ee_states is inconsistent")
        if not np.array_equal(replayed["init_state"], states[0]):
            raise ValueError(f"{demo.name}: replay init_state must match the first pre-action state")
    if not isinstance(replayed["model_file"], str) or not replayed["model_file"]:
        raise ValueError(f"{demo.name}: missing replay model XML")
    return length


def _write_episode(src, task, episode, destination, encoding, jpeg_quality, replayed):
    demo = src["data"][episode["source_demo"]]
    values = replayed["arrays"]
    length = len(replayed["source_indices"])
    path = Path(destination) / "episodes" / (episode["episode_id"] + ".hdf5")
    fd, temporary = tempfile.mkstemp(prefix=".libero_", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w", libver="latest", track_order=True) as dst:
            dst.attrs.create("seq_len", length, dtype=demo.attrs.get_id("num_samples").dtype)
            dst.attrs["model_file"] = replayed["model_file"]
            dst.attrs["init_state"] = replayed["init_state"]
            dst.attrs["replay_success"] = bool(replayed["success"])
            dst.attrs["instruction"] = json.dumps({"episode": {"task_id": task["task_id"]}})
            observations = dst.create_group("observations")
            fields = (
                ("proprio_joint", np.concatenate((values["obs/joint_states"], values["obs/gripper_states"]), axis=1)),
                ("proprio_ee", np.concatenate((values["obs/ee_states"], values["obs/gripper_states"]), axis=1)),
                ("proprio_states", values["robot_states"]),
            )
            for name, data in fields:
                _numeric(observations, name, data)
            for source_name, target_name in CAMERAS.items():
                # OpenVLA applies this rotation when building RLDS; wrobo stores
                # the same convention directly, before encoding either camera.
                images = np.ascontiguousarray(
                    values[f"obs/{source_name}"][:, ::-1, ::-1, :].transpose(0, 3, 1, 2)
                )
                image_dataset = create_image_dataset(
                    observations, target_name, images,
                    encoding=encoding, jpeg_quality=jpeg_quality,
                )
                image_dataset.attrs["image_orientation"] = IMAGE_ORIENTATION
                if encoding == "jpeg":
                    image_dataset.attrs["jpeg_quality"] = jpeg_quality
            _numeric(dst, "action_rel", values["actions"])
            for name in ("rewards", "dones"):
                _numeric(dst, name, values[name])
            auxiliary = dst.create_group("libero", track_order=True)
            _numeric(auxiliary, "states", values["states"])
            _numeric(auxiliary, "source_indices", np.asarray(replayed["source_indices"], dtype=np.int64))
            archive = auxiliary.create_group("source", track_order=True).create_group("demo", track_order=True)
            _archive_source(demo, archive)
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _convert_episode(arguments):
    task, episode, destination, options = arguments
    with h5py.File(task["source_file"], "r") as src:
        demo = src["data"][episode["source_demo"]]
        if _validate_demo(demo) != episode["source_seq_len"]:
            raise ValueError(f"{demo.name}: source length changed since preflight")
        _read_numeric(demo["actions"])
        if not np.isfinite(demo["states"][0]).all():
            raise ValueError(f"{demo.name}: non-finite source initial state")
        replayed = replay_demo(
            task, demo, image_size=options["image_size"], filter_noops=options["filter_noops"],
            noop_threshold=options["noop_threshold"],
        )
        length = _validate_replay(replayed, demo, options["image_size"])
        reason = ("empty_after_noop_filter" if length == 0 else
                  "failed_replay" if options["filter_failed"] and not replayed["success"] else None)
        result = dict(episode, seq_len=length, success=bool(replayed["success"]),
                      num_noops=int(replayed["num_noops"]),
                      source_indices=np.asarray(replayed["source_indices"]).tolist(),
                      replay_dataset_dtypes={name: str(values.dtype) for name, values in replayed["arrays"].items()},
                      kept=reason is None, exclusion_reason=reason)
        if reason is None:
            _write_episode(src, task, episode, destination, options["img_encoding"],
                           options["jpeg_quality"], replayed)
        return result


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _output_task(task, episodes, options):
    """Describe regenerated output, keeping original settings in source_tasks."""
    kwargs = task["parsed_environment"].get("env_kwargs", {})
    size = options["image_size"]
    return {
        **{key: task[key] for key in ("task_id", "suite", "name", "language_instruction")},
        "num_episodes": len(episodes),
        "num_steps": sum(episode["seq_len"] for episode in episodes),
        "image_shapes": {name: [3, size, size] for name in CAMERAS.values()},
        "environment": {
            "class": "libero.libero.envs.OffScreenRenderEnv",
            "bddl_file": task["suite"] + "/" + task["name"] + ".bddl",
            "kwargs": {
                "controller": "OSC_POSE", "robots": kwargs.get("robots", ["Panda"]),
                "camera_names": ["agentview", "robot0_eye_in_hand"],
                "camera_heights": size, "camera_widths": size,
                "camera_depths": False, "camera_segmentations": None,
                "control_freq": kwargs.get("control_freq", 20),
            },
        },
    }


def _suite_metadata(source, destination, tasks, episodes, options):
    descriptions = {}
    source_episodes = {}
    for episode in episodes:
        source_episodes.setdefault((episode["suite"], episode["task_id"]), []).append(episode)
    for task in tasks:
        suite = task["suite"]
        if suite not in descriptions:
            descriptions[suite] = {
                "format": "wrobo_LIBERO", "schema_version": 5, "status": "planned",
                "source": str(source), "destination": str(destination / suite), "suite": suite,
                "options": options, "tasks": {}, "source_tasks": {}, "episodes": {},
                "excluded_episodes": {}, "field_mapping": FIELD_MAPPING,
                "image_shapes": {name: [3, options["image_size"], options["image_size"]]
                                 for name in CAMERAS.values()},
                "output_counts_final": False,
                "metadata_update_policy": "after_source_task_or_30_seconds",
                "field_mapping_applies_to": "regenerated replay arrays; complete originals are archived",
                "attribute_mapping": {name: "libero/source/demo@" + name
                                      for name in ("num_samples", "model_file", "init_state")},
                "generated_attributes": {
                    "seq_len": "number of retained replay actions",
                    "model_file": "current replay simulator model XML",
                    "init_state": "first retained pre-action simulator state",
                },
                "proprio_components": PROPRIO_COMPONENTS,
                "action_semantics": ACTION_SEMANTICS, "source_index_policy": INDEX_POLICY,
                "image_orientation": IMAGE_ORIENTATION,
                "image_generation": {"rerendered": True, "resolution": options["image_size"],
                                     "rotation_degrees": 180,
                                     "settling_steps": 10, "initial_state_source": "source states[0]",
                                     "success_criterion": "final_environment_check_success"},
                "image_preservation": {
                    "encoding": options["img_encoding"],
                    "lossless": options["img_encoding"] != "jpeg",
                    "jpeg_quality": options["jpeg_quality"] if options["img_encoding"] == "jpeg" else None,
                },
                "slice_convention": "zero_based_half_open_column_ranges",
                "source_archive": {"path": "libero/source/demo",
                    "description": "Complete original demo, including all arrays and attributes, copied without resampling. libero/source_indices maps each regenerated row to its original action index."},
            }
        planned = source_episodes[(suite, task["task_id"])]
        descriptions[suite]["source_tasks"][str(task["task_id"])] = dict(
            task, num_episodes=len(planned), num_steps=sum(episode["source_seq_len"] for episode in planned),
        )
    for suite, description in descriptions.items():
        description["num_source_tasks"] = len(description["source_tasks"])
        description["num_source_episodes"] = sum(task["num_episodes"] for task in description["source_tasks"].values())
        description["num_source_steps"] = sum(task["num_steps"] for task in description["source_tasks"].values())
    _finalize_descriptions(descriptions, {})
    return descriptions


def _finalize_descriptions(descriptions, results):
    """Refresh a snapshot from completed results without consuming the source plan."""
    for suite, description in descriptions.items():
        records = sorted((record for (name, _), record in results.items() if name == suite),
                         key=lambda record: int(record["episode_id"][len("episode_"):]))
        kept = [record for record in records if record["kept"]]
        description["episodes"] = {record["episode_id"]: record for record in kept}
        description["excluded_episodes"] = {record["episode_id"]: record for record in records if not record["kept"]}
        task_episodes = {}
        for record in kept:
            task_episodes.setdefault(str(record["task_id"]), []).append(record)
        description["tasks"] = {key: _output_task(task, task_episodes[key], description["options"])
                                for key, task in description["source_tasks"].items() if key in task_episodes}
        description["num_tasks"] = len(description["tasks"])
        description["num_episodes"] = len(kept)
        description["num_steps"] = sum(record["seq_len"] for record in kept)
        description["num_replayed_episodes"] = len(records)
        description["num_pending_episodes"] = description["num_source_episodes"] - len(records)
        description["num_successful_replays"] = sum(record["success"] for record in records)
        description["num_failed_replays"] = sum(not record["success"] for record in records)
        description["num_noops_removed"] = sum(record["num_noops"] for record in records)
        description["num_filtered_failed"] = sum(record["exclusion_reason"] == "failed_replay" for record in records)
        description["num_filtered_empty"] = sum(record["exclusion_reason"] == "empty_after_noop_filter" for record in records)


def _summary(descriptions, source, destination, status):
    result = {"status": status, "source": str(source), "destination": str(destination), "suites": {}}
    fields = ("num_tasks", "num_episodes", "num_steps", "num_source_tasks", "num_source_episodes",
              "num_source_steps", "num_replayed_episodes", "num_pending_episodes", "num_successful_replays", "num_failed_replays",
              "num_noops_removed", "num_filtered_failed", "num_filtered_empty")
    for suite, description in descriptions.items():
        result["suites"][suite] = dict(destination=description["destination"],
                                      **{key: description[key] for key in fields if key in description})
    for key in fields:
        if all(key in value for value in result["suites"].values()):
            result[key] = sum(value[key] for value in result["suites"].values())
    result["output_counts_final"] = status == "complete"
    return result


def convert_libero_dataset(
    src_path, dst_path, *, suites=None, img_encoding="png", jpeg_quality=95,
    workers=1, dry_run=False, filter_failed=False, filter_noops=False,
    image_size=256,
):
    """Regenerate source demonstrations, then write one wrobo HDF5 per retained demo.

    Filtering is disabled by default. Replay always runs except in dry-run mode.
    Each worker creates and closes its own environment; original source files are
    read-only and archived in full. Selected output suites must be new or empty.
    """
    source, destination = Path(src_path).resolve(), Path(dst_path).resolve()
    if img_encoding not in ("png", "jpeg", "gzip", "raw"):
        raise ValueError("img_encoding must be png, jpeg, gzip or raw")
    if (not isinstance(jpeg_quality, int) or isinstance(jpeg_quality, bool)
            or not 0 <= jpeg_quality <= 100):
        raise ValueError("jpeg_quality must be an integer in [0, 100]")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 1:
        raise ValueError("image_size must be a positive integer")
    if not isinstance(filter_failed, bool) or not isinstance(filter_noops, bool):
        raise ValueError("filter_failed and filter_noops must be booleans")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination paths must not overlap")
    paths = _discover_files(source, suites)
    selected_suites = {path.parent.name for path in paths}
    _validate_destinations(destination, selected_suites)
    tasks, episodes = _plan(paths)
    options = {"img_encoding": img_encoding, "jpeg_quality": jpeg_quality, "workers": workers,
               "image_size": image_size,
               "filter_failed": filter_failed, "filter_noops": filter_noops, "noop_threshold": 1e-4}
    descriptions = _suite_metadata(source, destination, tasks, episodes, options)
    print(f"{'Planned' if dry_run else 'Replaying'} {len(episodes)} source episodes from {len(tasks)} tasks; "
          f"render={image_size}x{image_size}, filter_failed={filter_failed}, filter_noops={filter_noops}", flush=True)
    if dry_run:
        return _summary(descriptions, source, destination, "planned")
    _validate_destinations(destination, selected_suites)
    destination.mkdir(parents=True, exist_ok=True)
    for suite, description in descriptions.items():
        folder = destination / suite
        folder.mkdir(exist_ok=True)
        (folder / "episodes").mkdir()
        description["status"] = "in_progress"
        _write_json(folder / "data_description.json", description)
    lookup = {(task["suite"], task["task_id"]): task for task in tasks}
    arguments = [(lookup[(episode["suite"], episode["task_id"])], episode,
                  str(destination / episode["suite"]), options) for episode in episodes]
    results = {}
    completed_tasks = {}
    metadata_updated_at = {suite: time.monotonic() for suite in descriptions}
    kept_count = dropped_count = noop_count = 0
    try:
        with tqdm(total=len(episodes), desc="Replaying episodes", unit="episode", dynamic_ncols=True) as progress:
            def record(result):
                nonlocal kept_count, dropped_count, noop_count
                results[(result["suite"], result["episode_id"])] = result
                kept_count += int(result["kept"])
                dropped_count += int(not result["kept"])
                noop_count += result["num_noops"]
                progress.update(1)
                progress.set_postfix_str(
                    f"kept={kept_count}, dropped={dropped_count}, noops={noop_count}", refresh=False,
                )
                suite, task_id = result["suite"], str(result["task_id"])
                task_key = (suite, task_id)
                completed_tasks[task_key] = completed_tasks.get(task_key, 0) + 1
                now = time.monotonic()
                if (completed_tasks[task_key] == descriptions[suite]["source_tasks"][task_id]["num_episodes"]
                        or now - metadata_updated_at[suite] >= METADATA_UPDATE_INTERVAL_SECONDS):
                    _finalize_descriptions({suite: descriptions[suite]}, results)
                    _write_json(destination / suite / "data_description.json", descriptions[suite])
                    metadata_updated_at[suite] = now
            if workers == 1 or len(arguments) == 1:
                for argument in arguments:
                    record(_convert_episode(argument))
            else:
                with multiprocessing.get_context("spawn").Pool(min(workers, len(arguments))) as pool:
                    for result in pool.imap_unordered(_convert_episode, arguments):
                        record(result)
    except BaseException as error:
        _finalize_descriptions(descriptions, results)
        for suite, description in descriptions.items():
            description.update(status="failed", error=str(error))
            _write_json(destination / suite / "data_description.json", description)
        raise
    _finalize_descriptions(descriptions, results)
    for suite, description in descriptions.items():
        library = {}
        for task_id, task in description["tasks"].items():
            library[task_id] = {"name": task["name"], "type": "library",
                               "instructions": [task["language_instruction"]],
                               "episodes": [episode_id for episode_id, episode in description["episodes"].items()
                                            if episode["task_id"] == task["task_id"]]}
        _write_json(destination / suite / "instruction_library.json", library)
        description.update(status="complete", output_counts_final=True)
        _write_json(destination / suite / "data_description.json", description)
    return _summary(descriptions, source, destination, "complete")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="LIBERO root, suite directory or task HDF5")
    parser.add_argument("--dst", required=True,
                        help="Output parent; existing other suites are kept. Selected <dst>/<suite>/ must be new or empty")
    parser.add_argument("--suites", nargs="+", choices=SUITES)
    parser.add_argument("--img-encoding", choices=("png", "jpeg", "gzip", "raw"), default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality in [0, 100] (default: 95; only used with jpeg)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Replay process count; 1 runs serially (default: 1)")
    parser.add_argument("--image-size", type=int, default=256, help="Square render resolution (default: 256)")
    parser.add_argument("--filter-failed", action="store_true",
                        help="Discard demonstrations that fail when replayed (default: keep)")
    parser.add_argument("--filter-noops", action="store_true",
                        help="Skip OpenVLA-style no-op actions during replay (default: keep)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect source headers without simulation or output; final filtered counts are unknown")
    args = parser.parse_args(argv)
    result = convert_libero_dataset(
        args.src, args.dst, suites=args.suites, img_encoding=args.img_encoding,
        jpeg_quality=args.jpeg_quality, workers=args.workers, dry_run=args.dry_run,
        filter_failed=args.filter_failed, filter_noops=args.filter_noops,
        image_size=args.image_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

"""
python -m wrobo.dataset_format_convert.convert_LIBERO2wrobo \
  --src /media/data/wlty/RoBoDataset/LIBERO \
  --dst /media/data/wlty/RoBoDataset/wrobo/LIBERO \
  --filter-failed --filter-noops \
  --workers 12
"""

if __name__ == "__main__":
    raise SystemExit(main())
