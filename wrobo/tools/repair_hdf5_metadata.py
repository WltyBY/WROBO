#!/usr/bin/env python
"""Repair existing HDF5 metadata damaged by Bash quoting in Windows CMD.

Examples, from the repository root (CMD, Bash and Zsh):
    python -m wrobo.tools.repair_hdf5_metadata Dataset/SO101_pre --dry-run
    python -m wrobo.tools.repair_hdf5_metadata Dataset/SO101_pre

Only root attributes are edited. No backups are created. All selected files are
checked before any write. Healthy files are skipped, so rerunning is harmless.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np

from wrobo.utils.hdf5_metadata import _normalize_instruction


# Recover only the simple, unambiguous format observed in the damaged files.
# Other JSON structures are accepted only when their quotes remain intact.
_UNQUOTED_INSTRUCTION = re.compile(
    r"\{\s*episode\s*:\s*\{\s*task_id\s*:\s*(?P<task_id>\d+)\s*,"
    r"\s*params\s*:\s*\{\s*object\s*:\s*(?P<object>[^{}\[\],:\"\\\r\n]+?)"
    r"\s*\}\s*\}\s*\}\s*"
)


def _text(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"Expected a text attribute, got {type(value).__name__}.")
    text = value.strip()
    if text.startswith("'"):
        text = text[1:]
    if text.endswith("'"):
        text = text[:-1]
    return text.strip()


def _recover_instruction(raw):
    text = _text(raw)
    try:
        instruction = json.loads(text)
    except json.JSONDecodeError:
        match = _UNQUOTED_INSTRUCTION.fullmatch(text)
        if match is None:
            raise ValueError(
                "Cannot safely recover 'instruction: unsupported damaged JSON format. "
                f"Value: {raw!r}"
            ) from None
        object_name = match["object"].strip()
        if not object_name:
            raise ValueError("Cannot recover an empty object name.")
        # Quote loss makes JSON scalars ambiguous (e.g. 123 versus \"123\").
        try:
            json.loads(object_name)
        except json.JSONDecodeError:
            pass
        else:
            raise ValueError(f"Cannot safely recover the object value {object_name!r}.")
        instruction = {
            "episode": {
                "task_id": int(match["task_id"]),
                "params": {"object": object_name},
            }
        }
    return _normalize_instruction(instruction)


def _recover_success(raw):
    if isinstance(raw, (bool, np.bool_)):
        return bool(raw)
    text = _text(raw)
    if text not in ("true", "false"):
        raise ValueError(f"Cannot recover 'success: expected true or false, got {raw!r}.")
    return text == "true"


def _select_files(target):
    target = Path(target).expanduser().resolve()
    if target.is_file():
        if target.suffix.lower() not in (".h5", ".hdf5"):
            raise ValueError(f"Expected an .h5 or .hdf5 file: {target}")
        return [target]
    if not target.is_dir():
        raise ValueError(f"Target does not exist: {target}")

    if (target / "episodes").is_dir():
        directories = [target / "episodes"]
    else:
        # A parent such as SO101_pre may contain several independent datasets.
        # Exclude _incomplete/ and arbitrary HDF5 files outside episodes/.
        directories = [target, *sorted(
            path for path in target.rglob("episodes")
            if path.is_dir() and "_incomplete" not in path.relative_to(target).parts
        )]
    files = sorted({
        path.resolve()
        for directory in directories
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in (".h5", ".hdf5")
    })
    if not files:
        raise ValueError(f"No HDF5 episode files found in {target}")
    return files


@dataclass
class _Repair:
    path: Path
    updates: dict
    removals: tuple


def _plan(file, path):
    updates, removals = {}, []
    if "'instruction" in file.attrs:
        recovered = _recover_instruction(file.attrs["'instruction"])
        if "instruction" in file.attrs:
            current = _normalize_instruction(file.attrs["instruction"])
            if json.loads(current) != json.loads(recovered):
                raise ValueError("Existing instruction conflicts with recovered 'instruction.")
        updates["instruction"] = recovered
        removals.append("'instruction")
    if "'success" in file.attrs:
        # The ordinary success may be the recorder's default True. The explicit
        # damaged attribute carries the user's intended true/false annotation.
        updates["success"] = _recover_success(file.attrs["'success"])
        removals.append("'success")
    return _Repair(path, updates, tuple(removals))


def _dataset_shapes(file):
    shapes = {}

    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            shapes[name] = (obj.shape, obj.dtype.str)

    file.visititems(collect)
    return shapes


def _show_plan(file, repair, dry_run):
    print(f"\n{'[DRY RUN]' if dry_run else '[REPAIR]'} {repair.path}")
    for name, value in repair.updates.items():
        old = file.attrs[name] if name in file.attrs else "(missing)"
        print(f"  {name}: {old} -> {value}")
    for name in repair.removals:
        print(f"  Remove {name!r}: {file.attrs[name]}")


def repair_metadata(target, dry_run=False):
    """Preflight every file, then preview or repair root metadata in place."""
    files = _select_files(target)
    repairs = []
    for path in files:
        try:
            with h5py.File(path, "r") as file:
                repair = _plan(file, path)
                if repair.removals:
                    repairs.append(repair)
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            raise ValueError(f"{path}: {exc}") from exc

    for repair in repairs:
        with h5py.File(repair.path, "r" if dry_run else "r+") as file:
            # Catch changes since preflight before touching this file's attrs.
            if _plan(file, repair.path) != repair:
                raise RuntimeError(f"Metadata changed since preflight: {repair.path}")
            _show_plan(file, repair, dry_run)
            if dry_run:
                continue
            shapes = _dataset_shapes(file)
            for name, value in repair.updates.items():
                file.attrs[name] = value
            for name in repair.removals:
                del file.attrs[name]
            file.flush()
            for name, value in repair.updates.items():
                if file.attrs[name] != value:
                    raise RuntimeError(f"Metadata verification failed: {repair.path}, {name}")
            if any(name in file.attrs for name in repair.removals):
                raise RuntimeError(f"Malformed attribute remains: {repair.path}")
            if _dataset_shapes(file) != shapes:
                raise RuntimeError(f"Dataset structure changed: {repair.path}")

    summary = {"scanned": len(files), "repaired": len(repairs), "skipped": len(files) - len(repairs)}
    print(
        f"\nScanned {summary['scanned']} file(s); "
        f"{'would repair' if dry_run else 'repaired'} {summary['repaired']}; "
        f"skipped {summary['skipped']} healthy file(s)."
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="HDF5 file, dataset/episodes directory, or parent containing datasets")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing any files")
    args = parser.parse_args(argv)
    try:
        repair_metadata(args.target, dry_run=args.dry_run)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
