#!/usr/bin/env python
"""Add, replace or remove HDF5 attributes.

Accepts one HDF5 file, a dataset containing episodes/, or a directory of files.
Repeat --attr to update multiple attributes, including instruction metadata.
Use --remove-attr to clean up unwanted attributes, including malformed names.
"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py

from wrobo.utils.hdf5_metadata import _parse_assignment, _prepare_attributes, _validate_attribute_name


def _prepare_edits(attributes, remove_attributes):
    if not isinstance(attributes, dict):
        raise ValueError("Attributes to update must be a mapping.")
    prepared = _prepare_attributes(attributes) if attributes else {}
    if isinstance(remove_attributes, (str, bytes)):
        raise ValueError("remove_attributes must be a sequence of attribute names.")
    removals = tuple(dict.fromkeys(remove_attributes))
    for name in removals:
        # Explicit removal must also accept the malformed names we reject above.
        _validate_attribute_name(name, allow_shell_quotes=True)
    if not prepared and not removals:
        raise ValueError("Provide at least one --attr or --remove-attr.")
    overlap = set(prepared).intersection(removals)
    if overlap:
        raise ValueError(f"Cannot update and remove the same attribute: {', '.join(sorted(overlap))}.")
    return prepared, removals


def _target_object(file, object_path):
    if not isinstance(object_path, str) or not object_path.strip() or "\x00" in object_path:
        raise ValueError("--object must be a nonempty HDF5 object path.")
    try:
        return file[object_path]
    except KeyError as exc:
        raise ValueError(f"{file.filename}: HDF5 object {object_path!r} does not exist.") from exc


def _display_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _update_attributes(hdf5_path, attributes, dry_run, object_path, remove_attributes=()):
    with h5py.File(hdf5_path, "r" if dry_run else "r+") as file:
        obj = _target_object(file, object_path)
        for name, value in attributes.items():
            old_value = _display_value(obj.attrs[name]) if name in obj.attrs else "(none)"
            print(f"  {obj.name} attr {name!r}: {old_value} -> {_display_value(value)}")
            if not dry_run:
                # Assignment replaces the previous attribute's type and shape too.
                obj.attrs[name] = value
        for name in remove_attributes:
            exists = name in obj.attrs
            old_value = _display_value(obj.attrs[name]) if exists else "(none)"
            print(f"  {obj.name} attr {name!r}: {old_value} -> (removed)")
            if not dry_run and exists:
                del obj.attrs[name]
        print(f"  {'[DRY RUN] Would update' if dry_run else 'Updated'} {hdf5_path}")


def update_hdf5_attrs(
    hdf5_path: Path, attributes: dict, dry_run: bool = False,
    object_path: str = "/", remove_attributes=(),
):
    """Add, replace or remove attributes on an existing HDF5 object in place.

    Numbers, booleans and uniform arrays use native HDF5 types; dictionaries use
    JSON strings. instruction retains wrobo's episode/frames validation.
    """
    prepared, removals = _prepare_edits(attributes, remove_attributes)
    _update_attributes(Path(hdf5_path).expanduser(), prepared, dry_run, object_path, removals)


def _episode_indices(value):
    try:
        indices = {int(item.strip()) for item in value.split(",")}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--episodes requires comma-separated nonnegative integers.") from exc
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("--episodes requires comma-separated nonnegative integers.")
    return indices


def _select_files(target, episode_indices):
    target = Path(target).expanduser()
    if target.is_file():
        if target.suffix.lower() not in (".hdf5", ".h5"):
            raise ValueError(f"Expected an .hdf5 or .h5 file: {target}")
        if episode_indices is not None:
            raise ValueError("--episodes selects files in a directory; omit it for a single file.")
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"Target does not exist: {target}")
    directory = target / "episodes" if (target / "episodes").is_dir() else target
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in (".hdf5", ".h5")
    )
    if not files:
        raise ValueError(f"No HDF5 files found in {directory}")
    if episode_indices is not None:
        selected, found = [], set()
        for path in files:
            try:
                index = int(path.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if index in episode_indices:
                selected.append(path)
                found.add(index)
        missing = episode_indices - found
        if missing:
            raise ValueError(f"Episodes not found in {directory}: {', '.join(map(str, sorted(missing)))}")
        files = selected
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Edit HDF5 attributes using --attr NAME=VALUE and --remove-attr NAME.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from the repository root; single-quote JSON in Bash):
  # Add or change root attributes in a single file
  python -m wrobo.tools.add_hdf5_metadata episode_0.hdf5 --attr sim=true --attr fps=30

  # Batch edit a dataset or a directory of HDF5 files
  python -m wrobo.tools.add_hdf5_metadata dataset/ --attr robot_type=so101 --episodes 0,1 --dry-run

  # Numbers, lists and JSON objects; objects are stored as JSON strings
  python -m wrobo.tools.add_hdf5_metadata dataset/ --attr 'joint_names=["shoulder","gripper"]' --attr 'settings={"version":1}'

  # Edit an existing group's or dataset's attributes
  python -m wrobo.tools.add_hdf5_metadata episode_0.hdf5 --object /observations --attr source=real

  # Instruction and other attributes can be updated together
  python -m wrobo.tools.add_hdf5_metadata dataset/ --attr 'instruction={"episode":1}' --attr task_name=pick_cube
  --attr 'instruction={"episode":{"task_id":0,"params":{"object":"blue cube"}}}'

  # Remove an unwanted attribute; missing names are skipped
  python -m wrobo.tools.add_hdf5_metadata dataset/ --remove-attr obsolete --dry-run
Notes:
  --attr values are parsed as JSON when possible, otherwise as plain strings.
  To force a numeric-looking string, use --attr 'code="123"'. null is unsupported.
  Existing attribute types and shapes are replaced; other attributes/data stay intact.
  instruction uses the same --attr entry and accepts validated episode/frames JSON.
  Its task IDs and params must match your instruction_library.json.
"""
    )
    parser.add_argument(
        "target",
        type=str,
        help="HDF5 file, dataset containing episodes/, or directory of .hdf5/.h5 files",
    )
    parser.add_argument(
        "--attr", action="append", default=[], metavar="NAME=VALUE",
        help="Attribute to add or replace; repeat for multiple attributes. Values accept JSON.",
    )
    parser.add_argument(
        "--remove-attr", action="append", default=[], metavar="NAME",
        help="Attribute to remove; repeat for multiple names. Missing attributes are skipped.",
    )
    parser.add_argument("--object", default="/", help="Existing HDF5 group/dataset path (default: root /).")
    parser.add_argument(
        "--episodes",
        type=_episode_indices,
        default=None,
        help="Comma-separated episode indices (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files"
    )

    args = parser.parse_args(argv)

    try:
        attributes = dict(_parse_assignment(assignment) for assignment in args.attr)
        prepared, removals = _prepare_edits(attributes, args.remove_attr)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    try:
        hdf5_files = _select_files(args.target, args.episodes)
        # Check every selected file/object before any file is modified.
        for path in hdf5_files:
            with h5py.File(path, "r") as file:
                _target_object(file, args.object)
        print(f"Processing {len(hdf5_files)} HDF5 file(s)")
        for hdf5_file in hdf5_files:
            print(f"\n{hdf5_file.name}:")
            _update_attributes(hdf5_file, prepared, args.dry_run, args.object, removals)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"\n{'Preview' if args.dry_run else 'Update'} complete!")


if __name__ == "__main__":
    main()
