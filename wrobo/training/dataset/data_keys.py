"""Resolve configured HDF5 fields into single state/action input roles."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataKeyBindings:
    proprio_path: str
    proprio_key: str
    action_key: str


def resolve_data_keys(data_keys):
    """Select roles from data_keys, never by inspecting unselected file fields.

    The existing statistics collector expects direct proprio_* observations
    and root action_* datasets. Multiple selected candidates are ambiguous;
    they must not be silently ordered, renamed or concatenated.
    """
    if (not isinstance(data_keys, list) or not data_keys
            or any(not isinstance(key, str) or not key for key in data_keys)):
        raise ValueError("Training_settings.data_keys must be a nonempty list of strings")
    duplicates = sorted({key for key in data_keys if data_keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Training_settings.data_keys has duplicate fields: {duplicates}")
    proprio = [key for key in data_keys
               if key.startswith("observations/proprio_")
               and "/" not in key[len("observations/"):]]
    actions = [key for key in data_keys if key.startswith("action_") and "/" not in key]
    for role, candidates, convention in (
        ("proprio", proprio, "observations/proprio_*"),
        ("action", actions, "root action_*"),
    ):
        if len(candidates) != 1:
            raise ValueError(
                f"Training_settings.data_keys must select exactly one {role} field "
                f"({convention}); found {candidates}. Select one explicitly in data_keys."
            )
    return DataKeyBindings(proprio[0], proprio[0][len("observations/"):], actions[0])
