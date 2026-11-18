"""Load wrobo policies and adapt raw robot observations to their training inputs.

This module performs model I/O only. Robot connections and action execution belong
to the deployment runner and the LeRobot driver. Heavy model dependencies are
imported only when a policy is loaded.
"""

from dataclasses import dataclass
import copy
import json
from pathlib import Path

import numpy as np

from wrobo.training.dataset.data_keys import resolve_data_keys
from wrobo.utils.normalization import normalization_parameters

from .device import parse_device, resolve_local_device


@dataclass(kw_only=True)
class ModelOptions:
    checkpoint: str | Path
    config_file: str | Path
    norm_stats_key: str | Path
    method: str
    data_description: str | Path | None = None
    norm_stats_group: str | None = None
    device: str = "auto"
    instruction: str | None = None
    compile: bool = False
    amp: bool = False

    def __post_init__(self):
        for name in ("checkpoint", "config_file", "norm_stats_key", "data_description"):
            value = getattr(self, name)
            if value is None and name == "data_description":
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{name} must be a nonempty path")
            setattr(self, name, Path(value).expanduser().resolve())
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a registered method name")
        from wrobo.methods.registry import METHOD_TO_TRAINER

        self.method = self.method.upper()
        if self.method not in METHOD_TO_TRAINER:
            raise ValueError(
                f"Unknown method {self.method!r}; choose from {list(METHOD_TO_TRAINER)}"
            )
        for name in ("norm_stats_group", "instruction"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string when provided")
        if not isinstance(self.compile, bool) or not isinstance(self.amp, bool):
            raise ValueError("compile and amp must be booleans")
        if parse_device(self.device).is_remote:
            raise ValueError("Load the model on the server for a remote device")


def _json_mapping(path):
    with Path(path).open(encoding="utf-8") as file:
        result = json.load(file)
    if not isinstance(result, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return result


def _keys(value, label):
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(key, str) or not key.strip() for key in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{label} must contain unique nonempty column names")
    return list(value)


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _select_stats(options, bindings):
    stats = _json_mapping(options.norm_stats_key)
    # Flat statistics map channels to numeric moments; grouped statistics add
    # one dataset-name level. Recognize flat files even when a selected channel
    # is missing, so it is reported instead of treating channels as datasets.
    flat = any(key in stats for key in (bindings.proprio_key, bindings.action_key)) or any(
        isinstance(value, dict)
        and any(moment in value for moment in ("mean", "std", "q01", "q99"))
        for value in stats.values()
    )
    if flat:
        if options.norm_stats_group is not None:
            raise ValueError(
                "norm_stats_group is only used with per-dataset statistics"
            )
        return stats
    key = options.norm_stats_group
    if key is None:
        if len(stats) != 1:
            raise ValueError(
                "Provide norm_stats_group to select one dataset's normalization statistics"
            )
        key = next(iter(stats))
    if key not in stats or not isinstance(stats[key], dict):
        raise ValueError(
            f"Unknown normalization statistics group {options.norm_stats_group!r}"
        )
    return stats[key]


def _stats_vector(stats, key, size, *, mode="standard"):
    group = stats.get(key)
    if not isinstance(group, dict):
        raise ValueError(f"Missing normalization statistics for {key}")
    try:
        parameters = normalization_parameters(group, mode=mode)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid normalization statistics for {key}: {error}"
        ) from error
    vectors = []
    names = ("center", "scale") if mode == "quantile" else ("mean", "std")
    for name, parameter in zip(names, parameters, strict=True):
        array = np.asarray(parameter)
        if array.shape != (size,) or array.dtype.kind not in "fiu":
            raise ValueError(f"{key}.{name} must contain {size} numeric values")
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(np.float32)
        if not np.isfinite(array).all() or (
            name in ("std", "scale") and (array <= 0).any()
        ):
            raise ValueError(
                f"{key}.{name} must be finite; normalization scales must be positive"
            )
        vectors.append(array)
    return tuple(vectors)


def _stats_dimension(stats, key):
    group = stats.get(key)
    if not isinstance(group, dict):
        raise ValueError(f"Missing normalization statistics for {key}")
    values = np.asarray(group.get("mean"))
    if values.ndim != 1 or values.size < 1:
        raise ValueError(f"{key}.mean must contain a nonempty vector")
    return int(values.size)


def _image_transform(trainer):
    hook = getattr(trainer, "get_eval_img_transform", None)
    if hook is not None:
        return hook()
    # Registered Trainers' validation transforms are stateless. Calling the
    # unbound hook reuses the exact training transform without constructing a
    # Trainer and invoking its training/dataset initialization.
    return trainer.get_validation_img_transforms(None)


def _validate_model_dimensions(policy, method, state_dim, action_dim):
    # Adapter-only deployment wrappers contain a PEFT model. Its public getter
    # exposes the trained policy dimensions without loading another model.
    candidate = getattr(policy, "policy", policy)
    base_getter = getattr(candidate, "get_base_model", None)
    if callable(base_getter):
        candidate = base_getter()
    actual_state = getattr(candidate, "proprio_dim", None)
    actual_action = getattr(candidate, "action_dim", None)
    if method == "ACT":
        network = getattr(candidate, "model", None)
        actual_state = getattr(network, "proprio_dim", actual_state)
        actual_action = getattr(network, "action_dim", actual_state)
    elif method == "WVLA":
        actual_action = getattr(
            getattr(candidate, "action_expert", None), "action_dim", actual_action
        )
    elif method == "SMOLVLA":
        actual_action = getattr(
            getattr(candidate, "config", None), "action_dim", actual_action
        )
    for label, actual, expected in (
        ("state", actual_state, state_dim),
        ("action", actual_action, action_dim),
    ):
        if (
            actual is not None
            and _positive_int(actual, f"Model {label} dimension") != expected
        ):
            raise ValueError(
                f"Built model {label} dimension does not match normalization statistics"
            )


class LocalPolicy:
    """Prepare model inputs and return actions in the configured target units."""

    def __init__(
        self,
        *,
        policy,
        trainer,
        auxiliary,
        transform,
        device,
        metadata,
        state_stats,
        action_stats,
        amp=False,
    ):
        self.policy = policy
        self.trainer = trainer
        self.auxiliary = auxiliary
        self.transform = transform
        self.device = device
        self.metadata = metadata
        self.state_center, self.state_scale = state_stats
        self.action_center, self.action_scale = action_stats
        self.amp = amp
        self._closed = False

    def _inputs(self, window):
        import torch

        if not isinstance(window, dict):
            raise ValueError("Policy window must be a dictionary")
        state = np.asarray(window.get("state"))
        shape = (self.metadata["history_width"], self.metadata["state_dim"])
        if state.shape != shape or state.dtype.kind not in "fiu":
            raise ValueError(
                f"Raw policy state must be a numeric array with shape {shape}"
            )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            state = (state.astype(np.float32) - self.state_center) / self.state_scale
        if not np.isfinite(state).all():
            raise ValueError("Normalized policy state must be finite")
        raw_images = window.get("images")
        if not isinstance(raw_images, dict):
            raise ValueError(
                "Policy images must be a dictionary of camera history arrays"
            )
        images = {}
        for name in self.metadata["camera_keys"]:
            if name not in raw_images:
                raise ValueError(f"Missing camera {name!r} in policy window")
            array = np.asarray(raw_images[name])
            if (
                array.dtype != np.uint8
                or array.ndim != 4
                or array.shape[0] != shape[0]
                or array.shape[-1] != 3
                or min(array.shape[1:3]) < 1
            ):
                raise ValueError(
                    f"Camera {name!r} must contain uint8 RGB [history,H,W,3] frames"
                )
            frames = [
                self.transform(
                    torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
                )
                for frame in array
            ]
            for frame in frames:
                if (
                    not isinstance(frame, torch.Tensor)
                    or frame.ndim != 3
                    or frame.shape[0] != 3
                    or not frame.is_floating_point()
                    or not torch.isfinite(frame).all()
                ):
                    raise ValueError(
                        f"Camera {name!r} transform must return finite floating RGB [3,H,W]"
                    )
            images[name] = torch.stack(frames).unsqueeze(0)
        instruction = window.get("instruction", self.metadata.get("instruction"))
        if instruction is None:
            instruction = self.metadata.get("instruction")
        if instruction is not None and (
            not isinstance(instruction, str) or not instruction.strip()
        ):
            raise ValueError("Policy instruction must be a nonempty string")
        if self.metadata["requires_instruction"] and instruction is None:
            raise ValueError("This model requires an instruction for inference")
        bundle = {"image": images, "state": torch.from_numpy(state).unsqueeze(0)}
        if instruction is not None:
            bundle["instruction"] = instruction
        return self.trainer.build_policy_inputs(bundle, self.auxiliary, self.device)

    def predict(self, window):
        import torch

        if self._closed:
            raise RuntimeError("Policy is closed")
        inputs = self._inputs(window)
        with torch.inference_mode(), torch.autocast(
            self.device.type,
            dtype=torch.float16,
            enabled=self.amp and self.device.type == "cuda",
        ):
            output = self.policy(**inputs)
        if not isinstance(output, dict) or "pred" not in output:
            raise ValueError("Policy must return an output dictionary containing pred")
        prediction = output["pred"]
        expected = (1, self.metadata["action_chunk_size"], self.metadata["action_dim"])
        if (
            not isinstance(prediction, torch.Tensor)
            or tuple(prediction.shape) != expected
        ):
            raise ValueError(f"Policy pred must be a tensor with shape {expected}")
        if not prediction.is_floating_point() or not torch.isfinite(prediction).all():
            raise ValueError("Policy predictions must be finite floating values")
        normalized = prediction.detach().float().cpu().numpy()[0]
        with np.errstate(over="ignore", invalid="ignore"):
            actions = normalized * self.action_scale + self.action_center
        if not np.isfinite(actions).all():
            raise ValueError("Unnormalized policy actions must be finite")
        return np.ascontiguousarray(actions, dtype=np.float32)

    def close(self):
        self._closed = True
        self.policy = None
        self.auxiliary = None


def load_policy(options: ModelOptions):
    """Build through the registered Trainer and load a wrobo .pth checkpoint."""
    import torch
    import yaml
    from wrobo.methods import registry
    from wrobo.utils.load_model_weights import load_pretrained_weights

    if not isinstance(options, ModelOptions):
        raise TypeError("load_policy expects ModelOptions")
    for name in ("checkpoint", "config_file", "norm_stats_key"):
        path = getattr(options, name)
        if not path.is_file():
            raise ValueError(f"{name} file does not exist: {path}")
    artifact_dir = options.config_file.parent
    method = options.method
    with options.config_file.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("Policy"), dict)
        or not isinstance(config.get("Training_settings"), dict)
    ):
        raise ValueError("Model YAML requires Policy and Training_settings mappings")
    policy_config = copy.deepcopy(config["Policy"])
    data_keys = _keys(
        config["Training_settings"].get("data_keys"), "Training_settings.data_keys"
    )
    bindings = resolve_data_keys(data_keys)
    unsupported = [
        key
        for key in data_keys
        if key not in ("instruction", bindings.action_key, bindings.proprio_path)
        and not (
            key.startswith("observations/image_")
            and len(key) > len("observations/image_")
        )
    ]
    if unsupported:
        raise ValueError(f"Unsupported deployment training channels: {unsupported}")
    cameras = [
        key.removeprefix("observations/")
        for key in data_keys
        if key.startswith("observations/image_")
    ]
    if not cameras:
        raise ValueError(
            "Deployment requires at least one camera in training data_keys"
        )
    policy_config["camera_names"] = cameras
    requires_instruction = "instruction" in data_keys
    if method == "WVLA":
        language_config = policy_config.get("language_encoder")
        if requires_instruction != (language_config is not None):
            raise ValueError(
                "wVLA language_encoder and instruction training channel must be configured together"
            )
        if requires_instruction:
            if not isinstance(language_config, dict):
                raise ValueError("wVLA language_encoder must be a mapping")
            if not (artifact_dir / "language_model").is_dir():
                raise ValueError(
                    "Language-conditioned wVLA requires its training language_model artifacts"
                )
    description_path = options.data_description
    description = (
        _json_mapping(description_path) if description_path is not None else {}
    )
    if description_path is not None:
        try:
            json.dumps(description, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "data_description must contain finite JSON values"
            ) from exc
    state_keys = (
        _keys(description.get("state_keys"), "state_keys")
        if "state_keys" in description
        else None
    )
    action_keys = (
        _keys(description.get("action_keys"), "action_keys")
        if "action_keys" in description
        else None
    )
    robot_config = description.get("robot_config")
    if robot_config is not None and not isinstance(robot_config, dict):
        raise ValueError("data_description.robot_config must be a JSON object or null")
    stats = _select_stats(options, bindings)
    state_dim = _stats_dimension(stats, bindings.proprio_key)
    action_dim = _stats_dimension(stats, bindings.action_key)
    if state_keys is not None and len(state_keys) != state_dim:
        raise ValueError("State dimension does not match data_description.state_keys")
    if action_keys is not None and len(action_keys) != action_dim:
        raise ValueError(
            "Training action dimension does not match data_description.action_keys"
        )
    if "proprio_dim" in policy_config and policy_config["proprio_dim"] != state_dim:
        raise ValueError("Training proprio_dim does not match normalization statistics")
    if method == "ACT":
        configured_action = policy_config.get(
            "action_dim", policy_config.get("proprio_dim")
        )
    elif method == "WVLA":
        expert = policy_config.get("action_expert")
        if not isinstance(expert, dict):
            raise ValueError("wVLA Policy requires an action_expert mapping")
        configured_action = expert.get("action_dim")
    else:
        configured_action = policy_config.get("action_dim")
    if configured_action is not None and configured_action != action_dim:
        raise ValueError(
            "Training action dimension does not match normalization statistics"
        )
    normalization_mode = "quantile" if method == "PI05" else "standard"
    state_stats = _stats_vector(
        stats, bindings.proprio_key, state_dim, mode=normalization_mode
    )
    action_stats = _stats_vector(
        stats, bindings.action_key, action_dim, mode=normalization_mode
    )
    device = resolve_local_device(options.device)
    if options.amp and device.type != "cuda":
        raise ValueError("AMP deployment is supported on CUDA devices only")
    trainer = registry.get_trainer_class(registry.METHOD_TO_TRAINER[method])
    transform = _image_transform(trainer)
    policy, auxiliary = trainer.build_policy(
        policy_config, artifact_dir=str(artifact_dir)
    )
    _validate_model_dimensions(policy, method, state_dim, action_dim)
    if method == "WVLA" and requires_instruction:
        if (
            not isinstance(auxiliary, dict)
            or not callable(auxiliary.get("tokenizer"))
            or "text_max_len" not in auxiliary
        ):
            raise ValueError(
                "Language-conditioned wVLA did not provide its training tokenizer"
            )
        _positive_int(auxiliary["text_max_len"], "wVLA text_max_len")
    history = _positive_int(policy_config.get("history_width", 1), "history_width")
    chunk = policy_config.get("action_chunk_size")
    if chunk is None:
        chunk = policy_config.get("action_expert", {}).get("action_chunk_size")
    chunk = _positive_int(chunk, "action_chunk_size")
    load_pretrained_weights(policy, str(options.checkpoint))
    policy.to(device).eval()
    if options.compile:
        policy = torch.compile(policy)
    metadata = {
        "proprio_path": bindings.proprio_path,
        "proprio_key": bindings.proprio_key,
        "action_key": bindings.action_key,
        "state_keys": state_keys,
        "action_keys": action_keys,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "camera_keys": cameras,
        "history_width": history,
        "action_chunk_size": chunk,
        "robot_config": copy.deepcopy(robot_config),
        "requires_instruction": requires_instruction,
        "instruction": options.instruction,
        "method": method,
        "compute_device": str(device),
    }
    return LocalPolicy(
        policy=policy,
        trainer=trainer,
        auxiliary=auxiliary,
        transform=transform,
        device=device,
        metadata=metadata,
        state_stats=state_stats,
        action_stats=action_stats,
        amp=options.amp,
    )
