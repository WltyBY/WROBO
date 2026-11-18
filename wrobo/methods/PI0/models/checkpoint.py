"""Strict official safetensors loading, including tied embeddings and sharded files."""

import json
from pathlib import Path

import torch
from safetensors import safe_open


def _read_files(paths):
    """Inspect keys first; do not materialize unused vocabulary-head tensors."""
    keys = set()
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as file:
            overlap = keys.intersection(file.keys())
            if overlap:
                raise ValueError(f"Duplicate tensors in shards: {sorted(overlap)}")
            keys.update(file.keys())
    has_embedding = any(key.endswith("paligemma.model.language_model.embed_tokens.weight")
                        or key.endswith("paligemma.language_model.embed_tokens.weight") for key in keys)
    weights = {}
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as file:
            for key in file.keys():
                if key.endswith("paligemma_with_expert.gemma_expert.lm_head.weight"):
                    continue
                if has_embedding and key.endswith("paligemma_with_expert.paligemma.lm_head.weight"):
                    continue
                weights[key] = file.get_tensor(key)
    return weights, keys


def resolve_checkpoint(path, revision=None, local_files_only=False):
    path = str(path)
    if Path(path).exists():
        return Path(path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=path, revision=revision, local_files_only=local_files_only,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    ))


def read_weights(path):
    path = Path(path)
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError("Official weights must be safetensors; use --pretrained_weight for wrobo .pth")
        return _read_files([path])[0]
    single = path / "model.safetensors"
    if single.is_file():
        return _read_files([single])[0]
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(f"No model.safetensors or shard index in {path}")
    weight_map = json.loads(index.read_text())["weight_map"]
    paths = []
    for shard in sorted(set(weight_map.values())):
        shard_path = (path / shard).resolve()
        if not shard_path.is_relative_to(path.resolve()):
            raise ValueError(f"Invalid shard path: {shard}")
        paths.append(shard_path)
    weights, keys = _read_files(paths)
    if keys != set(weight_map):
        raise ValueError("Shard tensors do not match the checkpoint index")
    return weights


def canonical_weights(weights, expected):
    result = {}
    for key, value in weights.items():
        key = key.removeprefix("module.").removeprefix("model.")
        if key.startswith("time_mlp_in.") and "model.time_mlp_in.weight" not in expected:
            key = "action_time_mlp_in." + key[len("time_mlp_in."):]
        elif key.startswith("time_mlp_out.") and "model.time_mlp_out.weight" not in expected:
            key = "action_time_mlp_out." + key[len("time_mlp_out."):]
        for component in ("language_model", "vision_tower", "multi_modal_projector"):
            key = key.replace(
                f"paligemma_with_expert.paligemma.{component}.",
                f"paligemma_with_expert.paligemma.model.{component}.",
            )
        key = "model." + key
        if key in result:
            raise ValueError(f"Duplicate key after checkpoint mapping: {key}")
        result[key] = value
    prefix = "model.paligemma_with_expert.paligemma."
    embedding = prefix + "model.language_model.embed_tokens.weight"
    head = prefix + "lm_head.weight"
    # Hugging Face safetensors may store only one of these tied weights.
    for missing, present in ((embedding, head), (head, embedding)):
        if missing in expected and missing not in result and present in result:
            result[missing] = result[present]
    result.pop(head, None)
    result.pop("model.paligemma_with_expert.gemma_expert.lm_head.weight", None)
    return result


def load_official_weights(policy, path):
    expected = policy.state_dict()
    weights = canonical_weights(read_weights(path), expected)
    missing = sorted(expected.keys() - weights.keys())
    unexpected = sorted(weights.keys() - expected.keys())
    mismatched = [
        (key, tuple(weights[key].shape), tuple(expected[key].shape))
        for key in expected.keys() & weights.keys()
        if weights[key].shape != expected[key].shape
    ]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"{getattr(policy, 'policy_type', 'PI')} checkpoint is incompatible: missing={missing}, "
            f"unexpected={unexpected}, shape_mismatches={mismatched}"
        )
    policy.load_state_dict(weights, strict=True)
    return {"tensors_loaded": len(weights), "missing_keys": [], "unexpected_keys": []}
