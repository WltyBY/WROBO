"""Method registry + run-name (log path) scheme.

A single source of truth mapping a Trainer's class name to its class, plus the helpers
that encode/parse the run-name segment used in log paths. This lets eval resolve *which*
Trainer built a model purely from its log dir, and call that Trainer's policy-build hooks
(so a model is always reconstructed by the method that trained it).

Run-name scheme (the segment under <log_dir>/<method_name>/):
    "<TrainerName>__K1_V1__K2_V2__..."
  - pairs are separated by "__", and the key/value inside a pair by the last "_".
  - the leading segment has no value (it's the Trainer class name).
  e.g. "ACTTrainer__BS_8__GA_1__EPOCH_300__SEED_319__PRETRAINED_False__SAMPLE_uniform".

Parse with split("__") then rsplit("_", 1) on each pair, so keys may themselves contain
single underscores.
"""

# Map config method name to Trainer class name
METHOD_TO_TRAINER = {
    "ACT": "ACTTrainer",
    "PI0": "PI0Trainer",
    "PI05": "PI05Trainer",
    "SMOLVLA": "SmolVLATrainer",
    "WVLA": "wVLATrainer",
}


def get_trainer_class(name: str):
    """Resolve a Trainer class by its class name. Imports lazily to avoid import cycles
    (Trainers import training utilities that may import this module)."""
    if name == "PI05Trainer":
        from wrobo.methods.PI05.PI05Trainer import PI05Trainer

        return PI05Trainer
    if name == "PI0Trainer":
        from wrobo.methods.PI0.PI0Trainer import PI0Trainer

        return PI0Trainer
    if name == "SmolVLATrainer":
        from wrobo.methods.SmolVLA.SmolVLATrainer import SmolVLATrainer

        return SmolVLATrainer
    if name == "ACTTrainer":
        from wrobo.methods.ACT.ACTTrainer import ACTTrainer

        return ACTTrainer
    if name == "wVLATrainer":
        from wrobo.methods.wVLA.wVLATrainer import wVLATrainer

        return wVLATrainer
    raise KeyError(
        f"Unknown trainer '{name}'. Register it in wrobo/methods/registry.py."
    )


def build_run_name(trainer_name: str, pairs: dict) -> str:
    """Encode the run-name segment: '<trainer>__K_V__K_V...'."""
    parts = [trainer_name] + [f"{k}_{v}" for k, v in pairs.items()]
    return "__".join(parts)


def parse_run_name(run_segment: str):
    """Parse '<trainer>__K_V__...' -> (trainer_name, {K: V}) with V kept as a string.

    Robust to keys containing single underscores (splits the value off the last '_').
    """
    parts = run_segment.split("__")
    trainer_name = parts[0]
    pairs = {}
    for token in parts[1:]:
        if "_" not in token:
            continue
        key, value = token.rsplit("_", 1)
        pairs[key] = value
    return trainer_name, pairs
