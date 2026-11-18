"""Shared LoRA part selections and deployment exports that preserve the live model."""

import torch


LORA_PARTS = ("language", "vision", "action", "projection")
DEFAULT_LORA_PARTS = ("action", "projection")


def normalize_lora_parts(parts):
    """Validate model-part selections and give equivalent inputs a stable order."""
    if not parts:
        raise ValueError("lora_parts must select at least one part")
    if isinstance(parts, str):
        parts = [parts]
    selected = set(parts)
    if not selected:
        raise ValueError("lora_parts must select at least one part")
    if selected == {"all"}:
        return LORA_PARTS
    unknown = selected.difference(LORA_PARTS)
    if unknown:
        raise ValueError(
            f"Invalid lora_parts: {sorted(unknown)}; choose from {LORA_PARTS}, "
            "or use 'all' alone"
        )
    return tuple(part for part in LORA_PARTS if part in selected)


@torch.no_grad()
def merged_lora_state_dict(policy):
    """Build plain deployment weights, preserving the live base and adapter state."""
    from peft.tuners.lora.layer import LoraLayer

    base_policy = policy.get_base_model()
    layers = []
    for name, module in base_policy.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if module.merged:
            raise ValueError(f"Cannot export an already merged LoRA layer: {name}")
        if module.disable_adapters:
            raise ValueError(f"Cannot export a disabled LoRA layer: {name}")
        adapters = [adapter for adapter in module.active_adapters if adapter in module.lora_A]
        if not adapters:
            raise ValueError(f"LoRA layer has no active adapters: {name}")
        if not isinstance(module.get_base_layer(), torch.nn.Linear):
            raise NotImplementedError("Merged export supports ordinary Linear LoRA")
        for adapter in adapters:
            if module.use_dora.get(adapter, False):
                raise NotImplementedError("Merged export does not support DoRA")
            if (module.lora_A[adapter].bias is not None
                    or module.lora_B[adapter].bias is not None):
                raise NotImplementedError("Merged export does not support adapter biases")
        layers.append((name, module, adapters))
    if not layers:
        raise ValueError("Policy has no active LoRA layers to export")

    merged_state = {}
    for key, value in base_policy.state_dict().items():
        if ".lora_" in key:
            continue
        if ".modules_to_save." in key or ".original_module." in key:
            raise NotImplementedError("Merged export does not support modules_to_save")
        key = key.replace(".base_layer.", ".")
        if key in merged_state:
            raise RuntimeError(f"Duplicate key after merging LoRA state: {key}")
        merged_state[key] = value.detach().cpu()

    for name, module, adapters in layers:
        # Floating-point (W + delta) - delta need not equal W, even in FP32.
        # Merge into a temporary tensor so final export cannot contaminate best.
        weight = module.get_base_layer().weight.detach().clone()
        with torch.autocast(weight.device.type, enabled=False):
            for adapter in adapters:
                weight_a = module.lora_A[adapter].weight.detach()
                weight_b = module.lora_B[adapter].weight.detach()
                dtype = torch.promote_types(weight_a.dtype, weight_b.dtype)
                if dtype in (torch.float16, torch.bfloat16):
                    dtype = torch.float32
                delta = weight_b.to(dtype) @ weight_a.to(dtype)
                if module.fan_in_fan_out:
                    delta = delta.T
                # Preserve PEFT safe_merge's cast-before-add rounding rule.
                weight.add_((delta * module.scaling[adapter]).to(weight.dtype))
        if not torch.isfinite(weight).all().item():
            raise ValueError(f"Cannot export non-finite merged LoRA weights: {name}")
        merged_state[name + ".weight"] = weight.cpu()
    return merged_state
