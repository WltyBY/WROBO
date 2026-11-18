"""Configuration for the standalone wrobo SmolVLA implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SmolVLAConfig:
    # wrobo input/output contract.
    history_width: int = 1
    action_chunk_size: int = 50
    proprio_dim: int = 14
    action_dim: int = 14
    camera_names: tuple[str, ...] = ()

    # Official SmolVLA architecture defaults.
    max_state_dim: int = 32
    max_action_dim: int = 32
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    tokenizer_max_length: int = 48
    pad_language_to: str = "longest"
    task_prompt: str = ""
    num_steps: int = 10
    use_cache: bool = True
    freeze_vision_encoder: bool = True
    freeze_language_encoder: bool = True
    train_state_proj: bool = True
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False
    add_image_special_tokens: bool = False
    attention_mode: str = "cross_attn"
    prefix_length: int = -1
    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75
    min_period: float = 4e-3
    max_period: float = 4.0

    @property
    def chunk_size(self) -> int:
        """Official name retained for computation/checkpoint parity."""
        return self.action_chunk_size

    def get(self, key: str, default=None):
        """Provide the mapping-style access expected by wrobo's generic evaluator."""
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, values: dict) -> "SmolVLAConfig":
        aliases = {
            "chunk_size": "action_chunk_size",
            "model_name": "vlm_model_name",
            "num_inference_steps": "num_steps",
            "train_expert_only": "freeze_language_encoder",
            "freeze_language_model": "freeze_language_encoder",
        }
        normalized = dict(values)
        for old, new in aliases.items():
            if old in normalized and new not in normalized:
                normalized[new] = normalized[old]
        fields = cls.__dataclass_fields__
        kwargs = {key: value for key, value in normalized.items() if key in fields}
        if "camera_names" in kwargs:
            kwargs["camera_names"] = tuple(kwargs["camera_names"] or ())
        if "resize_imgs_with_padding" in kwargs:
            kwargs["resize_imgs_with_padding"] = tuple(kwargs["resize_imgs_with_padding"])
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self):
        if self.proprio_dim > self.max_state_dim:
            raise ValueError("proprio_dim cannot exceed max_state_dim")
        if self.action_dim > self.max_action_dim:
            raise ValueError("action_dim cannot exceed max_action_dim")
        if self.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")
        if self.pad_language_to not in {"longest", "max_length"}:
            raise ValueError("pad_language_to must be 'longest' or 'max_length'")
