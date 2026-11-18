"""LeRobot PI0 architecture with OpenPI PyTorch training precision defaults."""

from dataclasses import dataclass


@dataclass
class PI0Config:
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"
    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    image_resolution: tuple = (224, 224)
    freeze_vision_encoder: bool = False
    freeze_language_encoder: bool = False
    gradient_checkpointing: bool = False
    rtc_config: object = None
    tokenizer_max_length: int = 48
    text_tokenizer_name: str = "google/paligemma-3b-pt-224"

    @classmethod
    def from_policy_dict(cls, settings):
        # Old run configs remain loadable; new configs use independent switches.
        settings = dict(settings)
        if settings.pop("train_expert_only", False):
            if settings.get("freeze_language_encoder") is False:
                raise ValueError("Legacy train_expert_only=True conflicts with freeze_language_encoder=False")
            settings["freeze_vision_encoder"] = True
            settings["freeze_language_encoder"] = True
        if settings.get("history_width", 1) != 1:
            raise ValueError("PI0 requires history_width=1")
        if settings.get("deep_supervision", False):
            raise ValueError("PI0 does not support deep supervision")
        config = cls(**{
            key: settings[key] for key in cls.__dataclass_fields__ if key in settings
        })
        config.chunk_size = settings.get("action_chunk_size", config.chunk_size)
        if config.paligemma_variant != "gemma_2b" or config.action_expert_variant != "gemma_300m":
            raise ValueError("This port uses the official pi0_base architecture")
        if config.max_state_dim != 32 or config.max_action_dim != 32:
            raise ValueError("Official PI0 projections must remain 32-dimensional")
        if tuple(config.image_resolution) != (224, 224):
            raise ValueError("Official pi0_base uses 224x224 images")
        config.image_resolution = (224, 224)
        if config.dtype not in ("float32", "bfloat16"):
            raise ValueError("PI0 dtype must be float32 or bfloat16")
        if min(config.chunk_size, config.num_inference_steps, config.tokenizer_max_length) < 1:
            raise ValueError("Chunk size, inference steps and token length must be positive")
        if config.rtc_config is not None:
            raise ValueError("RTC is outside this training port")
        return config
