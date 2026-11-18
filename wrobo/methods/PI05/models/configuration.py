"""Standard single-observation LeRobot PI0.5 architecture configuration."""
from dataclasses import dataclass
from wrobo.methods.PI0.models.configuration import PI0Config


@dataclass
class PI05Config(PI0Config):
    tokenizer_max_length: int = 200

    @classmethod
    def from_policy_dict(cls, settings):
        for key in ("use_visual_memory", "use_proprioceptive_memory", "use_relative_actions",
                    "rtc_training_max_delay"):
            if settings.get(key):
                raise ValueError(f"{key} is outside this standard PI0.5 port")
        if settings.get("n_obs_steps", 1) != 1:
            raise ValueError("PI0.5 currently supports one observation")
        if settings.get("empty_cameras", 0):
            raise ValueError("Configure camera_names/data_keys instead of empty_cameras in wrobo")
        return super().from_policy_dict(settings)
