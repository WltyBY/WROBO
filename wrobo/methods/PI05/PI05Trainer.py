"""PI0.5 training orchestration; architecture and prompt processing live in models."""
import torch
from wrobo.methods.PI0.PI0Trainer import PI0Trainer
from wrobo.methods.PI05.models.configuration import PI05Config
from wrobo.methods.PI05.models.policy import PI05Policy


def raw_instruction(text):
    # State is sampled with the observation. Tokenization must happen AFTER batching.
    return str(text)


class PI05Trainer(PI0Trainer):
    normalization_mode = "quantile"
    model_artifact_dirname = "pi05_config"
    lora_artifact_filename = "pi05_lora_config.json"
    lora_checkpoint_marker = "pi05_lora_adapter_checkpoint"

    def get_default_training_args(self, training_args):
        super().get_default_training_args(training_args)
        PI05Config.from_policy_dict(self.config_dict["Policy"])

    @staticmethod
    def get_policy(policy_config):
        return PI05Policy(policy_config)

    def get_text_transform(self):
        return raw_instruction

    def initialize(self):
        if self.compile and torch.__version__.startswith("2.14"):
            raise ValueError("PI family compiled checkpointed training has OOM reports on torch 2.14; disable compile or use torch 2.11")
        super().initialize()
