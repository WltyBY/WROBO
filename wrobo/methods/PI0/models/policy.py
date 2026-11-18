"""wrobo I/O adapter around the vendored official continuous-action PI0 core."""

from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from wrobo.methods.PI0.models.configuration import PI0Config
from wrobo.methods.PI0.models.modeling_pi0 import PI0Pytorch
from wrobo.methods.PI0.models.utils import pad_vector, resize_with_pad_torch
from wrobo.methods.PI0.models.checkpoint import resolve_checkpoint, load_official_weights


class PI0Policy(nn.Module):
    config_class = PI0Config
    model_class = PI0Pytorch
    policy_type = "pi0"

    def __init__(self, policy_config):
        super().__init__()
        self.config = self.config_class.from_policy_dict(policy_config)
        self.action_dim = int(policy_config.get("action_dim", 6))
        self.proprio_dim = int(policy_config.get("proprio_dim", 6))
        if not 1 <= self.action_dim <= 32 or not 1 <= self.proprio_dim <= 32:
            raise ValueError("Robot action/state dimensions must be between 1 and 32")
        self.camera_names = list(policy_config.get("camera_names", []))
        self.pretrained_load_mode = policy_config.get("pretrained_load_mode", "full")
        self.pretrained_policy_path = policy_config.get(
            "pretrained_policy_path", f"lerobot/{self.policy_type}_base"
        )
        if self.pretrained_load_mode not in ("none", "full"):
            raise ValueError("PI0 supports pretrained_load_mode=none or full")
        self.pretrained_revision = policy_config.get("pretrained_revision")
        self.pretrained_load_report = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.text_tokenizer_name,
            padding_side="right",
            local_files_only=policy_config.get("local_files_only", False),
        )
        checkpoint = None
        if self.pretrained_load_mode == "full":
            if not self.pretrained_policy_path:
                raise ValueError("Full initialization requires pretrained_policy_path")
            checkpoint = resolve_checkpoint(
                self.pretrained_policy_path,
                revision=self.pretrained_revision,
                local_files_only=policy_config.get("local_files_only", False),
            )
            # Cache snapshot paths contain the resolved immutable HF commit.
            if checkpoint.parent.name == "snapshots":
                self.pretrained_revision = checkpoint.name
            metadata_path = (
                checkpoint if checkpoint.is_dir() else checkpoint.parent
            ) / "config.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("type", self.policy_type) != self.policy_type:
                    raise ValueError(f"Expected {self.policy_type} weights, got {metadata.get('type')}")
                for key in (
                    "paligemma_variant",
                    "action_expert_variant",
                    "max_state_dim",
                    "max_action_dim",
                ):
                    if key in metadata and metadata[key] != getattr(self.config, key):
                        raise ValueError(
                            f"Official checkpoint architecture mismatch: {key}"
                        )
        self.model = self.model_class(self.config)
        if checkpoint is not None:
            self.pretrained_load_report = load_official_weights(self, checkpoint)
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    @property
    def action_chunk_size(self):
        return self.config.chunk_size

    def save_artifacts(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(folder)
        (folder / f"{self.policy_type}_config.json").write_text(
            json.dumps(asdict(self.config), indent=2)
        )
        if self.pretrained_load_report is not None:
            (folder / "weight_load_report.json").write_text(
                json.dumps(
                    {
                        **self.pretrained_load_report,
                        "source": str(self.pretrained_policy_path),
                        "revision": self.pretrained_revision,
                    },
                    indent=2,
                )
            )

    def _images(self, image, camera_names):
        names = self.camera_names or list(camera_names or image.keys())
        present = [name for name in names if name in image]
        if not present:
            raise ValueError("PI0 requires at least one configured camera")
        images, masks = [], []
        for name in present:
            value = image[name]
            if value.ndim == 5:
                if value.shape[1] != 1:
                    raise ValueError("PI0 only accepts one observation frame")
                value = value[:, 0]
            if value.ndim != 4 or value.shape[1] != 3:
                raise ValueError("Expected RGB images [B,3,H,W] or [B,1,3,H,W]")
            value = value.float()
            value = resize_with_pad_torch(value, 224, 224) * 2 - 1
            images.append(value)
            masks.append(
                torch.ones(value.shape[0], dtype=torch.bool, device=value.device)
            )
        for _ in range(len(names) - len(present)):
            images.append(torch.full_like(images[0], -1))
            masks.append(torch.zeros_like(masks[0]))
        return images, masks

    def _language(self, language, batch_size, device):
        if isinstance(language, dict):
            return (
                language["input_ids"].to(device),
                language["attention_mask"].to(device).bool(),
            )
        if isinstance(language, torch.Tensor):
            tokens = language.to(device).long()
            return tokens, tokens != self.tokenizer.pad_token_id
        if language is None:
            language = [""] * batch_size
        elif isinstance(language, str):
            language = [language] * batch_size
        if len(language) != batch_size:
            raise ValueError("Language batch size must match state batch size")
        prompts = [text if text.endswith("\n") else text + "\n" for text in language]
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device).bool(),
        )

    def forward(
        self,
        image,
        proprio_state,
        language=None,
        actions=None,
        camera_names=None,
        noise=None,
        time=None,
    ):
        if proprio_state.ndim == 3:
            if proprio_state.shape[1] != 1:
                raise ValueError("PI0 requires history_width=1")
            proprio_state = proprio_state[:, 0]
        if proprio_state.shape[-1] != self.proprio_dim:
            raise ValueError("State dimension does not match proprio_dim")
        state = pad_vector(proprio_state.float(), 32)
        images, masks = self._images(image, camera_names)
        tokens, token_mask = self._language(language, state.shape[0], state.device)
        if actions is None:
            prediction = self.model.sample_actions(
                images,
                masks,
                tokens,
                token_mask,
                state,
                noise=noise,
            )[..., : self.action_dim]
            return {"action_chunk": prediction, "pred": prediction}
        if actions.shape[1:] != (self.config.chunk_size, self.action_dim):
            raise ValueError("Expected actions [B, action_chunk_size, action_dim]")
        actions = pad_vector(actions.float(), 32)
        noise = (
            self.model.sample_noise(actions.shape, actions.device)
            if noise is None
            else noise
        )
        time = (
            self.model.sample_time(actions.shape[0], actions.device)
            if time is None
            else time
        )
        losses = self.model(
            images,
            masks,
            tokens,
            token_mask,
            state,
            actions,
            noise,
            time,
        )
        return {"loss_per_element": losses[..., : self.action_dim]}

    def _get_default_peft_targets(self):
        # Exactly LeRobot PI0Policy._get_default_peft_targets, including projections.
        common = "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        return {
            "target_modules": rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common}))",
            "modules_to_save": [],
        }
