"""SmolVLA model and wrobo policy adapter.

``VLAFlowMatching`` mirrors the LeRobot computation and parameter names. The
``SmolVLAPolicy`` wrapper only translates wrobo's inputs and output dictionary.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wrobo.methods.SmolVLA.models.configuration import SmolVLAConfig
from wrobo.methods.SmolVLA.models.smolvlm_with_expert import SmolVLMWithExpertModel
from wrobo.methods.SmolVLA.models.utils import (
    create_sinusoidal_pos_embedding,
    euler_integrate,
    make_att_2d_masks,
    pad_sequence,
    pad_vector,
    require_package,
    resize_with_pad,
    sample_noise,
    sample_time_beta,
)


class VLAFlowMatching(nn.Module):
    """The checkpoint-compatible SmolVLA flow-matching network."""

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config
        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=config.vlm_model_name,
            freeze_vision_encoder=config.freeze_vision_encoder,
            freeze_language_encoder=config.freeze_language_encoder,
            load_vlm_weights=config.load_vlm_weights,
            attention_mode=config.attention_mode,
            num_expert_layers=config.num_expert_layers,
            num_vlm_layers=config.num_vlm_layers,
            self_attn_every_n_layers=config.self_attn_every_n_layers,
            expert_width_multiplier=config.expert_width_multiplier,
            device="auto",
        )
        text_width = self.vlm_with_expert.config.text_config.hidden_size
        expert_width = self.vlm_with_expert.expert_hidden_size
        self.state_proj = nn.Linear(config.max_state_dim, text_width)
        self.action_in_proj = nn.Linear(config.max_action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(expert_width * 2, expert_width)
        self.action_time_mlp_out = nn.Linear(expert_width, expert_width)

        tokenizer = self.vlm_with_expert.processor.tokenizer
        self.global_image_start_token = torch.tensor(
            [tokenizer.fake_image_token_id, tokenizer.global_image_token_id],
            dtype=torch.long,
        )
        self.image_end_token = torch.tensor(
            [tokenizer.fake_image_token_id], dtype=torch.long
        )
        self.add_image_special_tokens = config.add_image_special_tokens
        self.prefix_length = config.prefix_length
        for parameter in self.state_proj.parameters():
            parameter.requires_grad = config.train_state_proj

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        embeddings, pad_masks, attention_blocks = [], [], []
        for image, image_mask in zip(images, img_masks, strict=False):
            if self.add_image_special_tokens:
                start = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(
                            self.vlm_with_expert.vlm.device
                        )
                    )
                    .unsqueeze(0)
                    .expand(image.shape[0], -1, -1)
                )
                start_mask = torch.ones(
                    start.shape[:2], dtype=torch.bool, device=start.device
                )
                embeddings.append(start)
                pad_masks.append(start_mask)
                attention_blocks.extend([0] * start.shape[1])

            image_embedding = self.vlm_with_expert.embed_image(image)
            image_embedding = image_embedding * (image_embedding.shape[-1] ** 0.5)
            image_mask = image_mask[:, None].expand(
                image.shape[0], image_embedding.shape[1]
            )
            embeddings.append(image_embedding)
            pad_masks.append(image_mask)
            attention_blocks.extend([0] * image_embedding.shape[1])

            if self.add_image_special_tokens:
                end = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(image.shape[0], -1, -1)
                )
                end_mask = torch.ones(
                    end.shape[:2], dtype=torch.bool, device=end.device
                )
                embeddings.append(end)
                pad_masks.append(end_mask)
                attention_blocks.extend([0] * end.shape[1])

        language_embedding = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        language_embedding = language_embedding * (language_embedding.shape[-1] ** 0.5)
        embeddings.append(language_embedding)
        pad_masks.append(lang_masks)
        attention_blocks.extend([0] * language_embedding.shape[1])

        state_embedding = self.state_proj(state)
        if state_embedding.ndim == 2:
            state_embedding = state_embedding[:, None, :]
        embeddings.append(state_embedding)
        state_mask = torch.ones(
            state_embedding.shape[:2], dtype=torch.bool, device=state.device
        )
        pad_masks.append(state_mask)
        attention_blocks.extend([1] * state_embedding.shape[1])

        embeddings = torch.cat(embeddings, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        attention_blocks = torch.tensor(
            attention_blocks, dtype=torch.bool, device=pad_masks.device
        )[None].expand(embeddings.shape[0], -1)
        if self.prefix_length > 0 and embeddings.shape[1] < self.prefix_length:
            embeddings = pad_sequence(embeddings, self.prefix_length)
            pad_masks = pad_sequence(pad_masks, self.prefix_length)
            attention_blocks = pad_sequence(attention_blocks, self.prefix_length)
        return embeddings, pad_masks, attention_blocks

    def embed_suffix(self, noisy_actions, timestep):
        action_embedding = self.action_in_proj(noisy_actions)
        time_embedding = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            action_embedding.device,
        ).to(dtype=action_embedding.dtype)
        time_embedding = time_embedding[:, None, :].expand_as(action_embedding)
        fused = self.action_time_mlp_in(
            torch.cat([action_embedding, time_embedding], dim=2)
        )
        fused = self.action_time_mlp_out(F.silu(fused))
        mask = torch.ones(fused.shape[:2], dtype=torch.bool, device=fused.device)
        blocks = torch.ones(fused.shape[:2], dtype=torch.bool, device=fused.device)
        return fused, mask, blocks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
    ):
        if noise is None:
            noise = sample_noise(actions.shape, actions.device)
        if time is None:
            time = sample_time_beta(actions.shape[0], actions.device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        target = noise - actions
        prefix = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
        suffix = self.embed_suffix(x_t, time)
        pad_masks = torch.cat([prefix[1], suffix[1]], dim=1)
        attention_blocks = torch.cat([prefix[2], suffix[2]], dim=1)
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(pad_masks, attention_blocks),
            position_ids=torch.cumsum(pad_masks, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix[0], suffix[0]],
            use_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].float()
        return self.action_out_proj(suffix_out), target

    @torch.no_grad()
    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None
    ):
        if noise is None:
            noise = sample_noise(
                (state.shape[0], self.config.chunk_size, self.config.max_action_dim),
                state.device,
            )
        prefix = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
        _, cache = self.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(prefix[1], prefix[2]),
            position_ids=torch.cumsum(prefix[1], dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix[0], None],
            use_cache=self.config.use_cache,
        )
        return euler_integrate(
            lambda x, t: self.denoise_step(prefix[1], cache, x, t),
            noise,
            self.config.num_steps,
        )

    def denoise_step(self, prefix_pad_masks, cache, x_t, timestep):
        suffix = self.embed_suffix(x_t, timestep)
        suffix_len = suffix[1].shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_attention = prefix_pad_masks[:, None, :].expand(
            prefix_pad_masks.shape[0], suffix_len, prefix_len
        )
        outputs, _ = self.vlm_with_expert.forward(
            attention_mask=torch.cat(
                [prefix_attention, make_att_2d_masks(suffix[1], suffix[2])], dim=2
            ),
            position_ids=prefix_pad_masks.sum(dim=-1)[:, None]
            + torch.cumsum(suffix[1], dim=1)
            - 1,
            past_key_values=cache,
            inputs_embeds=[None, suffix[0]],
            use_cache=self.config.use_cache,
        )
        if cache is not None:
            cache.crop(prefix_len)
        return self.action_out_proj(outputs[1][:, -self.config.chunk_size :].float())


class SmolVLAPolicy(nn.Module):
    """wrobo-facing adapter around the official flow-matching network."""

    _FULL_CHECKPOINT_SHAPE_MISMATCH_ALLOWLIST = (
        "model.state_proj.",
        "model.action_in_proj.",
        "model.action_out_proj.",
    )

    def __init__(self, policy_config: dict):
        super().__init__()
        self.raw_config = dict(policy_config)
        self.pretrained_policy_path = policy_config.get("pretrained_policy_path")
        mode = policy_config.get("pretrained_load_mode")
        if mode is None:
            if self.pretrained_policy_path is None:
                mode = "none"
            elif "smolvlm" in self.pretrained_policy_path.lower():
                mode = "vlm"
            else:
                mode = "full"
        if mode not in {"none", "vlm", "full"}:
            raise ValueError("pretrained_load_mode must be one of: none, vlm, full")
        self.pretrained_load_mode = mode

        values = dict(policy_config)
        default_vlm = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        if mode == "vlm":
            if not self.pretrained_policy_path:
                raise ValueError(
                    "pretrained_load_mode='vlm' requires pretrained_policy_path"
                )
            values["vlm_model_name"] = self.pretrained_policy_path
            values["load_vlm_weights"] = True
        elif mode == "full":
            if not self.pretrained_policy_path:
                raise ValueError(
                    "pretrained_load_mode='full' requires pretrained_policy_path"
                )
            values["vlm_model_name"] = self._infer_full_vlm_model(
                self.pretrained_policy_path,
                policy_config.get("vlm_model_name", default_vlm),
            )
            # The policy checkpoint supplies all model tensors. Loading the base VLM first
            # is unnecessary and can overwrite the intended three-mode semantics.
            values["load_vlm_weights"] = False
        else:
            values["vlm_model_name"] = policy_config.get("vlm_model_name", default_vlm)
            values["load_vlm_weights"] = False
        self.config = SmolVLAConfig.from_dict(values)
        self.model = VLAFlowMatching(self.config)
        if mode == "full":
            self._load_full_checkpoint(self.pretrained_policy_path)

    @property
    def action_chunk_size(self):
        return self.config.action_chunk_size

    def forward(
        self,
        image: dict[str, Tensor],
        proprio_state: Tensor,
        actions: Tensor | None = None,
        is_pad: Tensor | None = None,
        camera_names: list[str] | tuple[str, ...] | None = None,
        language: str | list[str] | Tensor | None = None,
    ):
        batch_size = proprio_state.shape[0]
        images, image_masks = self._prepare_images(image, camera_names, batch_size)
        state = proprio_state[:, -1] if proprio_state.ndim == 3 else proprio_state
        state = pad_vector(state, self.config.max_state_dim)
        lang_tokens, lang_masks = self._prepare_language(
            language, batch_size, state.device
        )
        if actions is not None:
            actions = pad_vector(
                actions[:, : self.config.chunk_size], self.config.max_action_dim
            )
            model_pred, target = self.model(
                images, image_masks, lang_tokens, lang_masks, state, actions
            )
            return {
                "model_pred": model_pred[:, :, : self.config.action_dim],
                "target": target[:, :, : self.config.action_dim],
            }

        predicted = self.model.sample_actions(
            images, image_masks, lang_tokens, lang_masks, state
        )
        predicted = predicted[:, :, : self.config.action_dim]
        return {"action_chunk": predicted, "pred": predicted}

    def _prepare_images(self, image, camera_names, batch_size):
        names = list(camera_names or image.keys())
        if not names:
            raise ValueError("SmolVLA requires at least one camera image")
        images, masks = [], []
        for name in names:
            if name not in image:
                raise KeyError(f"Camera {name!r} is missing from image input")
            current = image[name]
            if current.ndim == 5:
                current = current[:, -1]
            if current.ndim != 4:
                raise ValueError(
                    f"Camera {name!r} must have shape (B,T,C,H,W) or (B,C,H,W)"
                )
            current = current.float()
            if self.config.resize_imgs_with_padding is not None:
                target_w, target_h = self.config.resize_imgs_with_padding
                current = resize_with_pad(
                    current,
                    height=target_h,
                    width=target_w,
                    pad_value=0.0,
                )
            current = current * 2.0 - 1.0
            images.append(current)
            masks.append(
                torch.ones(batch_size, dtype=torch.bool, device=current.device)
            )
        return images, masks

    def _prepare_language(self, language, batch_size, device):
        tokenizer = self.model.vlm_with_expert.processor.tokenizer
        if isinstance(language, Tensor):
            tokens = language.to(device=device, dtype=torch.long)
            if tokens.ndim == 1:
                tokens = tokens[None].expand(batch_size, -1)
            return tokens, tokens != tokenizer.pad_token_id
        if isinstance(language, (str, type(None))):
            prompt = self.config.task_prompt if language is None else language
            texts = [prompt] * batch_size
        else:
            texts = list(language)
        if len(texts) != batch_size:
            raise ValueError(
                f"Expected {batch_size} language strings, got {len(texts)}"
            )
        texts = [str(text) for text in texts]
        texts = [text if text.endswith("\n") else text + "\n" for text in texts]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=self.config.pad_language_to,
            truncation=True,
            max_length=self.config.tokenizer_max_length,
        )
        return (
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device).bool(),
        )

    def _load_full_checkpoint(self, pretrained_path):
        path = self._resolve_checkpoint(pretrained_path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            checkpoint = load_file(str(path), device="cpu")
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint = checkpoint.get("network_weights", checkpoint)

        current = self.state_dict()
        loaded, shape_mismatches, unexpected = [], [], []
        matched_targets = set()
        with torch.no_grad():
            for key, value in checkpoint.items():
                key = key[7:] if key.startswith("module.") else key
                key = key[10:] if key.startswith("_orig_mod.") else key
                candidates = [key, "model." + key]
                if key.startswith("model."):
                    candidates.append(key[6:])
                target = next(
                    (candidate for candidate in candidates if candidate in current),
                    None,
                )
                if target is None:
                    unexpected.append(key)
                    continue
                matched_targets.add(target)
                if current[target].shape != value.shape:
                    shape_mismatches.append(
                        (
                            target,
                            f"{target}: checkpoint {tuple(value.shape)} != "
                            f"model {tuple(current[target].shape)}",
                        )
                    )
                    continue
                current[target].copy_(value.to(dtype=current[target].dtype))
                loaded.append(target)
        missing = sorted(set(current) - matched_targets)
        allow_partial = bool(
            self.raw_config.get("allow_partial_full_checkpoint", False)
        )
        disallowed_shape_mismatches = [
            description
            for target, description in shape_mismatches
            if not target.startswith(self._FULL_CHECKPOINT_SHAPE_MISMATCH_ALLOWLIST)
        ]
        if disallowed_shape_mismatches:
            raise RuntimeError(
                f"Checkpoint {path} has shape mismatches outside the explicit "
                "state/action projection allowlist. First mismatches: "
                f"{disallowed_shape_mismatches[:5]}"
            )
        if (missing or unexpected) and not allow_partial:
            raise RuntimeError(
                f"Checkpoint {path} is not a complete SmolVLA checkpoint: "
                f"{len(missing)} model keys are missing and {len(unexpected)} keys are unexpected. "
                "Set allow_partial_full_checkpoint=true only when partial loading is intentional. "
                f"First missing keys: {missing[:5]}; first unexpected keys: {unexpected[:5]}"
            )
        self.load_state_dict(current)
        print(
            f"SmolVLA full checkpoint: loaded {len(loaded)} tensors, "
            f"skipped {len(shape_mismatches)} shape-incompatible tensors"
        )
        if shape_mismatches:
            print("Shape-incompatible tensors (kept randomly initialized):")
            for _, mismatch in shape_mismatches[:10]:
                print(f"  - {mismatch}")

    @staticmethod
    def _infer_full_vlm_model(pretrained_path, fallback):
        """Use bundled HF config/processor files when a full checkpoint is local."""
        local = Path(pretrained_path)
        if not local.is_dir():
            return fallback
        config_file = local / "config.json"
        if not config_file.exists():
            return fallback
        try:
            metadata = json.loads(config_file.read_text())
        except (OSError, json.JSONDecodeError):
            return fallback
        if metadata.get("model_type") or metadata.get("architectures"):
            return str(local)
        return metadata.get("vlm_model_name", fallback)

    @staticmethod
    def _resolve_checkpoint(pretrained_path):
        local = Path(pretrained_path)
        if local.is_dir():
            for name in ("model.safetensors", "pytorch_model.bin", "model.bin"):
                candidate = local / name
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"No supported model checkpoint found in {local}")
        if local.is_file():
            return local
        require_package("huggingface_hub")
        from huggingface_hub import hf_hub_download

        errors = []
        for name in ("model.safetensors", "pytorch_model.bin", "model.bin"):
            try:
                return Path(
                    hf_hub_download(repo_id=str(pretrained_path), filename=name)
                )
            except Exception as error:
                errors.append(error)
        raise FileNotFoundError(
            f"Could not resolve a model checkpoint from {pretrained_path!r}: {errors[-1]}"
        )
