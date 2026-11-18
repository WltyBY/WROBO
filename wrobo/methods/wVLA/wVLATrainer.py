import os

from torch import nn, optim
from torchvision import transforms

from wrobo.methods.wVLA.models.policy import wVLAPolicy
from wrobo.training.lr_scheduler.CosineAnnealingLR import CosineAnnealingLRScheduler
from wrobo.training.trainer.ABCTrainer import DDPABCTrainer
from wrobo.training.trainer.loss import LossTerm
from wrobo.training.utils.mics import is_main_process


# Subdirectory (under the run's Config_and_Trainer folder) where the HuggingFace
# tokenizer + language-model config are cached at train time, so eval is fully
# self-contained and offline (mirrors how miniVLA copies its vocab/embedding artifacts).
LANGUAGE_ARTIFACT_DIRNAME = "language_model"


class HFTextTransform:
    """Tokenize a raw instruction string into a fixed-length ``input_ids`` LongTensor.

    The text-side analog of ``get_*_img_transforms``: the Trainer owns the tokenizer and
    hands this callable to the dataset, which applies it per sample in the dataloader
    workers and yields a model-ready ``(text_max_len,)`` tensor (not a raw string). The
    attention mask is reconstructed from ``pad_id`` in the Trainer/eval, so only the ids
    need to flow through the collater (which stacks tensors by key).

    A small picklable class (not a closure) so it survives ``persistent_workers`` and
    DataLoader process spawning.
    """

    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, text):
        encoded = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0).long()  # (max_len,)


class wVLATrainer(DDPABCTrainer):
    """Trainer for the modular wVLA vision-language-action policy.

    Responsibility split mirrors ACTTrainer / miniVLATrainer:
      - the model (wVLAPolicy) only runs the forward computation and returns a dict of
        raw tensors keyed by the action expert (``model_pred`` plus ``target``/
        ``loss_type`` for the stochastic experts, ``pred`` at eval);
      - this Trainer provides the expert loss, optimizer/scheduler, and language
        adapters; the base class owns initialization, data loading, and train/val loops.

    The language side is a HuggingFace transformer (see language_encoders.py), so the
    Trainer's text adapter is an HF tokenizer (the analog of miniVLA's custom vocab
    tokenizer). When the Policy config has no ``language_encoder`` block (or ``data_keys``
    omits ``instruction``), the policy runs vision-only and the language plumbing is
    skipped end to end.
    """

    def get_hyperparameters_name(self):
        expert_type = self.config_dict["Policy"]["action_expert"].get("type", "NA")
        return f"EXPERT_{expert_type}"

    def setup_method(self):
        self.check_language_setup()

    def initialize_loss(self):
        """Choose the action expert criterion once for training and validation."""
        if self.expert_type == "regression":
            self._loss_fn = nn.L1Loss(reduction="none")
            self._loss_target_key = "actions"
        elif self.expert_type in ("ddpm", "ddim", "flow_matching"):
            self._loss_fn = nn.MSELoss(reduction="none")
            self._loss_target_key = "target"
        else:
            raise ValueError(
                f"Unsupported action-expert type {self.expert_type!r} for loss; "
                "expected 'regression', 'ddpm', 'ddim', or 'flow_matching'."
            )

    def get_optimizers(self):
        # AdamW + cosine-annealing-with-warmup: the standard recipe for transformer
        # policies. Decoupled weight decay regularizes without distorting Adam's moments;
        # cosine decay anneals far better than a constant LR. Warmup uses --warmup_epoch.
        optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=self.config_dict["Training_settings"]["base_lr"],
            weight_decay=self.config_dict["Training_settings"]["weight_decay"],
            betas=(0.9, 0.999),
        )
        lr_scheduler = CosineAnnealingLRScheduler(
            optimizer,
            max_steps=self.num_epoch,
            warmup_steps=self.warmup_epoch,
            min_lr=1e-7,
        )
        return optimizer, lr_scheduler

    def get_custom_training_args(self, training_args):
        # wVLA has no extra CLI hyperparameter beyond the shared defaults; everything
        # model-side is read from the config file. We do cache the action-expert type
        # here (config is already loaded by get_default_training_args) so the train/val
        # loss can be dispatched by expert type rather than by inspecting output tensors.
        self.expert_type = (
            self.config_dict["Policy"]["action_expert"].get("type", "regression").lower()
        )

    def get_policy(self, policy_config):
        return wVLAPolicy(policy_config)

    # ------------------------------------------------------------------ #
    # Language bootstrap (HuggingFace tokenizer + LM config)             #
    # ------------------------------------------------------------------ #
    def _policy_cfg(self, key, default=None):
        return self.config_dict["Policy"].get(key, default)

    def _action_chunk_size(self):
        """Action-chunk length, sourced solely from action_expert.action_chunk_size
        (its single source of truth; the action expert owns the chunk dimension)."""
        return self.config_dict["Policy"]["action_expert"]["action_chunk_size"]

    def _language_enabled(self):
        """Language is on iff the policy has a language_encoder block AND the dataset
        yields an 'instruction' channel. A mismatch is a config error, surfaced here."""
        has_encoder = self._policy_cfg("language_encoder") is not None
        has_instruction = (
            "instruction" in self.config_dict["Training_settings"]["data_keys"]
        )
        if has_encoder != has_instruction:
            raise ValueError(
                "Inconsistent language config: Policy.language_encoder is "
                f"{'set' if has_encoder else 'absent'} but data_keys "
                f"{'contains' if has_instruction else 'does not contain'} 'instruction'. "
                "Either provide both (language-conditioned) or neither (vision-only)."
            )
        return has_encoder

    def check_language_setup(self):
        """Prepare the HuggingFace tokenizer when the policy is language-conditioned.

        Sets self.use_language, self.tokenizer, self.text_max_len. The tokenizer id is
        taken from the language_encoder's backbone_name (so tokenizer and model share a
        vocabulary). Only the main process downloads/caches the tokenizer + LM config
        into the log dir; other ranks wait on the barrier, so eval is offline-capable.
        """
        self.use_language = self._language_enabled()
        self.tokenizer = None
        self.text_max_len = self._policy_cfg("text_max_len", 32)

        if not self.use_language:
            return

        from transformers import AutoConfig, AutoTokenizer

        backbone_name = self._policy_cfg("language_encoder")["backbone_name"]
        artifact_dir = os.path.join(
            self.config_and_code_save_path, LANGUAGE_ARTIFACT_DIRNAME
        )

        if is_main_process():
            os.makedirs(artifact_dir, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(backbone_name)
            # Cache tokenizer + LM config so eval can rebuild both offline from the log
            # dir (the LM weights are restored from the checkpoint, not re-downloaded).
            tokenizer.save_pretrained(artifact_dir)
            AutoConfig.from_pretrained(backbone_name).save_pretrained(artifact_dir)
            self.print_to_log_file(
                f"Cached language tokenizer + config ({backbone_name}) to {artifact_dir}"
            )
        self.sync_processes()
        # Every rank loads the tokenizer from the cached artifacts (main process wrote
        # them above), so dataloader workers tokenize with one consistent mapping.
        self.tokenizer = AutoTokenizer.from_pretrained(artifact_dir)


    # ------------------------------------------------------------------ #
    # Eval hooks: build & drive the policy from the log dir's artifacts   #
    # so a method-agnostic evaluator can run wVLA. See wrobo/sim_envs/Aloha/eval.py.
    # ------------------------------------------------------------------ #
    @classmethod
    def build_policy(cls, policy_config, log_dir=None, *, artifact_dir=None):
        """Construct the policy as training did, returning (policy, aux).

        Language artifacts (tokenizer + LM config) were cached under the log dir at train
        time, so eval rebuilds the language encoder from there offline and restores the LM
        weights from the checkpoint. aux carries the tokenizer for build_policy_inputs, or
        is None for a vision-only model.
        """
        policy_config = dict(policy_config)
        policy_config["camera_names"] = policy_config.get("camera_names") or []

        # The vision backbone weights come from the checkpoint, so build the architecture
        # only and avoid a DEFAULT-weights download at eval time.
        vision_cfg = dict(policy_config["vision_encoder"])
        vision_cfg["backbone_weights_name"] = None
        policy_config["vision_encoder"] = vision_cfg

        if artifact_dir is None:
            if log_dir is None:
                raise ValueError("Provide artifact_dir or a legacy log_dir to build the policy")
            artifact_dir = os.path.join(log_dir, "Config_and_Trainer")
        language_artifact_dir = os.path.join(artifact_dir, LANGUAGE_ARTIFACT_DIRNAME)
        use_language = (
            policy_config.get("language_encoder") is not None
            and os.path.isdir(language_artifact_dir)
        )

        aux = None
        if use_language:
            from transformers import AutoTokenizer

            # Point the language encoder at the cached local artifacts: build the
            # architecture only (weights come from the checkpoint), so eval never needs
            # network access. The original yaml block's other kwargs are preserved.
            language_cfg = dict(policy_config["language_encoder"])
            language_cfg["backbone_name"] = language_artifact_dir
            language_cfg["backbone_weights_name"] = None
            language_cfg["local_files_only"] = True
            policy_config["language_encoder"] = language_cfg

            aux = {
                "tokenizer": AutoTokenizer.from_pretrained(language_artifact_dir),
                "text_max_len": policy_config.get("text_max_len", 32),
            }

        return wVLAPolicy(policy_config), aux

    @staticmethod
    def build_policy_inputs(obs, aux, device):
        images = {k: v.to(device) for k, v in obs["image"].items()}

        inputs = {
            "image": images,
            "proprio_state": obs["state"].to(device),
        }
        if aux is not None and "instruction" in obs:
            tokenizer = aux["tokenizer"]
            encoded = tokenizer(
                obs["instruction"],
                max_length=aux["text_max_len"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs["lang_tokens"] = encoded["input_ids"].to(device)
            inputs["attention_mask"] = encoded["attention_mask"].to(device)
        return inputs

    def get_training_img_transforms(self):
        """Training image transforms. Accept Tensor with a shape of [C, H, W]."""
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_validation_img_transforms(self):
        """Validation image transforms. Accept Tensor with a shape of [C, H, W]."""
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_text_transform(self):
        """Build the text adapter handed to the dataset (None for vision-only).

        Text-side analog of get_*_img_transforms: tokenization runs in the dataloader
        workers and the dataset yields a model-ready ``(text_max_len,)`` LongTensor
        instead of a raw string."""
        if not self.use_language:
            return None
        return HFTextTransform(self.tokenizer, self.text_max_len)


    # ------------------------------------------------------------------ #
    # Train / val                                                        #
    # ------------------------------------------------------------------ #
    def prepare_inputs(self, batch):
        images, _, proprio_states, actions, is_pad, instruction = super().prepare_inputs(batch)
        lang_tokens = attention_mask = None
        if self.use_language:
            if instruction is None:
                raise KeyError("wVLA language-conditioned batches require 'instruction'")
            lang_tokens = instruction
            attention_mask = (lang_tokens != self.tokenizer.pad_token_id).long()
        return images, proprio_states, lang_tokens, attention_mask, actions, is_pad

    def _compute_loss(self, outputs, actions, is_pad):
        """Masked, padding-aware action loss.

        The criterion (self._loss_fn) and the target tensor (self._loss_target_key) are
        resolved once in initialize() from the configured expert type, so this hot path
        does no per-call dispatch:
          - "regression":            L1 between model_pred and the GT actions.
          - ddpm/ddim/flow_matching: MSE between model_pred and the expert's per-step
            ``target`` (noise / velocity).
        Padded action steps are masked out so the objective does not depend on how much
        the chunk was zero-padded (same masking as ACT / miniVLA).
        """
        target = actions if self._loss_target_key == "actions" else outputs["target"]
        per_elem = self._loss_fn(outputs["model_pred"], target)

        return self.masked_action_loss(per_elem, is_pad)

    def compute_training_loss(self, batch):
        images, proprio_states, lang_tokens, attention_mask, actions, is_pad = (
            self.prepare_inputs(batch)
        )

        with self.autocast_context():
            outputs = self.policy(
                image=images,
                proprio_state=proprio_states,
                lang_tokens=lang_tokens,
                attention_mask=attention_mask,
                actions=actions,
                is_pad=is_pad,
            )
            loss = self._compute_loss(outputs, actions, is_pad)

        return {"loss": LossTerm(loss)}
