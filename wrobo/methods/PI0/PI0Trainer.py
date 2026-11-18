import json
import os
import re
import torch

from functools import partial
from operator import itemgetter, methodcaller
from torch import optim
from torchvision import transforms

from wrobo.methods.PI0.models.policy import PI0Policy
from wrobo.training.trainer.ABCTrainer import DDPABCTrainer
from wrobo.training.trainer.loss import LossTerm
from wrobo.training.lr_scheduler.CosineAnnealingLR import CosineAnnealingLRScheduler
from wrobo.training.utils.lora import (
    DEFAULT_LORA_PARTS, LORA_PARTS, merged_lora_state_dict, normalize_lora_parts,
)
from wrobo.training.utils.mics import is_main_process
from wrobo.training.transforms.GammaTransform import GammaTransform


MODEL_ARTIFACT_DIRNAME = "pi0_config"
LORA_ARTIFACT_FILENAME = "pi0_lora_config.json"
LORA_CHECKPOINT_MARKER = "pi0_lora_adapter_checkpoint"
PROJECTION_NAMES = (
    "state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in",
    "action_time_mlp_out", "time_mlp_in", "time_mlp_out",
)


def _append_newline(text):
    text = str(text)
    return text if text.endswith("\n") else text + "\n"


class _PI0AdapterDeploymentPolicy(torch.nn.Module):
    """Expose adapter-only state dicts to wrobo's generic deployment loader."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    @property
    def config(self):
        return self.policy.get_base_model().config

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(self.policy)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        from peft import set_peft_model_state_dict

        return set_peft_model_state_dict(self.policy, state_dict)


class PI0Trainer(DDPABCTrainer):
    model_artifact_dirname = MODEL_ARTIFACT_DIRNAME
    lora_artifact_filename = LORA_ARTIFACT_FILENAME
    lora_checkpoint_marker = LORA_CHECKPOINT_MARKER

    def get_hyperparameters_name(self):
        if self.use_lora:
            return (
                f"lora_rank{self.lora_rank}_alpha{self.lora_alpha}_"
                f"dropout{self.lora_dropout}_{self.lora_save_mode}_"
                f"parts_{'-'.join(self.lora_parts)}"
            )
        return "default"

    def uses_pretrained_weights(self):
        policy = self.config_dict["Policy"]
        return (
            self.pretrained_weight is not None
            or policy.get("pretrained_load_mode", "none") != "none"
            or policy.get("pretrained_policy_path") is not None
        )

    @staticmethod
    def get_parameter_parts(policy):
        """Cover PI0/PI05 base weights and adapters exactly once by their owner."""
        if hasattr(policy, "peft_config"):
            policy = policy.get_base_model()
        flow = policy.model
        core = flow.paligemma_with_expert
        vlm = core.paligemma.model
        owners = {
            "vision": (vlm.vision_tower, vlm.multi_modal_projector),
            "language": (vlm.language_model,),
            "action": (core.gemma_expert,),
            "projection": tuple(getattr(flow, name) for name in PROJECTION_NAMES
                                if hasattr(flow, name)),
        }
        parts, assigned = {}, set()
        for part, modules in owners.items():
            parameters = {id(parameter): parameter
                          for module in modules for parameter in module.parameters()}
            if assigned.intersection(parameters):
                raise RuntimeError(f"PI parameters belong to multiple model parts: {part}")
            assigned.update(parameters)
            parts[part] = tuple(parameters.values())
        if assigned != {id(parameter) for parameter in policy.parameters()}:
            raise RuntimeError("PI model parts do not cover all policy parameters")
        return parts

    def log_model_parameters(self, network):
        if not is_main_process():
            return
        super().log_model_parameters(network)
        for part, parameters in self.get_parameter_parts(self.unwrap_network(network)).items():
            total = sum(parameter.numel() for parameter in parameters)
            trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
            percentage = trainable / total * 100 if total else 0.0
            self.print_to_log_file(
                f"  {part.title()}: {total:,} params "
                f"(trainable {trainable:,}, frozen {total - trainable:,}, {percentage:.2f}%)"
            )

    def build_training_policy(self):
        """Build the method policy, artifacts and optional LoRA adapters."""
        policy = self.get_policy(self.config_dict["Policy"])
        if self.use_lora and policy.pretrained_load_mode != "full":
            raise ValueError(
                "PI0 LoRA requires pretrained_load_mode='full': "
                "adapter-only weights cannot reconstruct a randomly initialized "
                "PI0 action expert."
            )
        self.tokenizer = policy.tokenizer
        artifact_dir = os.path.join(self.config_and_code_save_path, self.model_artifact_dirname)
        if is_main_process():
            os.makedirs(artifact_dir, exist_ok=True)
            policy.save_artifacts(artifact_dir)
        self.sync_processes()

        self.load_pretrained_policy_weights(policy)

        lora_config = {
            "enabled": self.use_lora,
            "rank": self.lora_rank,
            "alpha": self.lora_alpha,
            "dropout": self.lora_dropout,
            "parts": list(self.lora_parts),
            "save_mode": self.lora_save_mode,
            "pretrained_load_mode": policy.pretrained_load_mode,
            "pretrained_policy_path": (
                str(policy.pretrained_policy_path)
                if policy.pretrained_policy_path is not None
                else None
            ),
            "text_tokenizer_name": policy.config.text_tokenizer_name,
            "pretrained_revision": policy.pretrained_revision,
        }
        self.lora_config = lora_config
        if is_main_process():
            with open(
                os.path.join(
                    self.config_and_code_save_path,
                    self.lora_artifact_filename,
                ),
                "w",
            ) as file:
                json.dump(lora_config, file, indent=2)

        if self.use_lora:
            policy = self.wrap_policy_with_lora(policy, lora_config)
            if is_main_process():
                self.print_to_log_file(
                    "Enabled LoRA with "
                    f"rank={self.lora_rank}, alpha={self.lora_alpha}, "
                    f"dropout={self.lora_dropout}, parts={','.join(self.lora_parts)}"
                )
        self.sync_processes()

        if is_main_process():
            trainable = {name: list(param.shape) for name, param in policy.named_parameters() if param.requires_grad}
            with open(os.path.join(self.config_and_code_save_path, "trainable_parameters.json"), "w") as file:
                json.dump(trainable, file, indent=2)
            self.print_to_log_file("Trainable parameter tensors:", len(trainable))
        return policy

    def wrap_network(self, network):
        # Keep the conservative non-LoRA DDP setting for configurable frozen paths.
        # Vocabulary heads themselves have been removed from the model.
        if self.is_ddp:
            return torch.nn.parallel.DistributedDataParallel(
                network, device_ids=[self.device.index], output_device=self.device.index,
                find_unused_parameters=not self.use_lora,
            )
        return network

    def get_optimizers(self):
        """
        PI0 uses AdamW and wrobo epoch-based cosine decay + warmup.
        Official defaults: lr=2.5e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01.
        """
        optimizer = optim.AdamW(
            (parameter for parameter in self.policy.parameters() if parameter.requires_grad),
            lr=self.config_dict["Training_settings"]["base_lr"],
            weight_decay=self.config_dict["Training_settings"]["weight_decay"],
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        lr_scheduler = CosineAnnealingLRScheduler(
            optimizer,
            max_steps=self.num_epoch,
            warmup_steps=self.warmup_epoch,
            min_lr=self.config_dict["Training_settings"].get("min_lr", 1e-7),
        )
        return optimizer, lr_scheduler

    def initialize_loss(self):
        """The policy computes elementwise flow-matching MSE; no separate criterion is needed."""

    def get_custom_training_args(self, training_args):
        """Get PI0-specific training arguments"""
        self.use_lora = getattr(training_args, "use_lora", False)
        self.lora_rank = getattr(training_args, "lora_rank", 16)
        self.lora_alpha = getattr(training_args, "lora_alpha", 16)
        self.lora_dropout = getattr(training_args, "lora_dropout", 0.0)
        self.lora_save_mode = getattr(training_args, "lora_save_mode", "merged")
        self.lora_parts = normalize_lora_parts(
            getattr(training_args, "lora_parts", DEFAULT_LORA_PARTS)
        )
        if self.use_lora and self.pretrained_weight is not None:
            raise ValueError("For PI0 LoRA, set pretrained_policy_path to the base safetensors; use --continue_train for adapters")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.lora_save_mode not in {"merged", "adapter"}:
            raise ValueError("lora_save_mode must be one of: merged, adapter")

    @staticmethod
    def get_lora_target_modules(policy, parts):
        """Select action-dependent linear layers within the requested PI model parts."""
        parts = normalize_lora_parts(parts)
        prefix = r"model\.paligemma_with_expert\."
        patterns = {
            "vision": prefix + (
                r"paligemma\.model\.(vision_tower\..*\.self_attn\.(q|v)_proj|"
                r"multi_modal_projector\..*)"
            ),
            "language": prefix + r"paligemma\.model\.language_model\.layers\.(\d+)\.self_attn\.(q|k|v)_proj",
            "action": prefix + r"gemma_expert\.model\.layers\.\d+\.self_attn\.(q|v)_proj",
            "projection": rf"model\.({'|'.join(PROJECTION_NAMES)})",
        }
        if "language" in parts:
            last_expert_layer = policy.model.paligemma_with_expert.get_last_expert_layer_index()
        targets = {part: [] for part in parts}
        for name, module in policy.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            for part in parts:
                match = re.fullmatch(patterns[part], name)
                if match is None:
                    continue
                if part == "language":
                    layer_index, projection = int(match[1]), match[2]
                    allowed = {"k", "v"} if layer_index == last_expert_layer else {"q", "v"}
                    if layer_index > last_expert_layer or projection not in allowed:
                        continue
                targets[part].append(name)
        missing = [part for part, names in targets.items() if not names]
        if missing:
            raise ValueError(f"No LoRA target modules found for parts: {missing}")
        return targets

    @classmethod
    def wrap_policy_with_lora(cls, policy, lora_config):
        """Adapt selected parts, keeping all base weights and norms frozen."""
        from peft import LoraConfig, get_peft_model

        parts = normalize_lora_parts(lora_config.get("parts", DEFAULT_LORA_PARTS))
        grouped_targets = cls.get_lora_target_modules(policy, parts)
        config = LoraConfig(
            r=int(lora_config["rank"]),
            lora_alpha=float(lora_config["alpha"]),
            lora_dropout=float(lora_config["dropout"]),
            target_modules=[name for names in grouped_targets.values() for name in names],
            bias="none",
        )
        core = policy.model.paligemma_with_expert
        # Keep PEFT's default FP32 trainable adapters, even on a BF16 base.
        wrapped_policy = get_peft_model(policy, config, autocast_adapter_dtype=True)
        # PEFT freezes the base. These flags keep selected adapter dropout in train mode.
        if "vision" in parts:
            core.freeze_vision_encoder = False
        if "language" in parts:
            core.freeze_language_encoder = False
        wrapped_policy.train(policy.training)
        return wrapped_policy

    @staticmethod
    def _adapter_state_dict(policy):
        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(policy)

    @classmethod
    def _save_lora_deployment_weights(cls, policy, filename, save_mode):
        if save_mode == "adapter":
            torch.save(cls._adapter_state_dict(policy), filename)
            return
        if save_mode != "merged":
            raise ValueError(f"Unsupported LoRA save mode: {save_mode}")

        torch.save(merged_lora_state_dict(policy), filename)

    def save_checkpoint(self, filename: str, weights_only: bool = False) -> None:
        if not self.use_lora:
            super().save_checkpoint(filename, weights_only=weights_only)
            return
        if not is_main_process():
            return
        if self.disable_checkpointing:
            self.print_to_log_file(
                "Checkpoint saving is disabled; no file will be written."
            )
            return

        policy = self.unwrap_network(self.policy)
        if weights_only:
            self._save_lora_deployment_weights(
                policy, filename, self.lora_save_mode
            )
            return

        checkpoint = {
            self.lora_checkpoint_marker: True,
            "lora_config": self.lora_config,
            "network_weights": self._adapter_state_dict(policy),
            "training_batch_config": self.training_batch_config(),
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": (
                self.grad_scaler.state_dict() if self.grad_scaler else None
            ),
            "logging": (
                self.logger.get_checkpoint() if self.logger is not None else None
            ),
            "current_epoch": self.current_epoch,
            "LRScheduler_state": self.lr_scheduler.state_dict(),
            "best_ema": self._best_ema,
            "instruction_config": {
                "dataset_dir": self.dataset_dir,
                "instruction_mode": self.instruction_mode,
            },
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename_or_checkpoint):
        if isinstance(filename_or_checkpoint, (str, os.PathLike)):
            checkpoint = self.broadcast_checkpoint(
                lambda: torch.load(
                    filename_or_checkpoint,
                    map_location=self.device,
                    weights_only=False,
                )
            )
        else:
            checkpoint = filename_or_checkpoint

        is_adapter_resume = (
            isinstance(checkpoint, dict)
            and checkpoint.get(self.lora_checkpoint_marker, False)
        )
        is_adapter_weights = (
            isinstance(checkpoint, dict)
            and checkpoint
            and "network_weights" not in checkpoint
            and all(isinstance(key, str) and "lora_" in key for key in checkpoint)
        )
        if is_adapter_resume:
            self.validate_training_batch_config(checkpoint)

        if not (is_adapter_resume or is_adapter_weights):
            if isinstance(checkpoint, dict) and checkpoint.get("optimizer_state") is not None:
                if not self.was_initialized:
                    self.initialize()
                saved_groups = checkpoint["optimizer_state"]["param_groups"]
                current_groups = self.optimizer.state_dict()["param_groups"]
                if [len(g["params"]) for g in saved_groups] != [len(g["params"]) for g in current_groups]:
                    raise ValueError(
                        "PI0 optimizer parameter groups differ from this checkpoint (old checkpoints "
                        "may include removed vocabulary heads). Use --pretrained_weight to initialize "
                        "model weights with a new optimizer, rather than --continue_train."
                    )
            super().load_checkpoint(checkpoint)
            return

        if is_adapter_resume and "lora_config" in checkpoint:
            saved_config = checkpoint["lora_config"]
            if normalize_lora_parts(saved_config["parts"]) != self.lora_parts:
                raise ValueError("Resume LoRA parts differ from checkpoint")
            for name, current in (("rank", self.lora_rank), ("alpha", self.lora_alpha), ("dropout", self.lora_dropout)):
                if saved_config[name] != current:
                    raise ValueError(f"Resume LoRA {name} differs from checkpoint")
            if self.was_initialized:
                for name in ("pretrained_policy_path", "pretrained_revision"):
                    if saved_config.get(name) != self.lora_config.get(name):
                        raise ValueError(f"Resume base {name} differs from initialized policy")
            else:
                self.config_dict["Policy"]["pretrained_load_mode"] = saved_config["pretrained_load_mode"]
                self.config_dict["Policy"]["pretrained_policy_path"] = saved_config["pretrained_policy_path"]
                self.config_dict["Policy"]["pretrained_revision"] = saved_config.get("pretrained_revision")
        if not self.was_initialized:
            self.initialize()
        if not self.use_lora:
            raise RuntimeError("Cannot load a LoRA adapter checkpoint with LoRA disabled")

        from peft import set_peft_model_state_dict

        policy = self.unwrap_network(self.policy)
        adapter_weights = (
            checkpoint["network_weights"] if is_adapter_resume else checkpoint
        )
        expected_weights = self._adapter_state_dict(policy)
        missing = set(expected_weights).difference(adapter_weights)
        unexpected = set(adapter_weights).difference(expected_weights)
        wrong_shapes = [
            name for name in set(expected_weights).intersection(adapter_weights)
            if expected_weights[name].shape != adapter_weights[name].shape
        ]
        if missing or unexpected or wrong_shapes:
            raise RuntimeError(
                "LoRA checkpoint does not match the selected parts/rank: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
                f"incompatible_shapes={sorted(wrong_shapes)}"
            )
        set_peft_model_state_dict(policy, adapter_weights)
        if not is_adapter_resume:
            if is_main_process():
                self.print_to_log_file(
                    "Loaded LoRA adapter weights (optimizer state not restored)"
                )
            return

        self.current_epoch = checkpoint["current_epoch"] + 1
        if is_main_process() and self.logger is not None:
            self.logger.load_checkpoint(checkpoint["logging"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            scaler_state = checkpoint.get("grad_scaler_state")
            if scaler_state is not None:
                self.grad_scaler.load_state_dict(scaler_state)
        self.lr_scheduler.load_state_dict(checkpoint["LRScheduler_state"])
        self._best_ema = checkpoint.get("best_ema")
        if is_main_process():
            self.print_to_log_file(
                f"Resumed LoRA training from epoch {self.current_epoch}"
            )

    def get_default_training_args(self, training_args):
        super().get_default_training_args(training_args)
        policy_config = self.config_dict["Policy"]
        policy_config.setdefault("action_chunk_size", 50)
        from wrobo.methods.PI0.models.configuration import PI0Config
        PI0Config.from_policy_dict(policy_config)

    @staticmethod
    def get_policy(policy_config):
        return PI0Policy(policy_config)

    # ------------------------------------------------------------------ #
    # Eval hooks for method-agnostic evaluator                           #
    # ------------------------------------------------------------------ #
    @classmethod
    def build_policy(cls, policy_config, log_dir=None, *, artifact_dir=None):
        """
        Construct the policy exactly as training does.

        PI0 injects camera_names at train time, so re-inject from config.
        """
        # Mutate the evaluator-owned dictionary as well: its generic rollout code reads
        # history_width after this hook returns.
        policy_config.setdefault("history_width", 1)
        policy_config.setdefault("action_chunk_size", 50)
        policy_config["camera_names"] = policy_config.get("camera_names") or []
        if artifact_dir is None:
            if log_dir is None:
                raise ValueError("Provide artifact_dir or a legacy log_dir to build the policy")
            artifact_dir = os.path.join(log_dir, "Config_and_Trainer")
        lora_config_path = os.path.join(artifact_dir, cls.lora_artifact_filename)
        lora_config = None
        if os.path.isfile(lora_config_path):
            with open(lora_config_path) as file:
                lora_config = json.load(file)

        model_artifact_dir = os.path.join(artifact_dir, cls.model_artifact_dirname)
        save_mode = lora_config.get("save_mode") if lora_config else None
        if lora_config and lora_config.get("enabled", True) and save_mode == "adapter":
            base_mode = lora_config.get("pretrained_load_mode")
            base_path = lora_config.get("pretrained_policy_path")
            if base_mode != "full" or not base_path:
                raise RuntimeError(
                    "Adapter-only PI0 weights require the original full pretrained "
                    "policy path recorded in pi0_lora_config.json."
                )
            policy_config["pretrained_load_mode"] = "full"
            policy_config["pretrained_policy_path"] = base_path
            policy_config["pretrained_revision"] = lora_config.get("pretrained_revision")
        elif os.path.isdir(model_artifact_dir):
            policy_config["text_tokenizer_name"] = model_artifact_dir
            policy_config["pretrained_policy_path"] = None
            policy_config["pretrained_load_mode"] = "none"

        if os.path.isdir(model_artifact_dir):
            policy_config["text_tokenizer_name"] = model_artifact_dir
        policy = cls.get_policy(policy_config)
        if lora_config and lora_config.get("enabled", True):
            if save_mode == "merged":
                return policy, None
            policy = cls.wrap_policy_with_lora(policy, lora_config)
            if save_mode == "adapter":
                policy = _PI0AdapterDeploymentPolicy(policy)
        return policy, None

    def train_end(self):
        if not self.use_lora:
            super().train_end()
            return
        if is_main_process() and not self.already_finish_training and not self.disable_checkpointing:
            policy = self.unwrap_network(self.policy)
            self.save_checkpoint(os.path.join(self.logs_output_folder, "checkpoint_final_retrainable.pth"))
            self.save_checkpoint(os.path.join(self.logs_output_folder, "checkpoint_final.pth"), weights_only=True)
            best_path = os.path.join(self.logs_output_folder, "checkpoint_best_retrainable.pth")
            if os.path.isfile(best_path):
                from peft import set_peft_model_state_dict

                best = torch.load(best_path, map_location=self.device, weights_only=False)
                final_adapter = {
                    key: value.clone() for key, value in self._adapter_state_dict(policy).items()
                }
                try:
                    set_peft_model_state_dict(policy, best["network_weights"])
                    self._save_lora_deployment_weights(
                        policy, os.path.join(self.logs_output_folder, "checkpoint_best.pth"),
                        self.lora_save_mode,
                    )
                finally:
                    set_peft_model_state_dict(policy, final_adapter)
            # Keep the resume file until both final and best exports have succeeded.
            latest = os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
            if os.path.isfile(latest):
                os.remove(latest)
            self.print_to_log_file("Saved final/best LoRA deployment weights:", self.lora_save_mode)
        previous = self.already_finish_training
        try:
            # Reuse only common worker/logger cleanup, not base-class export logic.
            self.already_finish_training = True
            super().train_end()
        finally:
            self.already_finish_training = previous

    @staticmethod
    def build_policy_inputs(obs, aux, device):
        """Move dataset/evaluator-preprocessed inputs to the policy device."""
        images = {key: value.to(device) for key, value in obs["image"].items()}
        camera_names = list(obs["image"].keys())

        inputs = {
            "image": images,
            "camera_names": camera_names,
            "proprio_state": obs["state"].to(device),
        }
        if "instruction" in obs:
            inputs["language"] = obs["instruction"]
        return inputs

    def get_training_img_transforms(self):
        """Convert dataset uint8 frames to the [0, 1] range expected by the policy."""
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomApply([GammaTransform(gamma_range=(0.7, 1.3))], p=0.5),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.2
                ),
                transforms.ToTensor(),
            ]
        )

    def get_validation_img_transforms(self):
        """Convert validation uint8 frames to the [0, 1] range expected by the policy."""
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    def get_text_transform(self):
        """Tokenize one instruction using the method tokenizer."""
        data_keys = self.config_dict["Training_settings"]["data_keys"]
        text_transform = None
        if "instruction" in data_keys:
            text_transform = transforms.Compose(
                [
                    transforms.Lambda(_append_newline),
                    transforms.Lambda(
                        partial(
                            self.tokenizer,
                            max_length=self.config_dict["Policy"].get(
                                "tokenizer_max_length", 48
                            ),
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                        )
                    ),
                    transforms.Lambda(itemgetter("input_ids")),
                    transforms.Lambda(methodcaller("squeeze", 0)),
                    transforms.Lambda(methodcaller("long")),
                ]
            )
        return text_transform

    def _compute_loss(self, outputs, is_pad):
        """Compute the padding-aware Flow Matching velocity loss."""
        return self.masked_action_loss(outputs["loss_per_element"], is_pad)

    def compute_training_loss(self, batch):
        images, camera_names, proprio_states, actions, is_pad, instructions = self.prepare_inputs(batch)

        with self.autocast_context():
            outputs = self.policy(
                image=images,
                camera_names=camera_names,
                proprio_state=proprio_states,
                language=instructions,
                actions=actions,
            )
            loss = self._compute_loss(outputs, is_pad)

        return {"loss": LossTerm(loss)}
