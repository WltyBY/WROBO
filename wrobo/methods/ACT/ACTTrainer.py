import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms

from wrobo.methods.ACT.models.dimensions import resolve_act_dimensions
from wrobo.methods.ACT.models.policy import ACTPolicy
from wrobo.training.lr_scheduler.ConstantLR import ConstantLRScheduler
from wrobo.training.trainer.ABCTrainer import DDPABCTrainer
from wrobo.training.trainer.loss import LossTerm
from wrobo.training.transforms.GammaTransform import GammaTransform


class ACTTrainer(DDPABCTrainer):
    def get_hyperparameters_name(self):
        return f"w_KL_{self.w_KL}"

    def _validate_norm_stats(self):
        """Check the configured numerical fields against ACT state/action dimensions."""
        if not isinstance(self.norm_stats, dict):
            raise ValueError("ACT norm_stats.json must contain a dictionary")
        policy = self.config_dict["Policy"]
        proprio_dim, action_dim = resolve_act_dimensions(
            policy.get("proprio_dim"), policy.get("action_dim")
        )
        directories = self.dataset_dir if isinstance(self.dataset_dir, list) else [self.dataset_dir]
        per_dataset = any(directory in self.norm_stats for directory in directories)
        if per_dataset:
            missing_directories = [directory for directory in directories if directory not in self.norm_stats]
            if missing_directories:
                raise ValueError(f"ACT norm_stats.json is missing dataset groups: {missing_directories}")
            groups = [(directory, self.norm_stats[directory]) for directory in directories]
        else:
            groups = [("merged", self.norm_stats)]
        for group_name, stats in groups:
            if not isinstance(stats, dict):
                raise ValueError(f"ACT normalization group {group_name!r} must be a dictionary")
            missing = [key for key in (self.proprio_key, self.action_key) if key not in stats]
            if missing:
                raise ValueError(
                    f"ACT normalization group {group_name!r} is missing selected fields {missing}. "
                    "Recompute norm_stats.json for the current data_keys or use a new log directory."
                )
            for key, width, dimension_name in (
                (self.proprio_key, proprio_dim, "proprio_dim"),
                (self.action_key, action_dim, "action_dim"),
            ):
                entry = stats[key]
                if not isinstance(entry, dict) or not {"mean", "std"} <= set(entry):
                    raise ValueError(f"ACT normalization stats for {key!r} need mean and std")
                try:
                    mean = np.asarray(entry["mean"], dtype=np.float64)
                    std = np.asarray(entry["std"], dtype=np.float64)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"ACT normalization stats for {key!r} must be numeric") from error
                if mean.ndim != 1 or std.shape != mean.shape:
                    raise ValueError(f"ACT normalization stats for {key!r} must be matching 1-D mean/std vectors")
                if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
                    raise ValueError(f"ACT normalization stats for {key!r} must be finite with positive std")
                if mean.shape != (width,):
                    raise ValueError(
                        f"ACT normalization stats for selected field {key!r} have width {mean.size}, "
                        f"but Policy.{dimension_name}={width} requires width {width}. "
                        "Check the selected data key, configuration and normalization statistics."
                    )

    def initialize_loss(self):
        self.L1_loss = nn.L1Loss(reduction="none")
        if self.do_deep_supervision:
            self.L1_loss = self._build_deep_supervision_loss_object(
                self.L1_loss,
                self.config_dict["Policy"].get("num_decoder_layers", 1),
            )

    def get_optimizers(self):
        optimizer = optim.AdamW(
            (parameter for parameter in self.policy.parameters() if parameter.requires_grad),
            lr=self.config_dict["Training_settings"]["base_lr"],
            weight_decay=self.config_dict["Training_settings"]["weight_decay"],
        )
        lr_scheduler = ConstantLRScheduler(optimizer, warmup_steps=self.warmup_epoch)
        return optimizer, lr_scheduler

    def get_custom_training_args(self, training_args):
        self.w_KL = training_args.w_KL

    @staticmethod
    def get_policy(policy_config):
        return ACTPolicy(policy_config)

    # ------------------------------------------------------------------ #
    # Eval hooks: let a method-agnostic evaluator build & drive this      #
    # method's policy (model is reconstructed by whatever method trained  #
    # it). See wrobo/sim_envs/Aloha/eval.py.                              #
    # ------------------------------------------------------------------ #
    @classmethod
    def build_policy(cls, policy_config, log_dir=None, *, artifact_dir=None):
        """Construct the policy exactly as training does, and return (policy, aux).

        ACT injects camera_names into the config at train time (after the config was
        copied), so re-inject from data_keys here. ACT has no language, so aux is None.
        """
        # camera_names are derived from the image observation keys, same rule as the
        # dataset/cam_keys, so the per-view embeddings are rebuilt and loaded correctly.
        policy_config = dict(policy_config)
        policy_config["camera_names"] = policy_config.get("camera_names") or []
        # Evaluation loads the complete trained checkpoint, so backbone weights
        # need not be fetched again. Preserve their name for the BN architecture.
        policy_config.setdefault("backbone_load_pretrained", False)
        return cls.get_policy(policy_config), None

    @staticmethod
    def build_policy_inputs(obs, aux, device):
        images = {k: v.to(device) for k, v in obs["image"].items()}
        camera_names = list(obs["image"].keys())

        return {
            "image": images,
            "camera_names": camera_names,
            "proprio_state": obs["state"].to(device),
        }

    def get_training_img_transforms(self):
        """
        Get training transforms for images.

        Accept Tensor with a shape of [C, H, W]
        """
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomApply([GammaTransform(gamma_range=(0.7, 1.3))], p=0.5),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_validation_img_transforms(self):
        """
        Get validation transforms for images.

        Accept Tensor with a shape of [C, H, W]
        """
        return self.get_eval_img_transform()

    @staticmethod
    def get_eval_img_transform():
        """Return the deterministic image transform shared by validation/deployment."""
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def prepare_inputs(self, batch):
        return super().prepare_inputs(batch)[:5]

    def compute_training_loss(self, batch):
        images, camera_names, proprio_states, actions, is_pad = self.prepare_inputs(batch)

        with self.autocast_context():
            outputs = self.policy(
                image=images,
                camera_names=camera_names,
                proprio_state=proprio_states,
                actions=actions,
                is_pad=is_pad,
            )

            if self.do_deep_supervision:
                actions = [actions] * len(outputs["action_chunk"])

            # Compute Loss
            l1_per_elem = self.L1_loss(outputs["action_chunk"], actions)
            l1_ = self.masked_action_loss(l1_per_elem, is_pad)

            mu: torch.Tensor = outputs["mu"]
            logvar: torch.Tensor = outputs["logvar"]
            # mean over batch, sum over dimensions
            kl_ = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean(dim=0)

        return {
            "L1_loss": LossTerm(l1_),
            "KL_loss": LossTerm(kl_, normalization="samples", weight=self.w_KL),
        }
