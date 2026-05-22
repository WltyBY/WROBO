import os
import torch
import warnings

from datetime import datetime
from torch import GradScaler, optim, nn, autocast
from torchvision import transforms
from torch.utils.data import DistributedSampler, RandomSampler, DataLoader

from wrobo.methods.ACT.models.policy import ACTPolicy
from wrobo.training.ABCTrainer import DDPABCTrainer
from wrobo.utils.file_operations import copy_file_to_dstFolder, open_json
from wrobo.utils.load_model_weights import load_pretrained_weights
from wrobo.training.lr_scheduler.ConstantLR import ConstantLRScheduler
from wrobo.training.dataset.episodic_dataset import EpisodicDataset
from wrobo.training.dataloader.sampler import InfiniteSampler
from wrobo.training.dataloader.collater import BaseCollater
from wrobo.utils.others import dummy_context


class ACTTrainer(DDPABCTrainer):
    def __init__(
        self,
        training_args,
        verbose: bool = True,
    ):
        self.verbose = verbose

        self.save_every = 5
        self.disable_checkpointing = False

        self.get_default_training_args(training_args)
        self.get_custom_training_args(training_args)

        self.device = self.get_device()

        # Task-general params
        task_general_names = "BS_{}_GPU_NUM_{}_EPOCH_{}_SEED_{}_PRETRAINED_{}".format(
            self.batch_size,
            self.world_size,
            self.num_epoch,
            self.random_seed,
            self.pretrained_weight is not None,
        )

        # hyperparameter
        hyperparams_name = f"w_KL_{self.w_KL}"

        timestamp = datetime.now()
        time_ = "Train_Log_%d_%d_%d_%02.0d_%02.0d_%02.0d" % (
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            timestamp.second,
        )
        assert self.method_name is not None
        self.logs_output_folder = os.path.join(
            "./Logs",
            self.method_name,
            task_general_names,
            hyperparams_name,
            "fold_" + self.fold,
        )
        if not os.path.exists(self.logs_output_folder):
            os.makedirs(self.logs_output_folder, exist_ok=True)

        self.log_file = os.path.join(self.logs_output_folder, time_ + ".txt")
        with open(self.log_file, "w"):
            pass

        self.print_to_log_file(
            f"Using device: {self.device} | DDP: {self.is_ddp} | rank {self.rank}/{self.world_size}"
        )

        config_and_code_save_path = os.path.join(
            self.logs_output_folder, "Config_and_Trainer"
        )
        script_path = os.path.abspath(__file__)

        self.check_dataset_split()
        self.check_norm_stats()

        if self.is_main_process():
            os.makedirs(config_and_code_save_path, exist_ok=True)

            # copy the config file to the logs folder
            copy_file_to_dstFolder(self.config_file, config_and_code_save_path)

            # copy the trainer file to the logs folder
            copy_file_to_dstFolder(script_path, config_and_code_save_path)

            # copy dataset_split file to the logs folder
            copy_file_to_dstFolder(
                os.path.join(self.dataset_dir, "dataset_split.json"),
                config_and_code_save_path,
            )

            self.print_to_log_file(
                "Training logs will be saved in:", self.logs_output_folder
            )
        
        if torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            self.grad_scaler = None   # BF16 don't need grad scaler
            if self.is_main_process():
                self.print_to_log_file("Using BF16 precision for training.")
        else:
            self.amp_dtype = torch.float16
            self.grad_scaler = GradScaler() if self.device.type == "cuda" else None
            if self.is_main_process():
                self.print_to_log_file("Using FP16 precision for training with GradScaler.")

        self.logger = self.get_logger()
        self._best_ema = None
        self.ema_decay = getattr(self, "ema_decay", 0.9)
        self.norm_stats = open_json(
            os.path.join(self.logs_output_folder, "norm_stats.json")
        )
        self.was_initialized = False
        self.current_epoch = 0

        if self.continue_train:
            checkpoint = os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
            self.sync_processes()
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(
                    f"Continue training was requested but checkpoint not found: {checkpoint}"
                )
            self.load_checkpoint(checkpoint)
        self.sync_processes()

    def initialize(self):
        """
        Initialize networks, optimizers, schedulers, loss, etc.
        self.networks
        self.optimizers
        self.schedulers
        """
        if not self.was_initialized:
            self.init_random()
            self.setting_check()

            # build network
            self.network = self.get_policy(self.config_dict["Policy"])
            if self.pretrained_weight is not None:
                if self.is_main_process():
                    self.print_to_log_file(
                        f"Loading pretrained weight from {self.pretrained_weight}"
                    )
                load_pretrained_weights(self.network, self.pretrained_weight)

            self.network.to(self.device)

            if self.is_ddp:
                self.network = self.convert_bn2syncbn(self.network)
                self.network = torch.nn.parallel.DistributedDataParallel(
                    self.network,
                    device_ids=[self.device.index],
                    output_device=self.device.index,
                )

            if self.do_compile:
                if self.is_main_process():
                    self.print_to_log_file("Compiling network...")
                    warnings.warn(
                        "⚠️ Find you use --do_compile during training, which can significantly speed up the training."
                        "However, I found NaN during training while debugging with torch.compile, which may be due to some unstable operators in the model. "
                        "If you also encounter NaN during training, consider don't use --do_compile to False or try to find out which operator causes the instability and avoid using it. ",
                        RuntimeWarning,
                    )
                self.network = torch.compile(self.network)
                torch.compiler.reset()

            self.optimizer, self.lr_scheduler = self.get_optimizers()

            # initialize loss
            self.L1_loss = nn.L1Loss(reduction="none")

            if self.do_deep_supervision:
                self.L1_loss = self._build_deep_supervision_loss_object(
                    self.L1_loss,
                    self.config_dict["Policy"].get("num_decoder_layers", 1),
                )

            self.was_initialized = True
        else:
            raise RuntimeError(
                "self.initialize() should only be called once. "
                "Or initialization was done before initialize method???"
            )

    def get_optimizers(self):
        optimizer = optim.AdamW(
            self.network.parameters(),
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

    def get_training_img_transforms(self):
        """
        Get training transforms for images.

        Accept Tensor with a shape of [C, H, W]
        """
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(256),
                # transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomApply(
                #     transforms.ColorJitter(
                #         brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                #     ),
                #     p=0.8,
                # ),
                # transforms.RandomGrayscale(p=0.2),
                # transforms.RandomApply(
                #     [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3
                # ),
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
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(256),
                # transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_train_and_val_dataset(self):
        """
        Get training and validation dataset.
        """
        train_dataset = EpisodicDataset(
            dataset_dir=self.dataset_dir,
            split="train",
            fold=self.fold,
            data_keys=self.config_dict["Training_settings"]["data_keys"],
            history_width=self.config_dict["Policy"]["history_width"],
            action_chunk_size=self.config_dict["Policy"]["action_chunk_size"],
            norm_stats=self.norm_stats,
            img_transforms=self.get_training_img_transforms(),
            biased_sample=self.config_dict["Training_settings"]["biased_sample"],
        )
        val_dataset = EpisodicDataset(
            dataset_dir=self.dataset_dir,
            split="val",
            fold=self.fold,
            data_keys=self.config_dict["Training_settings"]["data_keys"],
            history_width=self.config_dict["Policy"]["history_width"],
            action_chunk_size=self.config_dict["Policy"]["action_chunk_size"],
            norm_stats=self.norm_stats,
            img_transforms=self.get_validation_img_transforms(),
            biased_sample=self.config_dict["Training_settings"]["biased_sample"],
        )
        return train_dataset, val_dataset

    def build_sampler(self, dataset):
        """
        If DDP is used, make sure to use DistributedSampler to partition the dataset.
        """
        if self.is_ddp:
            base_sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.random_seed,
                drop_last=True,
            )
        else:
            base_sampler = RandomSampler(dataset)
        return InfiniteSampler(base_sampler)

    def get_train_and_val_dataloader(self):
        """
        Get training and validation dataloader.
        """
        train_dataset, val_dataset = self.get_train_and_val_dataset()

        train_sampler = self.build_sampler(train_dataset)
        # shuffle is True here, because we expect patch based validation to be more comprehensive.
        # If the number of validation iteration is smaller than the valset, the validation
        # throughout the entire training process cannot cover all the images in the val set.
        val_sampler = self.build_sampler(val_dataset)

        this_num_workers = max(1, self.num_workers // self.world_size)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=this_num_workers,
            collate_fn=BaseCollater(),
            pin_memory=self.device.type == "cuda",
            persistent_workers=True,
            worker_init_fn=self.worker_init_fn,
            drop_last=True,
            prefetch_factor=max(1, 3 - (self.batch_size // 64)),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=max(1, this_num_workers // 2),
            collate_fn=BaseCollater(),
            pin_memory=self.device.type == "cuda",
            persistent_workers=True,
            drop_last=True,
            prefetch_factor=max(1, 2 - (self.batch_size // 64)),
        )

        return train_loader, val_loader

    def train_step(self, batch):
        images = batch["image"]
        proprio_states = batch["proprio_state"]
        actions = batch["action_abs"]
        is_pad = batch["is_pad"]

        # to device
        if isinstance(images, list):
            images = [img.to(self.device, non_blocking=True) for img in images]
        else:
            images = images.to(self.device, non_blocking=True)
        proprio_states = proprio_states.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        is_pad = is_pad.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with (
            autocast(self.device.type, dtype=self.amp_dtype, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            outputs = self.network(
                image=images,
                proprio_state=proprio_states,
                actions=actions,
                is_pad=is_pad,
            )

            if self.do_deep_supervision:
                actions = [actions] * len(outputs["action_chunk"])

            # Compute Loss
            l1_ = (
                self.L1_loss(outputs["action_chunk"], actions) * ~is_pad.unsqueeze(-1)
            ).mean()

            mu: torch.Tensor = outputs["mu"]
            logvar: torch.Tensor = outputs["logvar"]
            # mean over batch, sum over dimensions
            kl_ = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean(dim=0)

            # assert not (torch.isnan(l1_) or torch.isnan(kl_)), f"L1: {l1_}. KL: {kl_}"
            l = l1_ + self.w_KL * kl_

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
                    
        return {
            "loss": l.detach().cpu().numpy(),
            "L1_loss": l1_.detach().cpu().numpy(),
            "KL_loss": kl_.detach().cpu().numpy(),
        }

    @torch.inference_mode()
    def validation_step(self, batch):
        images = batch["image"]
        proprio_states = batch["proprio_state"]
        actions = batch["action_abs"]
        is_pad = batch["is_pad"]

        # to device
        if isinstance(images, list):
            images = [img.to(self.device, non_blocking=True) for img in images]
        else:
            images = images.to(self.device, non_blocking=True)
        proprio_states = proprio_states.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        is_pad = is_pad.to(self.device, non_blocking=True)

        with (
            autocast(self.device.type, dtype=self.amp_dtype, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            outputs = self.network(
                image=images,
                proprio_state=proprio_states,
                actions=actions,
                is_pad=is_pad,
            )

            if self.do_deep_supervision:
                actions = [actions] * len(outputs["action_chunk"])

            # Compute Loss
            l1_ = (
                self.L1_loss(outputs["action_chunk"], actions) * ~is_pad.unsqueeze(-1)
            ).mean()

            mu: torch.Tensor = outputs["mu"]
            logvar: torch.Tensor = outputs["logvar"]
            # mean over batch, sum over dimensions
            kl_ = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean(dim=0)

            l = l1_ + self.w_KL * kl_

        return {
            "loss": l.detach().cpu().numpy(),
            "L1_loss": l1_.detach().cpu().numpy(),
            "KL_loss": kl_.detach().cpu().numpy(),
        }
