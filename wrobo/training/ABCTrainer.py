import os
import torch
import random
import warnings

import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from time import time, sleep
from datetime import datetime
from torch import GradScaler
from torch._dynamo import OptimizedModule
from abc import ABC, abstractmethod
from torch.utils.data import DistributedSampler, RandomSampler

from wrobo.utils.dataset_split import dataset_split, verify_kfold_splits
from wrobo.utils.file_operations import (
    open_yaml,
    copy_file_to_dstFolder,
    open_json,
    save_json,
)
from wrobo.utils.dataset_statistic import get_norm_stats
from wrobo.utils.json_export import recursive_fix_for_json_export
from wrobo.utils.load_model_weights import load_pretrained_weights
from wrobo.utils.others import empty_cache
from wrobo.utils.collate_outputs import collate_outputs
from wrobo.training.utils.logger import TensorBoardLogger
from wrobo.training.dataloader.sampler import InfiniteSampler
from wrobo.training.loss.deep_supervision import DeepSupervisionWeightedSummator


class DDPABCTrainer(ABC):
    """
    You should at least implement the following abstract methods in the corresponding Trainer:
    - __init__(self, training_args, verbose=False)
    - initialize(self)
    - get_optimizers(self)
    - get_custom_training_args(self)
    - get_networks(self, network_settings: dict) -> nn.Module
    - get_training_img_transforms(self)
    - get_validation_img_transforms(self)
    - get_train_and_val_dataset(self)
    - build_sampler(self, dataset)
    - get_train_and_val_dataloader(self)
    - train_step(self, batch)
    - validation_step(self, batch)
    """

    def __init__(self, training_args, verbose=True):
        """
        Don't call this constructor directly by super.init().
        Just copy and rewrite the code in __init__() in the corresponding Trainer.
        """
        self.verbose = verbose

        self.was_initialized = False
        self.current_epoch = 0
        self.save_every = 1
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
        hyperparams_name = (
            "WRITE_THE_HYPERPARAMETERS_OF_THE_METHOD_LIKE_task_general_names"
        )

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

        if self.continue_train:
            checkpoint = os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
            self.sync_processes()
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(
                    f"Continue training was requested but checkpoint not found: {checkpoint}"
                )
            self.load_checkpoint(checkpoint)
        self.sync_processes()

    @abstractmethod
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
            self.network = self.get_networks(self.config_dict["Policy"])
            if self.pretrained_weight is not None:
                if self.is_main_process():
                    self.print_to_log_file(
                        f"Loading pretrained weight from {self.pretrained_weight}"
                    )
                load_pretrained_weights(self.network, self.pretrained_weight)

            self.network.to(self.device)

            if self.is_ddp:
                self.network = self.convert_bn2syncbn(self.network)
                self.network = nn.parallel.DistributedDataParallel(
                    self.network,
                    device_ids=[self.device.index],
                    output_device=self.device.index,
                )

            if self.do_compile:
                if self.is_main_process():
                    self.print_to_log_file("Compiling network...")
                self.network = torch.compile(self.network)
                torch.compiler.reset()

            self.optimizer, self.lr_scheduler = self.get_optimizers()

            # initialize loss
            self.loss = None

            if self.do_deep_supervision:
                self.loss = self._build_deep_supervision_loss_object(
                    self.loss,
                    self.config_dict["Policy"].get("num_decoder_layers", 1),
                )

            self.was_initialized = True
        else:
            raise RuntimeError(
                "self.initialize() should only be called once. "
                "Or initialization was done before initialize method???"
            )

    @abstractmethod
    def get_optimizers(self):
        """
        Return optimizers and lr schedulers.
        """
        pass

    @abstractmethod
    def get_custom_training_args(self, training_args):
        """
        Get custom training args in train.py in the corresponding method.
        training_args
        """
        pass

    @abstractmethod
    def get_policy(self, policy_config: dict) -> nn.Module:
        pass

    @abstractmethod
    def get_training_img_transforms(self):
        """
        Get training transforms for images.

        Accept Tensor with a shape of [C, H, W]
        """
        pass

    @abstractmethod
    def get_validation_img_transforms(self):
        """
        Get validation transforms for images.

        Accept Tensor with a shape of [C, H, W]
        """
        pass

    @abstractmethod
    def get_train_and_val_dataset(self):
        """
        Get training and validation dataset.
        """
        pass

    @abstractmethod
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

    @abstractmethod
    def get_train_and_val_dataloader(self):
        """
        Get training and validation dataloader.
        """
        pass

    @abstractmethod
    def train_step(self, batch):
        """
        Run one training step and return loss(Dict).
        """
        pass

    @abstractmethod
    def validation_step(self, batch):
        """
        Run one validation step and return loss(Dict).
        """
        pass

    def check_dataset_split(self):
        if self.is_main_process():
            splits_path = os.path.join(self.dataset_dir, "dataset_split.json")
            if not os.path.exists(splits_path):
                self.print_to_log_file(
                    f"Dataset split not found under {self.dataset_dir}. Creating new splits..."
                )

                case_lst = [
                    f.replace(".hdf5", "")
                    for f in os.listdir(self.dataset_dir)
                    if os.path.isfile(os.path.join(self.dataset_dir, f))
                    and f.endswith(".hdf5")
                ]
                # As model usually needs to be test in simulated or real environment, we use K-Fold cross-validation to split the dataset.
                # You can just use one fold for training or set --fold to 'all' to use all data for training.
                splits, is_same = dataset_split(
                    case_lst=case_lst, num_fold=5, random_seed=self.random_seed
                )
                verify_kfold_splits(
                    splits, case_lst
                )  # check if the splits are correct and non-overlapping
                splits["all_splits_same"] = is_same
                save_json(splits, splits_path)
            self.print_to_log_file(f"Using splits from {splits_path}")
        self.sync_processes()

    def check_norm_stats(self):
        """
        Get normalization stats (e.g. mean and std) for the dataset, which can be used in data normalization.
        You can compute the stats from the dataset or just set them to some fixed values.
        """
        if self.is_main_process():
            norm_stats_path = os.path.join(self.logs_output_folder, "norm_stats.json")
            if not os.path.exists(norm_stats_path):
                self.print_to_log_file(
                    f"Norm stats not found under {self.logs_output_folder}. Creating new stats..."
                )
                dataset_split = open_json(
                    os.path.join(self.dataset_dir, "dataset_split.json")
                )
                # get cases for training based on the split for training
                if self.fold == "all":
                    case_for_training = (
                        dataset_split["0"]["train"] + dataset_split["0"]["val"]
                    )
                else:
                    case_for_training = dataset_split[self.fold]["train"]

                norm_stats = get_norm_stats(
                    self.dataset_dir, case_lst=case_for_training, seed=self.random_seed
                )
                recursive_fix_for_json_export(norm_stats)
                save_json(norm_stats, norm_stats_path)
            self.print_to_log_file(f"Using norm stats from {norm_stats_path}")
        self.sync_processes()

    def is_main_process(self):
        return (not self.is_ddp) or self.rank == 0

    def sync_processes(self):
        if self.is_ddp:
            dist.barrier(device_ids=[self.device.index])

    def get_logger(self):
        if self.is_main_process():
            return TensorBoardLogger(self.logs_output_folder)
        else:
            return None

    def _get_latest(self, key):
        # get the latest logged value for the given key
        v = self.logger.logging[key][-1]
        return v[1] if isinstance(v, (list, tuple)) else v

    def convert_bn2syncbn(self, model):
        has_bn = False
        for m in model.modules():
            if isinstance(
                m,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                ),
            ):
                has_bn = True
                break

        if has_bn:
            if self.is_main_process():
                self.print_to_log_file(
                    "[DDP] BatchNorm detected → converting to SyncBatchNorm"
                )
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            if self.is_main_process():
                self.print_to_log_file(
                    "[DDP] No BatchNorm detected → skip SyncBatchNorm"
                )

        return model

    def get_default_training_args(self, training_args):
        """
        Get default training args in run_args.py.
        self.training_args
        """
        self.dataset_dir = training_args.dataset_dir
        self.log_dir = training_args.log_dir
        self.config_file = training_args.config_file
        self.fold = training_args.fold
        self.method_name = training_args.method_name
        self.num_epoch = training_args.num_epoch
        self.warmup_epoch = training_args.warmup_epoch
        self.batch_size = training_args.batch_size
        self.args_gpu = training_args.gpu
        self.do_compile = training_args.do_compile
        self.continue_train = training_args.continue_train
        self.pretrained_weight = training_args.pretrained_weight
        self.num_workers = training_args.num_workers
        self.random_seed = training_args.seed
        self.deterministic = training_args.no_deterministic

        self.config_dict = open_yaml(self.config_file)
        self.tr_iterations_per_epoch = self.config_dict["Training_settings"].get(
            "tr_iterations_per_epoch", 250
        )
        self.val_iterations_per_epoch = self.config_dict["Training_settings"].get(
            "val_iterations_per_epoch", 50
        )
        self.do_deep_supervision = self.config_dict["Policy"]["deep_supervision"]

    def get_device(self):
        """
        args_gpu:
            None  -> use all visible GPUs
            []    -> CPU
            [0,2] -> only see GPU 0 and 2
        DDP is determined ONLY by torchrun (WORLD_SIZE > 1).
        """
        # limit visible GPUs if specified
        if self.args_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.args_gpu))

        use_cuda = torch.cuda.is_available() and (self.args_gpu != [])

        ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1 and use_cuda

        if ddp:
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")

            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
            device = torch.device("cuda:0" if use_cuda else "cpu")

        self.is_ddp = ddp
        self.sync_processes()

        return device

    def _build_deep_supervision_loss_object(self, loss, num_deep_supervision_scales):
        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # When writing model's code, we assume that its multi-scales predictions range from high resolution to low resolution
        # (or from deeper to shallower layers in the decoder).
        weights = np.array([1.0 / (2**i) for i in range(num_deep_supervision_scales)])
        if len(weights) != 1:
            if self.is_ddp:
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

        # Normalize weights so that they sum to 1
        weights /= weights.sum()

        # Restructuring the loss
        return DeepSupervisionWeightedSummator(loss, weights)

    def init_random(self):
        if self.deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True
            cudnn.deterministic = False

        seed = self.random_seed
        if self.is_ddp:
            seed = seed + self.rank

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def worker_init_fn(self, worker_id):
        """
        Different dataloader workers should have different seeds, especially when DDP is used.
        Otherwise, they will generate the same random numbers and data augmentations, which can degrade performance.
        """
        seed = self.random_seed
        if self.is_ddp:
            seed = seed + self.rank * self.num_workers
        seed = seed + worker_id

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        timestamp = time()
        if add_timestamp:
            args = (f"{datetime.fromtimestamp(timestamp)}:", *args)

        for _ in range(5):
            try:
                with open(self.log_file, "a+") as f:
                    f.write(" ".join(map(str, args)) + "\n")
                break
            except IOError:
                sleep(0.5)

        if also_print_to_console:
            print(*args)

    def setting_check(self):
        model_setting_name = self.config_dict["Policy"]
        if (
            model_setting_name.__contains__("activate")
            and model_setting_name["activate"].lower() == "prelu"
            and self.weight_decay != 0
        ):
            warnings.warn(
                f"PReLU is used but weight_decay is set to {self.weight_decay}, which is not zero. "
                "This will push the learnable slope 'a' toward 0 and degrade performance. "
                "Consider setting weight_decay to 0 for PReLU parameters.",
                UserWarning,
            )

    @staticmethod
    def check_tensor(x, name):
        if torch.isnan(x).any():
            print(f"NaN in {name}, min={x.min()}, max={x.max()}")
        if torch.isinf(x).any():
            print(f"Inf in {name}, min={x.min()}, max={x.max()}")

    def save_checkpoint(self, filename: str) -> None:
        """
        Save checkpoint with separated model weights and training state.

        Args:
            filename: Full checkpoint file path (contains training state)
        """
        # Only the main process writes checkpoints to avoid concurrent writes.
        if not self.is_main_process():
            return

        # Skip saving if checkpointing is disabled
        if self.disable_checkpointing:
            self.print_to_log_file(
                "Checkpoint saving is disabled; no file will be written."
            )
            return

        # Extract the underlying model (remove DDP or OptimizedModule wrappers)
        mod = self.network
        while isinstance(mod, (OptimizedModule, nn.parallel.DistributedDataParallel)):
            mod = mod._orig_mod if isinstance(mod, OptimizedModule) else mod.module

        checkpoint = {
            "network_weights": mod.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": (
                self.grad_scaler.state_dict() if self.grad_scaler else None
            ),
            "logging": (
                self.logger.get_checkpoint() if self.logger is not None else None
            ),
            "current_epoch": self.current_epoch,  # the epoch of the weight from, epoch id begins with 0
            "LRScheduler_state": self.lr_scheduler.state_dict(),
            "best_ema": self._best_ema,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename_or_checkpoint):
        """
        Load checkpoint and training state.
        """
        if self.is_main_process():
            self.print_to_log_file("Loading checkpoint...")

        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            if self.is_ddp:
                checkpoint_holder = [None]
                if self.is_main_process():
                    checkpoint_holder[0] = torch.load(
                        filename_or_checkpoint,
                        map_location=self.device,
                        weights_only=False,
                    )
                dist.broadcast_object_list(checkpoint_holder, src=0)
                checkpoint = checkpoint_holder[0]
            else:
                checkpoint = torch.load(
                    filename_or_checkpoint, map_location=self.device, weights_only=False
                )
        else:
            checkpoint = filename_or_checkpoint

        # Handle 'module.' prefix (DDP compatibility)
        new_state_dict = {}
        for k, v in checkpoint["network_weights"].items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v

        # Load model weights (always executed)
        target_mod = self.network
        while isinstance(
            target_mod, (OptimizedModule, nn.parallel.DistributedDataParallel)
        ):
            target_mod = (
                target_mod._orig_mod
                if isinstance(target_mod, OptimizedModule)
                else target_mod.module
            )
        target_mod.load_state_dict(new_state_dict)

        self.current_epoch = checkpoint["current_epoch"] + 1

        if self.is_main_process() and self.logger is not None:
            self.logger.load_checkpoint(checkpoint["logging"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if self.grad_scaler is not None:
            scaler_state = checkpoint.get("grad_scaler_state")
            if scaler_state is not None:
                self.grad_scaler.load_state_dict(scaler_state)

        self.lr_scheduler.load_state_dict(checkpoint["LRScheduler_state"])

        self._best_ema = checkpoint.get("best_ema")

        if self.is_main_process():
            self.print_to_log_file(f"Resumed training from epoch {self.current_epoch}")

    def run_training(self):
        try:
            self.train_start()

            for epoch in range(self.current_epoch, self.num_epoch):
                self.epoch_start(epoch)

                # ---------- train ----------
                self.train_epoch_start(epoch)
                train_outputs = []
                tr_pbar = tqdm(
                    range(self.tr_iterations_per_epoch),
                    disable=(self.rank != 0) or not self.verbose,
                    desc="Train",
                    dynamic_ncols=True,
                )
                for _ in tr_pbar:
                    batch = next(self.train_iter)
                    loss_dict = self.train_step(batch)
                    train_outputs.append(loss_dict)

                    postfix = {k: f"{v:.4f}" for k, v in loss_dict.items()}
                    tr_pbar.set_postfix(postfix, refresh=True)
                self.train_epoch_end(train_outputs, epoch)

                # ---------- validate ----------
                with torch.no_grad():
                    self.validation_epoch_start()
                    val_outputs = []
                    val_pbar = tqdm(
                        range(self.val_iterations_per_epoch),
                        disable=(self.rank != 0) or not self.verbose,
                        desc="Val",
                        dynamic_ncols=True,
                    )
                    for _ in val_pbar:
                        batch = next(self.val_iter)
                        loss_dict = self.validation_step(batch)
                        val_outputs.append(loss_dict)

                        postfix = {k: f"{v:.4f}" for k, v in loss_dict.items()}
                        val_pbar.set_postfix(postfix, refresh=True)
                    self.validation_epoch_end(val_outputs, epoch)

                self.epoch_end(epoch)

            self.train_end()
        finally:
            if self.is_ddp and dist.is_initialized():
                dist.destroy_process_group()

    def train_start(self):
        if not self.was_initialized:
            self.initialize()

        empty_cache(self.device)

        self.dataloader_train, self.dataloader_val = self.get_train_and_val_dataloader()

        # check if final checkpoint already exists (training was already finished in a previous run), if yes, skip training and directly go to validation
        # must check after dataloader initialization, because we need to init self.allow_mirroring_axes_during_inference
        # through self.dataloader_train, self.dataloader_val = self.get_train_and_val_dataloader().
        final_ckpt = os.path.join(self.logs_output_folder, "checkpoint_final.pth")
        if os.path.isfile(final_ckpt):
            if self.is_main_process():
                self.print_to_log_file(
                    f"{final_ckpt} exists – training already finished."
                )
            self.current_epoch = self.num_epoch  # labeled as finished
            self.already_finish_training = True
        else:
            self.already_finish_training = False

    def train_end(self):
        # kill dataloader workers
        if hasattr(self, "dataloader_train"):
            if hasattr(self, "train_iter"):
                del self.train_iter
            if (
                hasattr(self, "dataloader_train")
                and self.dataloader_train._iterator is not None
            ):
                self.dataloader_train._iterator._shutdown_workers()
            del self.dataloader_train

        if hasattr(self, "dataloader_val"):
            if hasattr(self, "val_iter"):
                del self.val_iter
            if (
                hasattr(self, "dataloader_val")
                and self.dataloader_val._iterator is not None
            ):
                self.dataloader_val._iterator._shutdown_workers()
            del self.dataloader_val

        if self.is_main_process() and (not self.already_finish_training):
            # save final checkpoint
            self.save_checkpoint(
                os.path.join(self.logs_output_folder, "checkpoint_final.pth")
            )

            # del latest checkpoint
            latest_ckpt = os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
            if os.path.isfile(latest_ckpt):
                os.remove(latest_ckpt)

        empty_cache(self.device)

        if self.is_main_process():
            if self.logger is not None:
                self.logger.close()
            self.print_to_log_file("Training done.")

    def epoch_start(self, epoch):
        if self.is_ddp:
            for loader in [self.dataloader_train, self.dataloader_val]:
                sampler = loader.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

        # only rank 0 or the main process
        if self.is_main_process():
            self.logger.log("epoch_start_timestamps", time(), epoch)

    def epoch_end(self, epoch):
        self.sync_processes()
        # only rank 0 should do logging / saving / printing
        if self.is_main_process():

            self.logger.log("epoch_end_timestamps", time(), epoch)

            train_loss = self._get_latest("Train/loss")
            val_loss = self._get_latest("Val/loss")

            self.print_to_log_file("train_loss", np.round(train_loss, decimals=4))
            self.print_to_log_file("val_loss", np.round(val_loss, decimals=4))

            epoch_time = self._get_latest("epoch_end_timestamps") - self._get_latest(
                "epoch_start_timestamps"
            )
            self.print_to_log_file(f"Epoch time: {np.round(epoch_time, decimals=2)} s")
            self.logger.log("Epoch_time", epoch_time, epoch)

            # handling periodic checkpointing
            if (epoch + 1) % self.save_every == 0 and epoch != (self.num_epoch - 1):
                self.save_checkpoint(
                    os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
                )

            # EMA update
            ema_list = self.logger.logging.get("Val/ema_loss", [])

            if len(ema_list) == 0:
                ema_val = val_loss
            else:
                ema_val = (
                    self.ema_decay * self._get_latest("Val/ema_loss")
                    + (1 - self.ema_decay) * val_loss
                )

            # log EMA
            self.logger.log("Val/ema_loss", ema_val, epoch)

            # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
            if self._best_ema is None or ema_val < self._best_ema:
                self._best_ema = ema_val

                self.print_to_log_file(
                    f"New best validation ema loss: {np.round(self._best_ema, decimals=4)}"
                )
                self.save_checkpoint(
                    os.path.join(self.logs_output_folder, "checkpoint_best.pth")
                )
        self.sync_processes()
        self.current_epoch = epoch + 1

    def train_epoch_start(self, epoch):
        self.train_iter = iter(self.dataloader_train)
        self.network.train()
        self.lr_scheduler.step(epoch)

        if self.is_main_process():
            self.print_to_log_file("")
            self.print_to_log_file(f"Epoch {epoch}")
            lr_info = []
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = param_group["lr"]
                name = param_group.get("name", f"group_{i}")
                self.logger.log(f"LR/{name}", lr, epoch)
                lr_info.append(f"{name}: {lr:.6g}")
            self.print_to_log_file("learning rate: " + ", ".join(lr_info))

    def train_epoch_end(self, train_outputs, epoch):
        self.sync_processes()

        outputs = collate_outputs(train_outputs)

        log_dict = {}
        for key in outputs.keys():
            local_sum = float(np.sum(outputs[key]))
            local_count = len(outputs[key])

            if self.is_ddp:
                loss_sum = torch.tensor(local_sum, device=self.device)
                count = torch.tensor(local_count, device=self.device, dtype=torch.long)

                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)

                value = (loss_sum / count).item()
            else:
                value = local_sum / local_count

            log_dict[f"Train/{key}"] = value

        if self.is_main_process():
            self.logger.log_for_dict(log_dict, epoch)

    def validation_epoch_start(self):
        self.val_iter = iter(self.dataloader_val)
        self.network.eval()

    def validation_epoch_end(self, val_outputs, epoch):
        self.sync_processes()

        outputs = collate_outputs(val_outputs)
        log_dict = {}

        for key in outputs.keys():
            local_sum = float(np.sum(outputs[key]))
            local_count = len(outputs[key])

            if self.is_ddp:
                loss_sum = torch.tensor(local_sum, device=self.device)
                count = torch.tensor(local_count, device=self.device, dtype=torch.long)

                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)

                value = (loss_sum / count).item()
            else:
                value = local_sum / local_count

            log_dict[f"Val/{key}"] = value

        if self.is_main_process():
            self.logger.log_for_dict(log_dict, epoch)
