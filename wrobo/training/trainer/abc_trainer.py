import os
import inspect
import h5py
import torch
import random
import warnings
import math
from numbers import Integral
from contextlib import nullcontext

import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from time import time, sleep
from datetime import datetime
from torch import GradScaler
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

from wrobo.utils.dataset_split import dataset_split, verify_kfold_splits
from wrobo.utils.file_operations import (
    open_yaml,
    copy_file_to_dstFolder,
    open_json,
    save_json,
)
from wrobo.utils.dataset_statistic import get_norm_stats
from wrobo.utils.normalization import normalization_parameters
from wrobo.training.dataset.selection import normalize_task_ids, resolve_episode_ids
from wrobo.training.dataset.data_keys import resolve_data_keys
from wrobo.training.dataset.episodic_dataset import EpisodicDataset
from wrobo.training.dataloader.collater import BaseCollater
from wrobo.training.trainer.loss import LossTerm
from wrobo.utils.json_export import recursive_fix_for_json_export
from wrobo.utils.load_model_weights import load_pretrained_weights
from wrobo.training.utils.mics import (
    empty_cache,
    is_main_process,
)
from wrobo.utils.collate_outputs import collate_outputs
from wrobo.training.utils.logger import TensorBoardLogger
from wrobo.training.loss.deep_supervision import DeepSupervisionWeightedSummator
from wrobo.training.dataset.sampling import (
    normalize_biased_samples,
    normalize_uniform_sample_probs,
)


class ABCTrainer(ABC):
    """
    Engine-agnostic training base class.

    This class owns the entire, backend-independent training lifecycle: the epoch loop
    (`run_training`), all epoch hooks, checkpoint orchestration, logging, dataset-split /
    norm-stats bootstrap, seeding, AMP dtype selection, and the abstract method contract
    every method trainer must fill.

    Everything that depends on a specific distributed backend (process-group setup,
    barriers, gradient/loss reduction, network wrapping, checkpoint broadcast, sampler
    epoch, rank-based seeding, teardown) is delegated to a small set of abstract
    **backend primitives**, implemented by concrete subclasses:
      - `DDPABCTrainer`        -> torch.distributed (DDP)
      - `DeepSpeedABCTrainer`  -> (future) DeepSpeed ZeRO

    The shared update assumes DDP-style averaged gradients and torch optimizers.
    Other backends must adapt their gradient reduction and optimizer/AMP semantics.

    Method trainers (e.g. ACTTrainer) must implement the abstract methods:
    - get_optimizers(self)
    - get_custom_training_args(self)
    - get_policy(self, policy_config: dict) -> nn.Module
    - get_training_img_transforms(self)
    - get_validation_img_transforms(self)
    - compute_training_loss(self, batch)
    """

    normalization_mode = "standard"
    gradient_accumulation_steps = 1
    default_grad_clip_norm = 1.0

    def __init__(self, training_args, verbose=True):
        """Bootstrap every method through the same lifecycle and method hooks."""
        self.verbose = verbose
        self.save_every = 5
        self.disable_checkpointing = False
        self.get_default_training_args(training_args)
        self.get_custom_training_args(training_args)
        self.device = self.get_device()
        self.sync_processes()
        self._setup_run_directory()
        self.check_dataset_split()
        self.check_norm_stats()
        self.norm_stats = open_json(os.path.join(self.logs_output_folder, "norm_stats.json"))
        self._validate_norm_stats()
        self.setup_method()
        self._snapshot_training_sources()
        self._setup_amp()
        self.logger = self.get_logger()
        self._best_ema = None
        self.ema_decay = getattr(self, "ema_decay", 0.9)
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

    def get_hyperparameters_name(self):
        """Method-specific segment of the existing run-directory layout."""
        return "default"

    def uses_pretrained_weights(self):
        return self.pretrained_weight is not None

    def setup_method(self):
        """Optional lightweight setup before model initialization, e.g. a tokenizer."""

    def _validate_norm_stats(self):
        """Optional model-dimension checks after the shared statistics validation."""

    def _setup_run_directory(self):
        from wrobo.methods.registry import build_run_name

        run_name = build_run_name(type(self).__name__, {
            "BS": self.batch_size, "GA": self.gradient_accumulation_steps,
            "EPOCH": self.num_epoch, "SEED": self.random_seed,
            "PRETRAINED": self.uses_pretrained_weights(), **self.get_sampling_run_args(),
        })
        self.logs_output_folder = os.path.join(
            self.log_dir, self.method_name, run_name,
            self.get_hyperparameters_name(), "fold_" + self.fold,
        )
        self.config_and_code_save_path = os.path.join(self.logs_output_folder, "Config_and_Trainer")
        if is_main_process():
            os.makedirs(self.config_and_code_save_path, exist_ok=True)
            filename = datetime.now().strftime("Train_Log_%Y_%m_%d_%H_%M_%S.txt")
            self.log_file = os.path.join(self.logs_output_folder, filename)
            with open(self.log_file, "w"):
                pass
            self.print_to_log_file(
                f"Using device: {self.device.type} | DDP: {self.is_ddp} | world_size: {self.world_size}"
            )
            self.print_to_log_file("Training logs will be saved in:", self.logs_output_folder)
        else:
            self.log_file = None
        self.sync_processes()

    def _snapshot_training_sources(self):
        """Save the config and the complete Trainer inheritance chain on rank zero."""
        if not is_main_process():
            return
        copy_file_to_dstFolder(self.config_file, self.config_and_code_save_path)
        sources = {inspect.getfile(cls) for cls in type(self).__mro__
                   if isinstance(cls, type) and issubclass(cls, ABCTrainer)}
        sources.add(inspect.getfile(LossTerm))
        for filename in sorted(sources):
            copy_file_to_dstFolder(filename, self.config_and_code_save_path)

    def _setup_amp(self):
        self.grad_scaler = None
        if self.device.type != "cuda":
            self.amp_dtype = torch.float32
            message = "Using CPU float32 training (AMP disabled)."
        elif torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            message = "Using BF16 automatic mixed precision (AMP) for training."
        else:
            self.amp_dtype = torch.float16
            self.grad_scaler = GradScaler()
            message = "Using FP16 automatic mixed precision (AMP) with GradScaler."
        if is_main_process():
            self.print_to_log_file(message)

    # ------------------------------------------------------------------ #
    # Backend primitives — implemented by a concrete distributed backend  #
    # (e.g. DDPABCTrainer with torch.distributed, or a future DeepSpeed   #
    # backend). The training loop below calls only these, never dist.*.   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def setup_distributed(self):
        """Detect / initialize the distributed context and return the device.

        Must set ``self.is_ddp`` (bool), ``self.rank`` (int) and ``self.world_size``
        (int) and return the ``torch.device`` this process should use. Honors
        ``self.args_gpu`` for visible-GPU selection (None=all, []=CPU, [ids]=subset).
        """
        raise NotImplementedError

    @abstractmethod
    def barrier(self):
        """Synchronize all processes; no-op for single-process backends."""
        raise NotImplementedError

    @abstractmethod
    def wrap_network(self, network: nn.Module) -> nn.Module:
        """Wrap the network for distributed training (e.g. DDP/SyncBN). Return as-is
        when not distributed."""
        raise NotImplementedError

    @abstractmethod
    def unwrap_network(self, network: nn.Module) -> nn.Module:
        """Return the bare module, stripping any distributed / compile wrappers, for
        state_dict save/load."""
        raise NotImplementedError

    @abstractmethod
    def reduce_scalar(self, local_sum: float, local_count: int) -> float:
        """Reduce a (sum, count) pair across processes and return the global mean.
        For a single process this is just ``local_sum / local_count``."""
        raise NotImplementedError

    @abstractmethod
    def sum_across_processes(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum a tensor across ranks; return it unchanged on a single process."""
        raise NotImplementedError

    @abstractmethod
    def gradient_sync_context(self, sync: bool):
        """Context covering BOTH forward and backward for one micro-batch."""
        raise NotImplementedError

    @abstractmethod
    def broadcast_checkpoint(self, load_fn):
        """Load a checkpoint on rank 0 via ``load_fn()`` and broadcast it to all ranks.
        For a single process, just return ``load_fn()``."""
        raise NotImplementedError

    @abstractmethod
    def set_sampler_epoch(self, epoch: int):
        """Propagate the epoch to distributed samplers (for shuffling). No-op when not
        distributed."""
        raise NotImplementedError

    @abstractmethod
    def seed_offset(self) -> int:
        """Per-rank seed offset so different ranks draw different randomness."""
        raise NotImplementedError

    @abstractmethod
    def worker_seed_offset(self, worker_id: int) -> int:
        """Per-(rank, worker) seed offset for dataloader workers."""
        raise NotImplementedError

    @abstractmethod
    def teardown_distributed(self):
        """Tear down the distributed context (e.g. destroy the process group)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Method-specific contract — implemented by each method trainer       #
    # ------------------------------------------------------------------ #

    def initialize(self):
        """Build, place and wrap a policy, then initialize its optimizer and loss."""
        if self.was_initialized:
            raise RuntimeError("self.initialize() should only be called once.")
        self.init_random()
        self.setting_check()
        self.config_dict["Policy"]["camera_names"] = [
            key.replace("observations/", "")
            for key in self.config_dict["Training_settings"]["data_keys"]
            if "image" in key and "observations" in key
        ]
        self.policy = self.build_training_policy()
        self.policy.to(self.device)
        self.log_model_parameters(self.policy)
        self.policy = self.wrap_network(self.policy)
        if self.compile:
            if is_main_process():
                self.print_to_log_file("Compiling network...")
            # torch.set_float32_matmul_precision("high")
            # This Inductor optimization can corrupt compiled backward gradients;
            # use the same protection for every method without a global config change.
            self.policy = torch.compile(self.policy, options={"inplace_buffers": False})
            torch.compiler.reset()
        self.optimizer, self.lr_scheduler = self.get_optimizers()
        self.initialize_loss()
        self.was_initialized = True

    def build_training_policy(self):
        """Override for model artifacts or adapters; return the unwrapped policy."""
        policy = self.get_policy(self.config_dict["Policy"])
        self.load_pretrained_policy_weights(policy)
        return policy

    def load_pretrained_policy_weights(self, policy):
        if self.pretrained_weight is not None:
            if is_main_process():
                self.print_to_log_file(f"Loading pretrained weight from {self.pretrained_weight}")
            load_pretrained_weights(policy, self.pretrained_weight)

    @abstractmethod
    def initialize_loss(self):
        """Optional initialization of the method's unreduced loss functions."""
        pass

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

    def get_text_transform(self):
        return None

    def _action_chunk_size(self):
        return self.config_dict["Policy"]["action_chunk_size"]

    def get_train_and_val_dataset(self):
        """Build both splits from one set of source, sampling and normalization args."""
        common = {
            "dataset_dir": self.dataset_dir, "fold": self.fold,
            "data_keys": self.config_dict["Training_settings"]["data_keys"],
            "history_width": self.config_dict["Policy"]["history_width"],
            "action_chunk_size": self._action_chunk_size(), "norm_stats": self.norm_stats,
            "text_transform": self.get_text_transform(), "biased_sample": self.biased_sample,
            "uniform_sample_prob": self.uniform_sample_prob,
            "instruction_mode": self.instruction_mode, "task_ids": self.task_ids,
            "normalization_mode": self.normalization_mode,
        }
        return tuple(EpisodicDataset(split=split, img_transforms=transform, **common)
                     for split, transform in (
                         ("train", self.get_training_img_transforms()),
                         ("val", self.get_validation_img_transforms()),
                     ))

    @abstractmethod
    def build_sampler(self, dataset):
        """
        Build a sampler for the dataset. Distributed backends should partition the
        dataset across processes (e.g. DistributedSampler).
        """
        pass

    def get_train_and_val_dataloader(self):
        """Use global micro-batch semantics and the backend sampler for both splits."""
        if self.batch_size < 1 or self.batch_size % self.world_size:
            raise ValueError(
                f"Batch size {self.batch_size} must be positive and divisible by world size {self.world_size}"
            )
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        workers = 0 if self.num_workers == 0 else max(1, self.num_workers // self.world_size)
        return tuple(DataLoader(
            dataset, batch_size=self.batch_size // self.world_size,
            sampler=self.build_sampler(dataset), shuffle=False, num_workers=workers,
            collate_fn=BaseCollater(), pin_memory=self.device.type == "cuda",
            persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None,
            worker_init_fn=self.worker_init_fn if split == "train" else None, drop_last=True,
        ) for split, dataset in zip(("train", "val"), self.get_train_and_val_dataset()))

    def _resolve_data_keys(self):
        bindings = resolve_data_keys(self.config_dict["Training_settings"]["data_keys"])
        self.proprio_path = bindings.proprio_path
        self.proprio_key = bindings.proprio_key
        self.action_key = bindings.action_key

    def prepare_inputs(self, batch):
        """Move selected camera, state, action and language tensors to the device."""
        images = {name: batch[name].to(self.device, non_blocking=True)
                  for name in self.config_dict["Policy"]["camera_names"] if name in batch}
        required = (self.proprio_key, self.action_key, "is_pad")
        missing = [name for name in required if name not in batch]
        if missing:
            method = type(self).__name__.removesuffix("Trainer")
            raise KeyError(
                f"{method} batch is missing configured fields {missing}; available keys: {list(batch)}. "
                "Check Training_settings.data_keys."
            )
        proprio = batch[self.proprio_key].to(self.device, non_blocking=True)
        actions = batch[self.action_key].to(self.device, non_blocking=True)
        is_pad = batch["is_pad"].to(self.device, non_blocking=True)
        instruction = batch.get("instruction")
        if isinstance(instruction, torch.Tensor):
            instruction = instruction.to(self.device, non_blocking=True)
        return images, list(images), proprio, actions, is_pad, instruction

    def autocast_context(self):
        return (torch.autocast(self.device.type, dtype=self.amp_dtype)
                if self.device.type == "cuda" else nullcontext())

    def _compiled_training_context(self):
        """Compile backward for the AMP-free context used by train_step."""
        if not getattr(self, "compile", False):
            return nullcontext()
        from torch._functorch import config

        # AOTAutograd traces backward during the lazy compiled forward. Its
        # default assumes backward is inside forward's autocast context, while
        # all method loss hooks leave that context before train_step backward.
        return config.patch(backward_pass_autocast="off")

    @staticmethod
    def masked_action_loss(per_element, is_pad):
        """Average over valid action elements, with a float32 mask and denominator."""
        mask = (~is_pad[:, :per_element.shape[1]]).unsqueeze(-1).to(torch.float32)
        count = mask.sum() * per_element.shape[-1]
        return (per_element * mask).sum() / count.clamp_min(1)

    def _check_finite(self, values, *, stage, clear_grad=False, diagnostic=None):
        """Reject invalid numerics on every rank before updating weights or logs.

        All ranks must call this at the same stage, including ranks without a
        logger. The collective flag prevents one rank from raising while another
        enters a DDP backward or waits at the next barrier.
        """
        bad = {
            name: ~torch.isfinite(
                value.detach() if isinstance(value, torch.Tensor)
                else torch.as_tensor(value, dtype=torch.float64, device=self.device)
            ).all()
            for name, value in values.items()
        }
        local_bad = (torch.stack(list(bad.values())).any() if bad
                     else torch.zeros((), dtype=torch.bool, device=self.device))
        failed_ranks = self.sum_across_processes(local_bad.to(torch.int32))
        if failed_ranks.item():
            names = [name for name, flag in bad.items() if flag.item()]
            details = ", ".join(names) if names else "values on another rank"
            if names and diagnostic is not None:
                details += "; " + diagnostic()
            if clear_grad:
                self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"Non-finite {stage} at epoch {getattr(self, 'current_epoch', '?')}: {details}. "
                "Training stopped before the optimizer or checkpoint was updated."
            )

    def _nonfinite_gradient_details(self):
        """Inspect individual gradients only after the collective failure flag."""
        names = []
        for name, parameter in self.policy.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                gradient = gradient.coalesce().values()
            if not torch.isfinite(gradient).all().item():
                names.append(name)
                if len(names) == 5:
                    break
        if names:
            return "non-finite gradient parameters (up to 5): " + ", ".join(names)
        return "all gradient elements are finite; the total gradient norm itself is non-finite"

    @abstractmethod
    def compute_training_loss(self, batch) -> dict[str, LossTerm]:
        """
        Forward one micro-batch and return named local-mean loss components.

        Do not backward, clip, clear gradients, or step the optimizer here.
        The shared train_step owns those operations and the accumulation weights.
        """
        pass

    @torch.inference_mode()
    def validation_step(self, batch):
        """Evaluate the same forward/loss definitions without recording a graph."""
        terms = self.compute_training_loss(batch)
        self._check_finite(
            {**{name: term.value for name, term in terms.items()},
             "combined loss": sum(term.value * term.weight for term in terms.values())},
            stage="validation loss",
        )
        result = {name: term.value.detach().cpu().numpy() for name, term in terms.items()}
        result["loss"] = sum(result[name] * term.weight for name, term in terms.items())
        return result

    # ------------------------------------------------------------------ #
    # Engine-agnostic shared implementation                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _batch_counts(batch):
        """Read only CPU mask metadata before any micro-batch moves to the GPU."""
        mask = batch["is_pad"]
        if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("Training batches require a bool is_pad tensor shaped [B, T]")
        if mask.shape[0] == 0:
            raise ValueError("Training micro-batches must contain at least one sample")
        return {"valid_steps": int((~mask).sum().item()), "samples": mask.shape[0]}

    def train_step(self, batch):
        """Perform one optimizer update using K sequential forward/backward passes.

        The first CPU batch is supplied by the epoch loop. Prefetch K-1 more CPU
        batches to obtain the complete averaging denominators, then transfer and
        release each micro-batch's computation graph separately. Every outer
        iteration remains one optimizer update, even when K is greater than one.
        """
        batches = [batch]
        batches.extend(next(self.train_iter) for _ in range(self.gradient_accumulation_steps - 1))
        counts = [self._batch_counts(item) for item in batches]
        normalizers = ("valid_steps", "samples")
        global_counts = torch.tensor(
            [sum(item[name] for item in counts) for name in normalizers],
            dtype=torch.float64, device=self.device,
        )
        global_counts = self.sum_across_processes(global_counts).clamp_min_(1)
        world_size = self.world_size
        self.optimizer.zero_grad(set_to_none=True)
        component_sums = {}
        specifications = None

        for index, (micro_batch, micro_counts) in enumerate(zip(batches, counts)):
            with (self.gradient_sync_context(sync=index == len(batches) - 1),
                  self._compiled_training_context()):
                terms = self.compute_training_loss(micro_batch)
                if not isinstance(terms, dict) or not terms:
                    raise ValueError("compute_training_loss must return a non-empty dict of LossTerm")
                current_specifications = {}
                weighted_terms = []
                for name, term in terms.items():
                    if not isinstance(name, str) or not isinstance(term, LossTerm):
                        raise ValueError("Training loss components must be named LossTerm values")
                    if term.normalization not in normalizers:
                        raise ValueError(f"Unknown loss normalization: {term.normalization}")
                    if not isinstance(term.value, torch.Tensor) or term.value.ndim != 0:
                        raise ValueError(f"Training loss {name!r} must be a scalar tensor")
                    if not math.isfinite(term.weight):
                        raise ValueError(f"Training loss {name!r} has a non-finite weight")
                    current_specifications[name] = (term.normalization, term.weight)
                    denominator = global_counts[normalizers.index(term.normalization)]
                    # DDP averages gradients across ranks. Compensate here so both
                    # uneven padding and uneven sample counts match concatenation.
                    fraction = (micro_counts[term.normalization] * world_size / denominator).to(
                        dtype=term.value.dtype
                    )
                    contribution = term.value * fraction
                    weighted_terms.append(contribution * term.weight)
                    detached = contribution.detach().to(torch.float64)
                    component_sums[name] = component_sums.get(name, 0) + detached
                if specifications is None:
                    specifications = current_specifications
                    if "loss" in terms and len(terms) != 1:
                        raise ValueError("Name individual components separately; 'loss' is the total objective")
                elif current_specifications != specifications:
                    raise ValueError("Loss components and weights must stay constant within an update")

                loss = sum(weighted_terms)
                if not loss.requires_grad:
                    raise ValueError("The combined training loss must be differentiable")
                self._check_finite(
                    {**{name: term.value for name, term in terms.items()}, "combined loss": loss},
                    stage=f"training loss (micro-batch {index + 1}/{len(batches)})",
                    clear_grad=True,
                )
                if self.grad_scaler is not None:
                    self.grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
                del loss, terms, weighted_terms, contribution, term

        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
        clip_norm = self.config_dict["Training_settings"].get(
            "grad_clip_norm", self.default_grad_clip_norm
        )
        if self.grad_scaler is None:
            parameters = list(self.policy.parameters())
            gradient_norm = torch.nn.utils.get_total_norm(
                [parameter.grad for parameter in parameters if parameter.grad is not None]
            )
            self._check_finite(
                {"gradient norm": gradient_norm}, stage="gradients", clear_grad=True,
                diagnostic=self._nonfinite_gradient_details,
            )
            torch.nn.utils.clip_grads_with_norm_(parameters, clip_norm, gradient_norm)
        else:
            # GradScaler records non-finite gradients during unscale_ and skips
            # the optimizer update while lowering the scale in update().
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), clip_norm)
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        names = sorted(component_sums)
        metrics = self.sum_across_processes(
            torch.stack([component_sums[name] for name in names])
        ) / world_size
        result = dict(zip(names, metrics.cpu().numpy()))
        result["loss"] = sum(result[name] * specifications[name][1] for name in names)
        return result

    def training_batch_config(self):
        """Batch semantics saved with every resumable checkpoint."""
        return {
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "world_size": self.world_size,
            "effective_batch_size": self.batch_size * self.gradient_accumulation_steps,
        }

    def validate_training_batch_config(self, checkpoint):
        expected = self.training_batch_config()
        if checkpoint.get("training_batch_config") != expected:
            raise ValueError(
                "Checkpoint training_batch_config differs from this run: "
                f"saved={checkpoint.get('training_batch_config')}, requested={expected}. "
                "Resume with the same batch size, accumulation steps, and world size; "
                "use weights-only initialization for a new experiment."
            )

    def check_dataset_split(self):
        """
        Check and create dataset splits for all datasets.
        Each dataset must have its own dataset_split.json.
        """
        if is_main_process():
            for dataset_dir in self.dataset_dir:
                splits_path = os.path.join(dataset_dir, "dataset_split.json")
                if not os.path.exists(splits_path):
                    self.print_to_log_file(
                        f"Dataset split not found under {dataset_dir}."
                    )
                    self.print_to_log_file("Creating new splits...")

                    # Episodes are in episodes/ subdirectory
                    episodes_dir = os.path.join(dataset_dir, "episodes")
                    case_lst = [
                        f.replace(".hdf5", "")
                        for f in os.listdir(episodes_dir)
                        if os.path.isfile(os.path.join(episodes_dir, f))
                        and f.endswith(".hdf5")
                    ]
                    # As model usually needs to be test in simulated or real environment, we use K-Fold cross-validation to split the dataset.
                    # You can just use one fold for training or set --fold to 'all' to use all data for training.
                    case_lst.sort()
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
        """Collect one statistics format from the selected training episodes."""
        self.split_episode_ids = [
            {split: resolve_episode_ids(directory, self.fold, split, tasks)
             for split in ("train", "val")}
            for directory, tasks in zip(self.dataset_dir, self.task_ids)
        ]
        selection = {
            "schema_version": 1,
            "fold": str(self.fold),
            "seed": self.random_seed,
            "normalization_mode": self.normalization_mode,
            "merge_norm_stats": self.merge_norm_stats,
            "data_keys": self.config_dict["Training_settings"]["data_keys"],
            "num_sample": 1000,
            "img_num_sample": 200,
            "datasets": [
                {"source": os.path.realpath(directory), "norm_stats_key": directory,
                 "task_ids": tasks, **members}
                for directory, tasks, members in zip(
                    self.dataset_dir, self.task_ids, self.split_episode_ids)
            ],
        }
        if is_main_process():
            for directory, tasks, members in zip(
                    self.dataset_dir, self.task_ids, self.split_episode_ids):
                self.print_to_log_file(
                    f"Dataset {directory}: tasks={tasks if tasks is not None else 'all'}, "
                    f"train={len(members['train'])}, val={len(members['val'])}"
                )
            stats_path = os.path.join(self.logs_output_folder, "norm_stats.json")
            selection_path = os.path.join(self.logs_output_folder, "norm_stats_selection.json")
            if os.path.exists(stats_path) or os.path.exists(selection_path):
                if not (os.path.isfile(stats_path) and os.path.isfile(selection_path)):
                    raise ValueError(
                        "Normalization statistics require norm_stats.json and "
                        "norm_stats_selection.json. Generate statistics in a new log directory."
                    )
                if open_json(selection_path) != selection:
                    raise ValueError(
                        "Training data selection or normalization settings changed for this run. "
                        "Use a new log directory; statistics and checkpoints cannot be reused."
                    )
                stats = open_json(stats_path)
                self._validate_statistics_format(stats)
                self.print_to_log_file(
                    f"Reusing cached normalization statistics from {stats_path}; no recomputation."
                )
            else:
                if any(name.startswith("checkpoint") and name.endswith(".pth")
                       for name in os.listdir(self.logs_output_folder)):
                    raise ValueError(
                        "Checkpoints exist without normalization statistics and their selection record. "
                        "Use a new log directory instead of regenerating statistics for these weights."
                    )
                paths_by_dataset = [
                    [os.path.join(directory, "episodes", f"{episode}.hdf5")
                     for episode in members["train"]]
                    for directory, members in zip(self.dataset_dir, self.split_episode_ids)
                ]
                started = time()
                if self.merge_norm_stats or len(self.dataset_dir) == 1:
                    paths = [path for group in paths_by_dataset for path in group]
                    self.print_to_log_file(
                        f"Computing merged normalization statistics: {len(paths_by_dataset)} dataset(s), "
                        f"{len(paths)} train episodes, num_workers={self.num_workers}."
                    )
                    stats = get_norm_stats(paths, seed=self.random_seed,
                                           num_workers=self.num_workers,
                                           log_fn=self.print_to_log_file)
                else:
                    if len(set(self.dataset_dir)) != len(self.dataset_dir):
                        raise ValueError(
                            "Separate normalization groups require distinct dataset_dir values. "
                            "Use one task selection per directory or --merge_norm_stats."
                        )
                    stats = {}
                    for index, (directory, paths) in enumerate(zip(self.dataset_dir, paths_by_dataset), 1):
                        self.print_to_log_file(
                            f"Computing separate normalization statistics [{index}/{len(paths_by_dataset)}]: "
                            f"{directory}, {len(paths)} train episodes, num_workers={self.num_workers}."
                        )
                        stats[directory] = get_norm_stats(
                            paths, seed=self.random_seed, num_workers=self.num_workers,
                            log_fn=self.print_to_log_file,
                        )
                recursive_fix_for_json_export(stats)
                self._validate_statistics_format(stats)
                save_json(stats, stats_path)
                save_json(selection, selection_path)
                self.print_to_log_file(
                    f"Normalization statistics saved to {stats_path} in {time() - started:.1f}s."
                )
            self.print_to_log_file(f"Using norm stats from {stats_path}")
        self.sync_processes()

    def _validate_statistics_format(self, stats):
        """Require genuine moments/quantiles matching the selected numerical fields."""
        if not isinstance(stats, dict) or not stats:
            raise ValueError("Normalization statistics must be a non-empty dictionary")
        if self.merge_norm_stats or len(self.dataset_dir) == 1:
            groups = [(stats, list(range(len(self.dataset_dir))))]
        else:
            if set(stats) != set(self.dataset_dir):
                raise ValueError("Normalization groups do not match dataset_dir")
            groups = [(stats[directory], [index])
                      for index, directory in enumerate(self.dataset_dir)]
        fields = [path for path in self.config_dict["Training_settings"]["data_keys"]
                  if path.startswith("observations/proprio_") or path.startswith("action_")]
        for group, sources in groups:
            if not isinstance(group, dict) or not group:
                raise ValueError("Normalization groups must contain channel statistics")
            missing = [path.removeprefix("observations/") for path in fields
                       if path.removeprefix("observations/") not in group]
            if missing:
                raise ValueError(f"Normalization statistics are missing selected fields {missing}")
            for key, entry in group.items():
                mean, _ = normalization_parameters(entry, "standard")
                if "image" not in key.lower():
                    center, _ = normalization_parameters(entry, "quantile")
                    if center.shape != mean.shape:
                        raise ValueError(f"Inconsistent moments and quantiles for {key}")
            # Header checks also prevent a one-value cached vector from silently
            # broadcasting across a multi-dimensional state or action.
            for source in sources:
                directory = self.dataset_dir[source]
                episode = self.split_episode_ids[source]["train"][0]
                filename = os.path.join(directory, "episodes", f"{episode}.hdf5")
                with h5py.File(filename, "r") as handle:
                    for path in fields:
                        key = path.removeprefix("observations/")
                        if (path not in handle or not isinstance(handle[path], h5py.Dataset)
                                or handle[path].ndim != 2):
                            raise ValueError(f"Selected numerical field {path!r} is missing or invalid in {filename}")
                        if len(group[key]["mean"]) != handle[path].shape[1]:
                            raise ValueError(f"Normalization width for {key!r} does not match {filename}")

    def sync_processes(self):
        """Backend-agnostic process synchronization (delegates to ``barrier``)."""
        self.barrier()

    def get_logger(self):
        if is_main_process():
            return TensorBoardLogger(self.logs_output_folder)
        else:
            return None

    def _get_latest(self, key):
        # get the latest logged value for the given key
        v = self.logger.logging[key][-1]
        return v[1] if isinstance(v, (list, tuple)) else v

    def get_default_training_args(self, training_args):
        """
        Get default training args in run_args.py.
        self.training_args
        """
        # Normalize dataset_dir to always be a list for uniform processing
        dataset_dir = training_args.dataset_dir
        if isinstance(dataset_dir, str):
            self.dataset_dir = [dataset_dir]
        elif isinstance(dataset_dir, list):
            self.dataset_dir = dataset_dir
        else:
            raise TypeError(f"dataset_dir must be str or list, got {type(dataset_dir)}")

        self.task_ids = normalize_task_ids(training_args.task_ids, len(self.dataset_dir))

        self.log_dir = training_args.log_dir
        self.config_file = training_args.config_file
        self.fold = training_args.fold
        self.method_name = training_args.method_name
        self.num_epoch = training_args.num_epoch
        self.warmup_epoch = training_args.warmup_epoch
        self.batch_size = training_args.batch_size
        accumulation_steps = training_args.gradient_accumulation_steps
        if (isinstance(accumulation_steps, bool) or not isinstance(accumulation_steps, Integral)
                or accumulation_steps < 1):
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        self.gradient_accumulation_steps = int(accumulation_steps)
        self.args_gpu = training_args.gpu
        self.compile = training_args.compile
        self.continue_train = training_args.continue_train
        self.pretrained_weight = training_args.pretrained_weight
        self.num_workers = training_args.num_workers
        self.random_seed = training_args.seed
        self.deterministic = training_args.no_deterministic
        sampling_modes = normalize_biased_samples(training_args.biased_sample, len(self.dataset_dir))
        # Keep shared-mode run names unchanged, including when resuming old runs.
        self.biased_sample = (
            sampling_modes[0] if len(set(sampling_modes)) == 1 else sampling_modes
        )
        uniform_probs = normalize_uniform_sample_probs(
            getattr(training_args, "uniform_sample_prob", 0.0), len(self.dataset_dir)
        )
        self.uniform_sample_prob = (
            uniform_probs[0] if len(set(uniform_probs)) == 1 else uniform_probs
        )
        self.merge_norm_stats = getattr(training_args, "merge_norm_stats", True)

        self.config_dict = open_yaml(self.config_file)
        self.config_dict["Policy"].setdefault("deep_supervision", False)
        self.config_dict["Policy"].setdefault("history_width", 1)
        self._resolve_data_keys()
        from wrobo.training.dataset.instruction import normalize_instruction_modes
        self.instruction_mode = (
            normalize_instruction_modes(getattr(training_args, "instruction_mode", None), len(self.dataset_dir))
            if "instruction" in self.config_dict["Training_settings"]["data_keys"] else None
        )
        self.tr_iterations_per_epoch = self.config_dict["Training_settings"].get(
            "tr_iterations_per_epoch", 250
        )
        self.val_iterations_per_epoch = self.config_dict["Training_settings"].get(
            "val_iterations_per_epoch", 50
        )
        self.do_deep_supervision = self.config_dict["Policy"]["deep_supervision"]

    def get_sampling_run_args(self):
        """Identify sampling settings in the run name."""
        values = {"SAMPLE": self.biased_sample}
        probability = self.uniform_sample_prob
        probabilities = probability if isinstance(probability, list) else [probability]
        if any(value != 0.0 for value in probabilities):
            values["UNIFORM_PROB"] = probability
        return values

    def get_device(self):
        """
        Resolve the device via the backend's ``setup_distributed`` primitive, which also
        sets ``self.is_ddp`` / ``self.rank`` / ``self.world_size``.
        """
        return self.setup_distributed()

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

        seed = self.random_seed + self.seed_offset()

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
        seed = self.random_seed + self.worker_seed_offset(worker_id)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if not is_main_process() or self.log_file is None:
            if also_print_to_console:
                print(*args, flush=True)
            return

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
            print(*args, flush=True)

    def log_model_parameters(self, network):
        """Print the model's total parameter count to the run's log .txt.

        Reused by any Trainer subclass: call it once after the network is built (and on
        the device). Main process only; print_to_log_file is already rank-guarded."""
        if not is_main_process():
            return
        total = sum(p.numel() for p in network.parameters())
        trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
        trainable_pct = (trainable / total * 100) if total > 0 else 0
        self.print_to_log_file(
            f"Num of Params of Policy: {total:,} (trainable {trainable:,}, {trainable_pct:.2f}%)"
        )

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

    def save_checkpoint(self, filename: str, weights_only: bool = False) -> None:
        """
        Save checkpoint with full training state or model weights only.

        Args:
            filename: Full checkpoint file path
            weights_only: If True, save only model weights (state_dict).
                         If False, save full training state for resume.
        """
        # Only the main process writes checkpoints to avoid concurrent writes.
        if not is_main_process():
            return

        # Skip saving if checkpointing is disabled
        if self.disable_checkpointing:
            self.print_to_log_file(
                "Checkpoint saving is disabled; no file will be written."
            )
            return

        # Extract the underlying model (remove distributed / compile wrappers)
        mod = self.unwrap_network(self.policy)

        if weights_only:
            # Save only model weights for inference/deployment
            torch.save(mod.state_dict(), filename)
        else:
            # Save full training state for resuming training
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
                "training_batch_config": self.training_batch_config(),
                "instruction_config": {
                    "dataset_dir": self.dataset_dir,
                    "instruction_mode": self.instruction_mode,
                },
            }
            torch.save(checkpoint, filename)

    def load_checkpoint(self, filename_or_checkpoint):
        """
        Load checkpoint and training state.

        Handles both full checkpoints (with optimizer state) and weights-only checkpoints.
        """
        if is_main_process():
            self.print_to_log_file("Loading checkpoint...")

        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = self.broadcast_checkpoint(
                lambda: torch.load(
                    filename_or_checkpoint,
                    map_location=self.device,
                    weights_only=False,
                )
            )
        else:
            checkpoint = filename_or_checkpoint

        # Detect if this is a weights-only checkpoint or full checkpoint
        is_weights_only = (
            not isinstance(checkpoint, dict) or "network_weights" not in checkpoint
        )

        if is_weights_only:
            # This is a weights-only checkpoint (state_dict directly)
            state_dict = checkpoint

            # Handle 'module.' prefix (DDP compatibility)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v

            # Load model weights only
            target_mod = self.unwrap_network(self.policy)
            target_mod.load_state_dict(new_state_dict)

            if is_main_process():
                self.print_to_log_file(
                    "Loaded weights-only checkpoint (optimizer state not restored)"
                )
        else:
            # This is a full checkpoint with training state
            self.validate_training_batch_config(checkpoint)
            # Handle 'module.' prefix (DDP compatibility)
            new_state_dict = {}
            for k, v in checkpoint["network_weights"].items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v

            # Load model weights (always executed)
            target_mod = self.unwrap_network(self.policy)
            target_mod.load_state_dict(new_state_dict)

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
                    f"Resumed training from epoch {self.current_epoch}"
                )

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
            self.teardown_distributed()

    def train_start(self):
        if not self.was_initialized:
            self.initialize()

        if is_main_process():
            batch_config = self.training_batch_config()
            save_json(batch_config, os.path.join(self.logs_output_folder, "training_batch_config.json"))
            self.print_to_log_file(
                f"Global micro-batch: {self.batch_size} | per GPU: {self.batch_size // self.world_size} | "
                f"accumulation steps: {self.gradient_accumulation_steps} | "
                f"effective global batch: {batch_config['effective_batch_size']} | "
                f"optimizer updates per epoch: {self.tr_iterations_per_epoch}"
            )
            save_json(
                {"dataset_dir": self.dataset_dir,
                 "task_ids": self.task_ids,
                 "normalization_mode": self.normalization_mode,
                 "instruction_mode": self.instruction_mode,
                 "data_keys": self.config_dict["Training_settings"]["data_keys"]},
                os.path.join(self.logs_output_folder, "instruction_config.json"),
            )

        empty_cache(self.device)

        self.dataloader_train, self.dataloader_val = self.get_train_and_val_dataloader()

        # check if final checkpoint already exists (training was already finished in a previous run), if yes, skip training and directly go to validation
        # must check after dataloader initialization, because we need to init self.allow_mirroring_axes_during_inference
        # through self.dataloader_train, self.dataloader_val = self.get_train_and_val_dataloader().
        final_ckpt = os.path.join(self.logs_output_folder, "checkpoint_final.pth")
        if os.path.isfile(final_ckpt):
            if is_main_process():
                self.print_to_log_file(
                    f"{final_ckpt} exists – training already finished."
                )
            self.current_epoch = self.num_epoch  # labeled as finished
            self.already_finish_training = True
        else:
            self.already_finish_training = False
            if is_main_process():
                from wrobo.training.utils.dataset_snapshot import save_dataset_snapshot
                save_dataset_snapshot(
                    self.dataloader_train.dataset,
                    self.dataloader_val.dataset,
                    os.path.join(self.logs_output_folder, "Config_and_Trainer"),
                )

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

        if is_main_process() and (not self.already_finish_training):
            # Save final checkpoint with full training state (for fine-tuning)
            self.save_checkpoint(
                os.path.join(
                    self.logs_output_folder, "checkpoint_final_retrainable.pth"
                ),
                weights_only=False,
            )
            self.print_to_log_file(
                "Saved checkpoint_final_retrainable.pth (full training state)"
            )

            # Save final weights-only checkpoint (for inference/deployment)
            self.save_checkpoint(
                os.path.join(self.logs_output_folder, "checkpoint_final.pth"),
                weights_only=True,
            )
            self.print_to_log_file("Saved checkpoint_final.pth (weights only)")

            # If best retrainable checkpoint exists, extract and save its weights-only version
            best_retrainable = os.path.join(
                self.logs_output_folder, "checkpoint_best_retrainable.pth"
            )
            if os.path.isfile(best_retrainable):
                best_checkpoint = torch.load(
                    best_retrainable, map_location="cpu", weights_only=False
                )
                # Extract model weights from the full checkpoint
                best_weights = best_checkpoint["network_weights"]
                # Save as weights-only checkpoint
                best_weights_path = os.path.join(
                    self.logs_output_folder, "checkpoint_best.pth"
                )
                torch.save(best_weights, best_weights_path)
                self.print_to_log_file("Saved checkpoint_best.pth (weights only)")

            # Delete latest checkpoint (training finished, no need for periodic resume)
            latest_ckpt = os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
            if os.path.isfile(latest_ckpt):
                os.remove(latest_ckpt)
                self.print_to_log_file("Removed checkpoint_latest.pth")

        empty_cache(self.device)

        if is_main_process():
            if self.logger is not None:
                self.logger.close()
            self.print_to_log_file("Training done.")

    def epoch_start(self, epoch):
        self.set_sampler_epoch(epoch)

        # only rank 0 or the main process
        if is_main_process():
            self.logger.log("epoch_start_timestamps", time(), epoch)

    def epoch_end(self, epoch):
        self.sync_processes()
        main_process = is_main_process()
        epoch_metrics = {}
        if main_process:
            train_loss = self._get_latest("Train/loss")
            val_loss = self._get_latest("Val/loss")
            ema_list = self.logger.logging.get("Val/ema_loss", [])
            if len(ema_list) == 0:
                ema_val = val_loss
            else:
                previous_ema = self._get_latest("Val/ema_loss")
                epoch_metrics["previous validation EMA"] = previous_ema
                ema_val = self.ema_decay * previous_ema + (1 - self.ema_decay) * val_loss
            epoch_metrics.update(train_loss=train_loss, val_loss=val_loss, validation_ema=ema_val)
            if self._best_ema is not None:
                epoch_metrics["best validation EMA"] = self._best_ema
        self._check_finite(epoch_metrics, stage="epoch metrics")
        # only rank 0 should do logging / saving / printing
        if main_process:

            self.logger.log("epoch_end_timestamps", time(), epoch)

            self.print_to_log_file("train_loss", np.round(train_loss, decimals=4))
            self.print_to_log_file("val_loss", np.round(val_loss, decimals=4))

            epoch_time = self._get_latest("epoch_end_timestamps") - self._get_latest(
                "epoch_start_timestamps"
            )
            self.print_to_log_file(f"Epoch time: {np.round(epoch_time, decimals=2)} s")
            self.logger.log("Epoch_time", epoch_time, epoch)

            # log EMA
            self.logger.log("Val/ema_loss", ema_val, epoch)

            # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
            if self._best_ema is None or ema_val < self._best_ema:
                self._best_ema = ema_val

                self.print_to_log_file(
                    f"New best validation ema loss: {np.round(self._best_ema, decimals=4)}"
                )
                # Save best checkpoint with full training state (for resuming/fine-tuning)
                self.save_checkpoint(
                    os.path.join(
                        self.logs_output_folder, "checkpoint_best_retrainable.pth"
                    ),
                    weights_only=False,
                )

            # Save the periodic resume checkpoint only after this epoch's EMA and best
            # state have been updated, so all serialized training state is consistent
            # with the model weights and current_epoch stored for this epoch.
            if epoch % self.save_every == 0 and epoch != (self.num_epoch - 1):
                self.save_checkpoint(
                    os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
                )
        self.sync_processes()
        self.current_epoch = epoch + 1

    def train_epoch_start(self, epoch):
        """Select the current zero-based epoch's LR before its optimizer updates."""
        self.train_iter = iter(self.dataloader_train)
        self.policy.train()
        # On resume, epoch is the saved epoch + 1, so the LR is not repeated.
        self.lr_scheduler.step(epoch)

        if is_main_process():
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
            log_dict[f"Train/{key}"] = self.reduce_scalar(local_sum, local_count)

        self._check_finite(log_dict, stage="training epoch metrics")
        if is_main_process():
            self.logger.log_for_dict(log_dict, epoch)

    def validation_epoch_start(self):
        self.val_iter = iter(self.dataloader_val)
        self.policy.eval()

    def validation_epoch_end(self, val_outputs, epoch):
        self.sync_processes()

        outputs = collate_outputs(val_outputs)
        log_dict = {}

        for key in outputs.keys():
            local_sum = float(np.sum(outputs[key]))
            local_count = len(outputs[key])
            log_dict[f"Val/{key}"] = self.reduce_scalar(local_sum, local_count)

        self._check_finite(log_dict, stage="validation epoch metrics")
        if is_main_process():
            self.logger.log_for_dict(log_dict, epoch)
