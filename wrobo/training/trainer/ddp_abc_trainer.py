import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist

from torch._dynamo import OptimizedModule
from torch.utils.data import DistributedSampler, RandomSampler

from wrobo.training.trainer.abc_trainer import ABCTrainer
from wrobo.training.dataloader.sampler import InfiniteSampler
from wrobo.training.utils.mics import (
    get_rank,
    get_world_size,
    is_main_process,
    is_dist_avail_and_initialized,
)


class DDPABCTrainer(ABCTrainer):
    """
    torch.distributed (DDP) backend for ABCTrainer.

    Implements the backend primitives declared in ABCTrainer using torch.distributed:
    process-group setup, barrier, loss all-reduce, DistributedDataParallel / SyncBatchNorm
    wrapping, checkpoint broadcast, sampler epoch, rank-based seeding, and teardown. The
    engine-agnostic training loop lives entirely in ABCTrainer.

    DDP is determined ONLY by torchrun (WORLD_SIZE > 1); when not launched under torchrun
    the primitives degrade to single-process no-ops, so the same trainer runs unchanged on
    one GPU / CPU.

    Method trainers (e.g. ACTTrainer) subclass this and implement the method-specific
    abstract methods (initialize, get_optimizers, compute_training_loss, ...).
    """

    # ------------------------------------------------------------------ #
    # Backend primitives                                                  #
    # ------------------------------------------------------------------ #

    def setup_distributed(self):
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

            if not is_dist_avail_and_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
        else:
            device = torch.device("cuda:0" if use_cuda else "cpu")

        self.is_ddp = ddp
        # When DDP is not active these helpers fall back to rank 0 / world_size 1,
        # so they are equivalent to the previous explicit assignments.
        self.rank = get_rank()
        self.world_size = get_world_size()

        return device

    def barrier(self):
        if self.is_ddp:
            dist.barrier(device_ids=[self.device.index])

    def wrap_network(self, network: nn.Module) -> nn.Module:
        if self.is_ddp:
            network = self.convert_bn2syncbn(network)
            network = nn.parallel.DistributedDataParallel(
                network,
                device_ids=[self.device.index],
                output_device=self.device.index,
            )
        return network

    def unwrap_network(self, network: nn.Module) -> nn.Module:
        mod = network
        while isinstance(mod, (OptimizedModule, nn.parallel.DistributedDataParallel)):
            mod = mod._orig_mod if isinstance(mod, OptimizedModule) else mod.module
        return mod

    def reduce_scalar(self, local_sum: float, local_count: int) -> float:
        if self.is_ddp:
            loss_sum = torch.tensor(local_sum, device=self.device)
            count = torch.tensor(local_count, device=self.device, dtype=torch.long)

            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

            return (loss_sum / count).item()
        return local_sum / local_count

    def sum_across_processes(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.is_ddp:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def gradient_sync_context(self, sync: bool):
        """Keep the DDP wrapper when bypassing compile for ``no_sync``."""
        if not self.is_ddp or sync:
            return nullcontext()
        network = self.policy
        while isinstance(network, OptimizedModule):
            network = network._orig_mod
        return network.no_sync()

    def broadcast_checkpoint(self, load_fn):
        if self.is_ddp:
            checkpoint_holder = [None]
            if is_main_process():
                checkpoint_holder[0] = load_fn()
            dist.broadcast_object_list(checkpoint_holder, src=0)
            return checkpoint_holder[0]
        return load_fn()

    def set_sampler_epoch(self, epoch: int):
        if self.is_ddp:
            for loader in [self.dataloader_train, self.dataloader_val]:
                sampler = loader.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

    def seed_offset(self) -> int:
        return self.rank if self.is_ddp else 0

    def worker_seed_offset(self, worker_id: int) -> int:
        offset = worker_id
        if self.is_ddp:
            offset += self.rank * self.num_workers
        return offset

    def teardown_distributed(self):
        if self.is_ddp and is_dist_avail_and_initialized():
            dist.destroy_process_group()

    # ------------------------------------------------------------------ #
    # DDP-specific helpers                                                #
    # ------------------------------------------------------------------ #

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
            if is_main_process():
                self.print_to_log_file(
                    "[DDP] BatchNorm detected → converting to SyncBatchNorm"
                )
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            if is_main_process():
                self.print_to_log_file(
                    "[DDP] No BatchNorm detected → skip SyncBatchNorm"
                )

        return model

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
