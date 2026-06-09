import gc

import torch
import torch.distributed as dist

try:
    import deepspeed.comm.comm as ds_dist
except Exception:
    ds_dist = None


def _ds_initialized():
    """Whether deepspeed.comm has an initialized communication backend."""
    if ds_dist is None:
        return False
    try:
        return ds_dist.is_initialized()
    except Exception:
        return False


def is_dist_avail_and_initialized():
    if _ds_initialized():
        return True
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if _ds_initialized():
        return ds_dist.get_world_size()
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if _ds_initialized():
        return ds_dist.get_rank()
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def empty_cache(device: torch.device = None):
    gc.collect()
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        pass


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
