import argparse

from wrobo.training.dataset.sampling import SAMPLING_MODES, normalize_uniform_sample_probs


def parse_uniform_sample_prob(value):
    """Argparse type for a finite uniform-mixture probability in [0, 1]."""
    try:
        return normalize_uniform_sample_probs(float(value), 1)[0]
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "uniform_sample_prob must be a finite number in [0, 1]"
        ) from exc


def parse_device(v):
    """Custom type: convert string to list[int] or None."""
    if v.lower() == "all":
        return None  # Do not set CUDA_VISIBLE_DEVICES, use all
    if v == "-1":
        return []  # Empty list -> CPU
    try:
        return [int(x) for x in v.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid format: {v}")


def positive_integer(value):
    """An argparse integer that cannot silently disable training updates."""
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def build_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--dataset_dir",
        "-d",
        type=str,
        nargs="+",
        help="Path(s) of dataset directory. Can be a single path or multiple paths for multi-dataset training.",
        required=True,
    )
    p.add_argument(
        "--log_dir",
        type=str,
        # default="./Logs",
        help="Path of checkpoint directory",
        required=True,
    )
    p.add_argument(
        "--config_file",
        type=str,
        help="Path of config yaml, which contains setting of Policy, etc.",
        required=True,
    )
    p.add_argument(
        "--method_name",
        "-m",
        type=str,
        help="Method name for saving logs and models.",
        required=True,
    )
    p.add_argument(
        "--fold",
        "-f",
        type=str,
        help="Fold of dataset. Can be 'all' for using both train and val split for training.",
        required=True,
    )
    p.add_argument(
        "--num_epoch",
        type=int,
        help="Number of epochs for training",
        required=True,
    )
    p.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        help="Global micro-batch size across all GPUs, before gradient accumulation",
        required=True,
    )
    p.add_argument(
        "--gradient_accumulation_steps",
        type=positive_integer,
        default=1,
        help="Micro-batches per optimizer update (default: 1). Effective global "
             "batch size is batch_size * gradient_accumulation_steps.",
    )
    p.add_argument(
        "--gpu",
        type=parse_device,
        default="0",
        help="CPU=-1; GPU ID=0; Multi-GPU=0,1,2...; All GPUs=all",
    )
    p.add_argument(
        "--warmup_epoch",
        type=int,
        default=0,
        help="Number of epochs for warmup",
    )
    p.add_argument(
        "--instruction_mode", nargs="+", default=None,
        help="One mode per dataset: episode_only (default), frame_only, episode_plus_frame, "
             "frame_with_episode_fallback. Ignored when data_keys omits instruction.",
    )
    p.add_argument(
        "--task_ids", nargs="+", default=None,
        help="One whole-episode task selection per dataset_dir: all, 0, or 0,2. "
             "Uses instruction_library.json task episodes; independent of instruction_mode.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Enable compile for training.",
    )
    p.add_argument(
        "--continue_train",
        "--c",
        action="store_true",
        help="If call this param, continue training based on the given Params: --dataset_dir/-d, --fold/-f, --method_name/-m, etc.",
    )
    p.add_argument(
        "--pretrained_weight",
        type=str,
        help="Continue training from a specific checkpoint folder.",
    )
    p.add_argument("--num_workers", type=int, default=12, help="Number of workers")
    p.add_argument("--seed", type=int, default=319, help="Random seed")
    p.add_argument(
        "--no_deterministic",
        action="store_false",
        help="Enable deterministic behavior. Call this param to disable it.",
    )
    p.add_argument(
        "--biased_sample",
        type=str,
        nargs="+",
        default=["uniform"],
        choices=SAMPLING_MODES,
        help="Timestep sampling: one shared mode or one mode per dataset_dir, in the same order. "
             "'uniform' (default), 'early' (bias towards start), 'late' (bias towards end).",
    )
    p.add_argument(
        "--uniform_sample_prob",
        type=parse_uniform_sample_prob,
        nargs="+",
        default=[0.0],
        help="Probability of choosing uniform sampling instead of early/late Beta sampling "
             "for each item. One shared probability or one per dataset_dir, in the same order. "
             "0 (default) keeps the selected mode; 0.5 mixes equally; 1 always samples uniformly. "
             "Uniform mode always stays uniform.",
    )
    p.add_argument(
        "--merge_norm_stats",
        action="store_true",
        help="For multi-dataset training: if set, merge normalization stats from all datasets into unified stats. If not set, each dataset uses its own norm_stats.",
    )
    return p
