"""CLI only: parse running args and launch PI05Trainer; no network/training math."""
import argparse
from wrobo.training.running.run_args import build_train_parser
from wrobo.training.utils.lora import DEFAULT_LORA_PARTS, LORA_PARTS


def build_parser():
    parser = argparse.ArgumentParser(parents=[build_train_parser()], description="PI0.5 training")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument(
        "--lora_parts", nargs="+", choices=(*LORA_PARTS, "all"),
        default=list(DEFAULT_LORA_PARTS),
        help="Parts to adapt (default: action projection); use all alone for VLAP. "
             "Vision includes the visual projector; projection covers action/time projections.",
    )
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_save_mode", choices=("merged", "adapter"), default="merged")
    return parser


if __name__ == "__main__":
    from wrobo.methods.PI05.PI05Trainer import PI05Trainer
    PI05Trainer(build_parser().parse_args()).run_training()
