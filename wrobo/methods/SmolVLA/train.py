import argparse

from wrobo.training.running.run_args import build_train_parser
from wrobo.methods.SmolVLA.SmolVLATrainer import (
    DEFAULT_LORA_PARTS,
    LORA_PARTS,
    SmolVLATrainer,
)


def build_parser():
    parser = argparse.ArgumentParser(
        parents=[build_train_parser()], description="SmolVLA training args"
    )

    # SmolVLA-specific arguments
    parser.add_argument(
        "--use_lora", action="store_true", default=False, help="Use LoRA for fine-tuning"
    )
    parser.add_argument(
        "--lora_parts",
        nargs="+",
        choices=(*LORA_PARTS, "all"),
        default=list(DEFAULT_LORA_PARTS),
        help="Parts to adapt with LoRA (default: action projection); use all alone "
        "to adapt language, vision, action, and projection. "
        "Selected attention uses Q/K/V/O (the final active language layer uses K/V); "
        "transformer FFN and norms remain frozen. "
        "Vision includes the vision-language connector; projection covers "
        "state/action/time projections.",
    )
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="LoRA alpha (scaling factor)"
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument(
        "--lora_save_mode",
        choices=("merged", "adapter"),
        default="merged",
        help="Save standalone merged deployment weights or LoRA adapter weights only",
    )
    return parser


if __name__ == "__main__":
    import os

    args = build_parser().parse_args()
    os.environ.update(
        {
            "HTTP_PROXY": "http://127.0.0.1:7897",
            "HTTPS_PROXY": "http://127.0.0.1:7897",
            "http_proxy": "http://127.0.0.1:7897",
            "https_proxy": "http://127.0.0.1:7897",
        }
    )
    Trainer = SmolVLATrainer(training_args=args)
    Trainer.run_training()
