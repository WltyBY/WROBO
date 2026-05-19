import argparse

from wrobo.training.running.run_args import build_train_parser
from wrobo.methods.ACT.ACTTrainer import ACTTrainer

parser = argparse.ArgumentParser(
    parents=[build_train_parser()], description="Training-specific args"
)
parser.add_argument(
    "--w_KL", type=float, default=10.0, help="weight of KL Divergence Loss"
)
args = parser.parse_args()


if __name__ == "__main__":
    # import torch
    # torch.backends.cuda.enable_math_sdp(True)
    # torch.backends.cuda.enable_flash_sdp(False)
    # torch.backends.cuda.enable_mem_efficient_sdp(False)

    Trainer = ACTTrainer(training_args=args)
    Trainer.run_training()