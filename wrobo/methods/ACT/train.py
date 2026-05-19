import torch
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
    Trainer = ACTTrainer(training_args=args)
    Trainer.run_training()