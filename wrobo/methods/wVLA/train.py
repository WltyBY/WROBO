import argparse

from wrobo.training.running.run_args import build_train_parser
from wrobo.methods.wVLA.wVLATrainer import wVLATrainer

parser = argparse.ArgumentParser(
    parents=[build_train_parser()], description="Training-specific args"
)
args = parser.parse_args()


if __name__ == "__main__":
    import os

    os.environ.update(
        {
            "HTTP_PROXY": "http://127.0.0.1:7897",
            "HTTPS_PROXY": "http://127.0.0.1:7897",
            "http_proxy": "http://127.0.0.1:7897",
            "https_proxy": "http://127.0.0.1:7897",
        }
    )
    Trainer = wVLATrainer(training_args=args)
    Trainer.run_training()
