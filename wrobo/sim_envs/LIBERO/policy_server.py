"""Serve a wrobo policy to the LIBERO Python 3.8 evaluation process."""

import argparse
from pathlib import Path
import random

import numpy as np

from .transport import make_server, _seed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--method", type=str.upper, default="SMOLVLA")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-file", required=True, help="YAML saved with the training model artifacts.")
    parser.add_argument("--norm-stats-key", required=True, metavar="FILE")
    parser.add_argument("--norm-stats-group")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=319)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be in [0, 65535]")
    try:
        _seed(args.seed)
        import torch
        from wrobo.real_envs.utils.policy import ModelOptions, load_policy

        def reset_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        reset_seed(args.seed)
        predictor = load_policy(ModelOptions(
            checkpoint=args.checkpoint, config_file=args.config_file,
            norm_stats_key=args.norm_stats_key, norm_stats_group=args.norm_stats_group,
            method=args.method, device=args.device, amp=args.amp, compile=args.compile,
        ))
        try:
            predictor.metadata.update({
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "config_file": str(Path(args.config_file).expanduser().resolve()),
                "norm_stats_file": str(Path(args.norm_stats_key).expanduser().resolve()),
                "norm_stats_group": args.norm_stats_group,
                "amp": args.amp, "compile": args.compile,
            })
            server = make_server(predictor, host=args.host, port=args.port, reset_callback=reset_seed)
            try:
                print("LIBERO policy server: http://{}:{} ({}, {})".format(
                    args.host, server.server_port, predictor.metadata["method"],
                    predictor.metadata["compute_device"],
                ), flush=True)
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        finally:
            predictor.close()
    except (ValueError, TypeError, RuntimeError, ImportError, OSError) as error:
        parser.exit(1, "error: {}\n".format(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
