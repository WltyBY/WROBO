"""Run a wrobo checkpoint on a separate inference machine."""

import argparse

from wrobo.methods.registry import METHOD_TO_TRAINER

from .utils.device import parse_device


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True,
                        help="Listening address and compute device: HOST:PORT:DEVICE (e.g. 0.0.0.0:8080:cuda).")
    parser.add_argument("--checkpoint", required=True, help="Explicit checkpoint file path.")
    parser.add_argument("--method", type=str.upper, choices=tuple(METHOD_TO_TRAINER), required=True,
                        help="Registered method name.")
    parser.add_argument("--config-file", required=True, help="Explicit training YAML file path.")
    parser.add_argument("--data-description", help="Optional training data_description.json for explicit column ordering and units.")
    parser.add_argument("--norm-stats-key", metavar="FILE", required=True, help="Explicit normalization JSON file path.")
    parser.add_argument("--norm-stats-group", help="Exact group name if the normalization JSON contains multiple datasets.")
    parser.add_argument("--instruction", help="Default language instruction; clients may supply their own.")
    parser.add_argument("--compile", action="store_true", help="Compile the loaded model for inference.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        target = parse_device(args.device)
        if not target.is_remote:
            raise ValueError("A policy server requires --device HOST:PORT:DEVICE (e.g. 0.0.0.0:8080:cuda).")
        host, port = target.address.rsplit(":", 1)
        if host.startswith("["):
            host = host[1:-1]
        port = int(port)
        from .utils.policy import ModelOptions, load_policy
        from .utils.policy_transport import PolicyRPCServer

        predictor = load_policy(ModelOptions(
            checkpoint=args.checkpoint,
            method=args.method,
            config_file=args.config_file,
            data_description=args.data_description,
            norm_stats_key=args.norm_stats_key,
            norm_stats_group=args.norm_stats_group,
            device=target.compute_device,
            instruction=args.instruction,
            compile=args.compile,
            amp=args.amp,
        ))
        try:
            server = PolicyRPCServer(
                predictor,
                compute_device=predictor.metadata["compute_device"],
                host=host,
                port=port,
                instruction=args.instruction,
            )
            try:
                print(f"Policy server listening on {target.address} ({server.compute_device}, {predictor.metadata['method']}).", flush=True)
                server.wait()
            except KeyboardInterrupt:
                pass
            finally:
                server.close()
        finally:
            predictor.close()
    except (ValueError, TypeError, FileNotFoundError, ImportError, RuntimeError, OSError, AssertionError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
