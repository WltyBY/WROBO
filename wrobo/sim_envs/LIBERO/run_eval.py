"""Load a trained policy and run LIBERO evaluation with one command.

Run this entry in the model environment. It serves the model locally while a
separate conda process runs the existing LIBERO evaluator, then releases both.
"""

import argparse
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import threading

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wrobo.methods.registry import METHOD_TO_TRAINER
from wrobo.sim_envs.LIBERO.eval import add_rollout_arguments, validate_metadata
from wrobo.sim_envs.LIBERO.transport import make_server


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_gpu_index(value):
    index = int(value)
    if index < -1:
        raise argparse.ArgumentTypeError("gpu must be a single CUDA index or -1 for a CPU policy")
    return index


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path,
                        help="Exact training output directory containing the checkpoint, norm_stats.json and Config_and_Trainer/.")
    parser.add_argument("--method", required=True, type=str.upper, choices=tuple(METHOD_TO_TRAINER))
    parser.add_argument("--checkpoint-name", default="checkpoint_final.pth")
    parser.add_argument("--libero-env", default="libero", help="Conda environment running the simulator.")
    parser.add_argument("--conda-executable", help="Conda executable; defaults to CONDA_EXE or PATH.")
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--gpu", type=parse_gpu_index, default=0,
                        help="CUDA GPU ID (default: 0); -1 runs the policy on CPU. Evaluation uses one model GPU.")
    device.add_argument("--device", help="Explicit policy device, e.g. cpu, mps or cuda:1; alternative to --gpu.")
    parser.add_argument("--norm-stats-group", help="Explicit statistics group for a per-dataset statistics file.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return add_rollout_arguments(parser)


def configure_device(args):
    """Select the policy GPU before importing PyTorch or initializing CUDA."""
    if args.device is not None:
        return args.device
    if args.gpu == -1:
        # The simulator may still need EGL; keep its existing CUDA visibility.
        return "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return "cuda:0"


def resolve_artifacts(log_dir, checkpoint_name="checkpoint_final.pth"):
    """Read artifacts from the specified training run, without searching parents."""
    root = Path(log_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Training directory does not exist: {}".format(root))
    if (not isinstance(checkpoint_name, str) or not checkpoint_name
            or Path(checkpoint_name).name != checkpoint_name or checkpoint_name in (".", "..")):
        raise ValueError("checkpoint-name must be a filename")
    checkpoint = root / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Checkpoint not found: {}. --log-dir must be the exact training output directory, "
            "including the run name, method settings and fold directory.".format(checkpoint)
        )
    run = root
    artifact_dir = run / "Config_and_Trainer"
    configs = sorted(list(artifact_dir.glob("*.yaml")) + list(artifact_dir.glob("*.yml")))
    stats = run / "norm_stats.json"
    if len(configs) != 1 or not stats.is_file():
        raise ValueError("Expected one saved training YAML and norm_stats.json in {}".format(run))
    return {
        "checkpoint": str(checkpoint), "config_file": str(configs[0]),
        "norm_stats_key": str(stats), "train_run_dir": str(run),
    }


def build_simulator_command(args, server_url, conda_executable):
    command = [str(conda_executable), "run", "--no-capture-output", "-n", args.libero_env,
               "python", "-u", "-m", "wrobo.sim_envs.LIBERO.eval", "--server-url", server_url,
               "--suites", *args.suites, "--num-trials-per-task", str(args.num_trials_per_task),
               "--replan-steps", str(args.replan_steps), "--seed", str(args.seed),
               "--workers", str(args.workers),
               "--output-dir", str(Path(args.output_dir).expanduser().resolve()),
               "--timeout", str(args.timeout)]
    if args.task_ids is not None:
        command.extend(["--task-ids", *map(str, args.task_ids)])
    if args.save_video:
        command.append("--save-video")
    return command


def _reset_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stop_process(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The conda wrapper may have exited before one of its descendants.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_evaluation(args, *, policy_loader=None, server_factory=None,
                   process_factory=None, seed_callback=None):
    """Load one model, serve it locally, and wait for the simulator process."""
    if len(set(args.suites)) != len(args.suites):
        raise ValueError("suites must be unique")
    if args.task_ids is not None and (len(args.suites) != 1 or len(set(args.task_ids)) != len(args.task_ids)):
        raise ValueError("task-ids must be unique and require one suite")
    if args.seed >= 2**32:
        raise ValueError("seed must fit an unsigned 32-bit integer")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Evaluation output must be absent or empty: {}".format(output))
    artifacts = resolve_artifacts(args.log_dir, args.checkpoint_name)
    conda = args.conda_executable or os.environ.get("CONDA_EXE") or "conda"
    conda = shutil.which(conda)
    if conda is None:
        raise FileNotFoundError("Conda executable not found; set --conda-executable")

    device = configure_device(args)
    from wrobo.real_envs.utils.policy import ModelOptions, load_policy

    reset_seed = seed_callback or _reset_seed
    reset_seed(args.seed)
    predictor = (policy_loader or load_policy)(ModelOptions(
        checkpoint=artifacts["checkpoint"], config_file=artifacts["config_file"],
        norm_stats_key=artifacts["norm_stats_key"], norm_stats_group=args.norm_stats_group,
        method=args.method, device=device, amp=args.amp, compile=args.compile,
    ))
    server, thread, process = None, None, None
    try:
        validate_metadata(predictor.metadata, args.replan_steps)
        predictor.metadata.update({
            "checkpoint": artifacts["checkpoint"], "config_file": artifacts["config_file"],
            "norm_stats_file": artifacts["norm_stats_key"], "norm_stats_group": args.norm_stats_group,
            "amp": args.amp, "compile": args.compile,
        })
        server = (server_factory or make_server)(
            predictor, host="127.0.0.1", port=0, reset_callback=reset_seed,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = "http://127.0.0.1:{}".format(server.server_port)
        command = build_simulator_command(args, url, conda)
        environment = os.environ.copy()
        environment["MUJOCO_GL"] = environment.get("MUJOCO_GL", "egl")
        if args.device is None and args.gpu >= 0:
            environment["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
        # CPython reads this at interpreter startup; spawned simulators must
        # inherit the same hash ordering as their simulation parent.
        environment["PYTHONHASHSEED"] = str(args.seed)
        environment["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        for name in ("NO_PROXY", "no_proxy"):
            environment[name] = environment.get(name, "") + ",127.0.0.1,localhost"
        print("Evaluating checkpoint: {}".format(artifacts["checkpoint"]), flush=True)
        print("LIBERO results: {}".format(output), flush=True)
        process = (process_factory or subprocess.Popen)(
            command, cwd=str(PROJECT_ROOT), env=environment, start_new_session=True,
        )
        return process.wait()
    finally:
        try:
            _stop_process(process)
        finally:
            try:
                if server is not None:
                    if thread is not None and thread.is_alive():
                        server.shutdown()
                    server.server_close()
                    if thread is not None:
                        thread.join(timeout=5)
            finally:
                predictor.close()


class _SignalExit(BaseException):
    def __init__(self, signum):
        self.signum = signum


def _terminate(signum, frame):
    raise _SignalExit(signum)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _terminate)
    try:
        result = run_evaluation(args)
        return result if result >= 0 else 128 - result
    except _SignalExit as error:
        return 128 + error.signum
    except KeyboardInterrupt:
        return 130
    except (ValueError, TypeError, FileNotFoundError, ImportError, RuntimeError, OSError) as error:
        parser.exit(1, "error: {}\n".format(error))
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
