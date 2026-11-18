"""Evaluate a separately served wrobo policy in the LIBERO Python environment.

The simulator process needs LIBERO, NumPy and tqdm. Model dependencies stay in
the policy server. Images follow the converter's 180-degree convention and
actions remain native OSC_POSE commands.
"""

import argparse
from collections import deque
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import random
import signal
import sys
import time

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")


SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
IMAGE_SIZE = 256
SETTLING_STEPS = 10
ENV_SEED = 319
DUMMY_ACTION = [0.0] * 6 + [-1.0]
CAMERAS = {"image_top": "agentview_image", "image_wrist": "robot0_eye_in_hand_image"}
STATE_DIMENSIONS = {"proprio_ee": 8, "proprio_joint": 9, "proprio_states": 9}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def add_rollout_arguments(parser):
    """Share rollout options with the model-side single-command evaluator."""
    parser.add_argument("--suites", nargs="+", choices=tuple(SUITE_MAX_STEPS), default=["libero_spatial"])
    parser.add_argument("--task-ids", nargs="+", type=nonnegative_int,
                        help="Official task indices in task_order_index=0; allowed with one suite only.")
    parser.add_argument("--num-trials-per-task", type=positive_int, default=50,
                        help="Use the first N official initial states for every task (default: 50).")
    parser.add_argument("--replan-steps", type=positive_int, default=5,
                        help="Execute this many predicted actions before requesting a fresh chunk.")
    parser.add_argument("--workers", type=positive_int, default=1,
                        help="Independent simulator processes sharing one model server (default: 1).")
    parser.add_argument("--seed", type=nonnegative_int, default=319,
                        help="Base model sampling seed (default: 319); simulator seed remains 319.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--save-video", action="store_true",
                        help="Save both rotated camera views side by side as MP4; requires imageio/ffmpeg.")
    parser.add_argument("--timeout", type=positive_float, default=120.0,
                        help="Timeout in seconds for each model server request.")
    return parser


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    return add_rollout_arguments(parser)


def validate_metadata(metadata, replan_steps):
    """Reject a mismatched policy before moving the simulated robot."""
    if not isinstance(metadata, dict):
        raise ValueError("Policy metadata must be a dictionary")
    key = metadata.get("proprio_key")
    if key not in STATE_DIMENSIONS or metadata.get("state_dim") != STATE_DIMENSIONS[key]:
        raise ValueError("LIBERO requires proprio_ee (8), proprio_joint (9), or proprio_states (9)")
    if metadata.get("proprio_path") != "observations/" + key:
        raise ValueError("Policy proprio_path does not match proprio_key")
    if metadata.get("action_key") != "action_rel" or metadata.get("action_dim") != 7:
        raise ValueError("LIBERO evaluation requires native action_rel with seven components")
    if metadata.get("history_width") != 1:
        raise ValueError("LIBERO evaluator currently requires history_width=1")
    cameras = metadata.get("camera_keys")
    if (not isinstance(cameras, list) or not cameras or len(set(cameras)) != len(cameras)
            or any(name not in CAMERAS for name in cameras)):
        raise ValueError("LIBERO cameras must be a nonempty selection of image_top and image_wrist")
    chunk = metadata.get("action_chunk_size")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < replan_steps:
        raise ValueError("replan_steps must not exceed the policy action_chunk_size")


def observation_window(obs, metadata, instruction):
    """Match converter state order and image orientation, without normalizing."""
    # This helper is NumPy-only and is also used by the dataset regeneration.
    from wrobo.dataset_format_convert.libero_replay import _observation_arrays

    arrays = _observation_arrays(obs)
    key = metadata["proprio_key"]
    if key == "proprio_ee":
        state = np.concatenate((arrays["obs/ee_states"], arrays["obs/gripper_states"]))
    elif key == "proprio_joint":
        state = np.concatenate((arrays["obs/joint_states"], arrays["obs/gripper_states"]))
    else:
        state = arrays["robot_states"]
    if state.shape != (metadata["state_dim"],) or not np.isfinite(state).all():
        raise ValueError("LIBERO observation contains an invalid proprioceptive state")
    images = {}
    for camera in metadata["camera_keys"]:
        frame = np.asarray(obs[CAMERAS[camera]])
        if frame.dtype != np.uint8 or frame.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError("LIBERO cameras must render uint8 RGB 256x256 frames")
        images[camera] = np.ascontiguousarray(frame[::-1, ::-1])[None]
    return {"state": np.ascontiguousarray(state, dtype=np.float32)[None],
            "images": images, "instruction": instruction}


def make_suite(name):
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[name](task_order_index=0)


def make_environment(task):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if not bddl.is_file():
        raise FileNotFoundError("Task BDDL does not exist: {}".format(bddl))
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl), controller="OSC_POSE", robots=["Panda"],
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE,
        control_freq=20, hard_reset=True,
    )


def _action_bounds(env):
    # LIBERO's ControlEnv exposes robosuite as .env, not action_spec itself.
    low, high = getattr(env, "env", env).action_spec
    low, high = np.asarray(low), np.asarray(high)
    if (low.shape != (7,) or high.shape != (7,) or not np.isfinite(low).all()
            or not np.isfinite(high).all() or not (low < high).all()):
        raise ValueError("LIBERO controller must provide finite seven-dimensional action bounds")
    return low, high


def _video_writer(path):
    import imageio.v2 as imageio

    return imageio.get_writer(str(path), fps=20, ffmpeg_params=["-threads", "1"])


def _video_frame(obs):
    return np.ascontiguousarray(np.concatenate(
        [np.asarray(obs[key])[::-1, ::-1] for key in CAMERAS.values()], axis=1))


def _episode_seed(seed, suite_name, task_id, init_state_index):
    # Stable across suite/task subsets and Python's randomized hash seed.
    suite_id = tuple(SUITE_MAX_STEPS).index(suite_name)
    return int(np.random.SeedSequence([seed, suite_id, task_id, init_state_index]).generate_state(1)[0])


def _hash_observation(digest, window, camera_keys):
    """Hash exactly the rotated images and float32 state supplied to the policy."""
    digest.update(np.asarray(window["state"], dtype="<f4").tobytes())
    for camera in camera_keys:
        frame = window["images"][camera]
        digest.update(camera.encode("utf-8") + b"\0")
        digest.update(np.asarray(frame.shape, dtype="<i8").tobytes())
        digest.update(frame.tobytes())


def _check_stopped(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Evaluation cancelled")


def _interrupt(signum, frame):
    raise KeyboardInterrupt()


def rollout(env, client, initial_state, instruction, *, max_steps, replan_steps,
            model_seed, record, video_path=None, video_factory=None, stop_event=None):
    """Execute a single episode; populate record even when a step raises."""
    writer = None
    action_digest, observation_digest = hashlib.sha256(), hashlib.sha256()
    record.update(success=False, steps=0, inference_calls=0, inference_seconds=0.0,
                  clipped_action_components=0, termination="error", error=None)
    try:
        _check_stopped(stop_event)
        client.reset(model_seed)
        env.seed(ENV_SEED)
        env.reset()
        obs = env.set_init_state(initial_state)
        for _ in range(SETTLING_STEPS):
            _check_stopped(stop_event)
            obs, _, _, _ = env.step(DUMMY_ACTION)
        low, high = _action_bounds(env)
        if video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = (video_factory or _video_writer)(video_path)
        actions = deque()
        for step in range(max_steps):
            _check_stopped(stop_event)
            window = observation_window(obs, client.metadata, instruction)
            _hash_observation(observation_digest, window, client.metadata["camera_keys"])
            if writer is not None:
                writer.append_data(_video_frame(obs))
            if not actions:
                started = time.perf_counter()
                prediction = np.asarray(client.predict(window))
                record["inference_seconds"] += time.perf_counter() - started
                record["inference_calls"] += 1
                if (prediction.ndim != 2 or prediction.shape[1] != 7
                        or not replan_steps <= prediction.shape[0] <= client.metadata["action_chunk_size"]
                        or prediction.dtype.kind not in "fiu" or not np.isfinite(prediction).all()):
                    raise ValueError("Policy returned an invalid finite (T,7) action chunk")
                actions.extend(np.array(prediction[:replan_steps], copy=True))
            action = actions.popleft()
            clipped = np.clip(action, low, high)
            record["clipped_action_components"] += int(np.count_nonzero(clipped != action))
            obs, _, done, _ = env.step(clipped.tolist())
            action_digest.update(np.asarray(clipped, dtype="<f4").tobytes())
            record["steps"] = step + 1
            if env.check_success():
                record.update(success=True, termination="success")
                break
            if done:
                record["termination"] = "environment_done"
                break
        else:
            record["termination"] = "step_limit"
        _hash_observation(observation_digest, observation_window(obs, client.metadata, instruction),
                          client.metadata["camera_keys"])
        if writer is not None:
            writer.append_data(_video_frame(obs))
        return record
    except BaseException as error:
        record["success"] = False
        record["error"] = "{}: {}".format(type(error).__name__, error)
        record["termination"] = "interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "error"
        raise
    finally:
        record["action_sha256"] = action_digest.hexdigest()
        record["observation_sha256"] = observation_digest.hexdigest()
        if writer is not None:
            try:
                writer.close()
            except BaseException as error:
                message = "{}: {}".format(type(error).__name__, error)
                if record["error"] is not None:
                    # Preserve the original inference/simulation exception.
                    record["video_close_error"] = message
                else:
                    record.update(success=False, error=message, termination="video_error")
                    raise


def _evaluate_task(plan, options, client, emit, *, env_factory=None, video_factory=None,
                   stop_event=None):
    """Reuse one simulator for a task; emit each attempted episode immediately."""
    task_index, suite_name, task_id, task, initial_states = plan
    env = None
    try:
        _check_stopped(stop_event)
        # Some model properties are sampled during construction, before env.seed
        # is available. Seed every task, independently of worker assignment.
        random.seed(ENV_SEED)
        np.random.seed(ENV_SEED)
        env = (env_factory or make_environment)(task)
        for init_index, initial_state in enumerate(initial_states):
            _check_stopped(stop_event)
            record = {"suite": suite_name, "task_id": task_id, "task_name": task.name,
                      "instruction": str(task.language), "init_state_index": init_index,
                      "model_seed": _episode_seed(options["seed"], suite_name, task_id, init_index)}
            video = None
            if options["save_video"]:
                video = (Path(options["output_dir"]) / "videos" / suite_name
                         / "task_{:02d}_trial_{:03d}.mp4".format(task_id, init_index))
                record["video"] = str(video.relative_to(options["output_dir"]))
            try:
                rollout(env, client, initial_state, str(task.language),
                        max_steps=options["max_steps"][suite_name], replan_steps=options["replan_steps"],
                        model_seed=record["model_seed"], record=record,
                        video_path=video, video_factory=video_factory, stop_event=stop_event)
            finally:
                emit(task_index * options["num_trials_per_task"] + init_index, record)
    finally:
        if env is not None:
            env.close()


def _simulation_worker(worker_id, tasks, results, stop_event, options, env_factory, video_factory):
    """Spawn entry point: simulator-only dependencies, no model or CUDA tensors."""
    from .transport import PolicyClient

    # The parent coordinates Ctrl-C and gives writers/environments time to close.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _interrupt)
    client = None
    failed = False
    try:
        client = PolicyClient(options["server_url"], timeout=options["timeout"])
        validate_metadata(client.metadata, options["replan_steps"])
        while not stop_event.is_set():
            try:
                plan = tasks.get(timeout=0.2)
            except queue.Empty:
                continue
            if plan is None:
                break
            _evaluate_task(plan, options, client,
                           lambda index, record: results.put(("record", index, record)),
                           env_factory=env_factory, video_factory=video_factory, stop_event=stop_event)
    except InterruptedError:
        if not stop_event.is_set():
            failed = True
            stop_event.set()
            results.put(("error", worker_id, "InterruptedError: Evaluation interrupted"))
    except BaseException as error:
        failed = True
        stop_event.set()
        results.put(("error", worker_id, "{}: {}".format(type(error).__name__, error)))
    finally:
        if client is not None:
            try:
                client.close()
            except BaseException as error:
                if not failed:
                    stop_event.set()
                    results.put(("error", worker_id, "{}: {}".format(type(error).__name__, error)))
        results.put(("done", worker_id, None))


def _stop_workers(processes, stop_event, results, emit):
    """Cooperate first, then stop blocked workers without leaking children."""
    stop_event.set()
    deadline = time.monotonic() + 5.0
    while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
        try:
            kind, index, payload = results.get(timeout=0.05)
        except queue.Empty:
            continue
        if kind == "record":
            emit(index, payload)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
    # Queue feeder threads can deliver a final completed episode after process exit.
    while True:
        try:
            kind, index, payload = results.get_nowait()
        except queue.Empty:
            break
        if kind == "record":
            emit(index, payload)


def _evaluate_parallel(plans, options, workers, emit, *, env_factory=None, video_factory=None):
    context = multiprocessing.get_context("spawn")
    tasks, results = context.Queue(), context.Queue()
    stop_event = context.Event()
    processes = []
    try:
        for plan in plans:
            tasks.put(plan)
        for _ in range(workers):
            tasks.put(None)
        for worker_id in range(workers):
            process = context.Process(
                target=_simulation_worker,
                args=(worker_id, tasks, results, stop_event, options, env_factory, video_factory),
                name="libero-sim-{}".format(worker_id),
            )
            process.start()
            processes.append(process)
        done, exited = set(), {}
        while len(done) != len(processes):
            try:
                kind, index, payload = results.get(timeout=0.2)
            except queue.Empty:
                now = time.monotonic()
                for index, process in enumerate(processes):
                    if process.exitcode is not None and index not in done:
                        if process.exitcode != 0 or now - exited.setdefault(index, now) > 1.0:
                            raise RuntimeError("LIBERO worker {} exited unexpectedly (exit code {})".format(
                                index, process.exitcode))
                continue
            if kind == "record":
                emit(index, payload)
            elif kind == "error":
                raise RuntimeError("LIBERO worker {} failed: {}".format(index, payload))
            elif kind == "done":
                done.add(index)
            else:
                raise RuntimeError("Unknown LIBERO worker message: {}".format(kind))
    finally:
        _stop_workers(processes, stop_event, results, emit)
        for channel in (tasks, results):
            channel.cancel_join_thread()
            channel.close()


def _group_summary(records):
    successes = sum(int(record["success"]) for record in records)
    return {"episodes": len(records), "successes": successes,
            "success_rate": successes / len(records) if records else None,
            "errors": sum(record.get("error") is not None for record in records)}


def _write_summary(path, records, protocol, metadata, planned, status, error, elapsed_seconds):
    tasks, suites = {}, {}
    for record in records:
        suites.setdefault(record["suite"], []).append(record)
        tasks.setdefault((record["suite"], record["task_id"]), []).append(record)
    task_results = []
    for (suite, task_id), episodes in tasks.items():
        task_results.append(dict(suite=suite, task_id=task_id,
                                 task_name=episodes[0]["task_name"], **_group_summary(episodes)))
    total = _group_summary(records)
    # A partial run is never presented as a completed benchmark percentage.
    if status != "complete":
        total["observed_success_rate"] = total["success_rate"]
        total["success_rate"] = None
    result = {"schema_version": 1, "status": status, "error": error,
              "elapsed_seconds": elapsed_seconds,
              "planned_episodes": planned, "protocol": protocol,
              "policy_metadata": metadata, "overall": total,
              "suites": {name: _group_summary(items) for name, items in suites.items()},
              "tasks": task_results}
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def evaluate(args, *, client=None, suite_factory=None, env_factory=None,
             progress_factory=None, video_factory=None):
    """Run selected official episodes and save a report; factories support tests."""
    started = time.perf_counter()
    if isinstance(args.workers, bool) or not isinstance(args.workers, int) or args.workers < 1:
        raise ValueError("workers must be a positive integer")
    if len(set(args.suites)) != len(args.suites):
        raise ValueError("suites must be unique")
    if args.task_ids is not None and len(args.suites) != 1:
        raise ValueError("--task-ids requires exactly one suite")
    if args.task_ids is not None and len(set(args.task_ids)) != len(args.task_ids):
        raise ValueError("task_ids must be unique")
    if args.seed > 2 ** 32 - 1:
        raise ValueError("seed must fit an unsigned 32-bit integer")
    destination = Path(args.output_dir).expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("Evaluation output must be absent or empty: {}".format(destination))
    if client is None:
        from .transport import PolicyClient

        client = PolicyClient(args.server_url, timeout=args.timeout)
    progress = None
    records = []
    record_order = {}
    planned = 0
    protocol = {}
    status, failure = "failed", None
    report_started = False
    try:
        validate_metadata(client.metadata, args.replan_steps)
        task_plans = []
        for suite_name in args.suites:
            suite = (suite_factory or make_suite)(suite_name)
            task_ids = list(range(suite.n_tasks)) if args.task_ids is None else list(args.task_ids)
            for task_id in task_ids:
                if not 0 <= task_id < suite.n_tasks:
                    raise ValueError("Task index {} is outside {}".format(task_id, suite_name))
                task = suite.get_task(task_id)
                initial_states = suite.get_task_init_states(task_id)
                if args.num_trials_per_task > len(initial_states):
                    raise ValueError("Requested {} trials but {}/{} has only {} official initial states".format(
                        args.num_trials_per_task, suite_name, task.name, len(initial_states)))
                initial_states = [np.asarray(state).copy()
                                  for state in initial_states[:args.num_trials_per_task]]
                task_plans.append((len(task_plans), suite_name, task_id, task, initial_states))
        planned = len(task_plans) * args.num_trials_per_task
        if not planned:
            raise ValueError("No LIBERO tasks selected")
        protocol = {
            "suites": list(args.suites), "task_order_index": 0,
            "task_ids": args.task_ids, "num_trials_per_task": args.num_trials_per_task,
            "initial_state_indices": list(range(args.num_trials_per_task)),
            "initial_state_source": "official get_task_init_states",
            "seed": args.seed, "environment_seed": ENV_SEED,
            "environment_construction_seed": ENV_SEED,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "model_seed_derivation": "numpy.SeedSequence([seed,suite_index,task_id,init_state_index]).generate_state(1)[0]",
            "suite_index_order": list(SUITE_MAX_STEPS),
            "replan_steps": args.replan_steps, "settling_steps": SETTLING_STEPS,
            "dummy_action": DUMMY_ACTION, "image_size": IMAGE_SIZE,
            "image_orientation": "rotate_180_from_renderer", "hard_reset": True,
            "control_freq": 20, "controller": "OSC_POSE", "action_clipping": "environment action_spec",
            "success_criterion": "first env.check_success() after a policy action",
            "max_steps": {name: SUITE_MAX_STEPS[name] for name in args.suites},
            "save_video": args.save_video,
            "workers": args.workers,
            "trajectory_hashes": {
                "action_sha256": "completed clipped policy actions as little-endian float32",
                "observation_sha256": "policy-step observations and final observation; state float32, rotated RGB cameras",
            },
        }
        destination.mkdir(parents=True, exist_ok=True)
        report_started = True
        if progress_factory is None:
            from tqdm import tqdm

            progress_factory = tqdm
        progress = progress_factory(total=planned, desc="Evaluating LIBERO", unit="episode",
                                    dynamic_ncols=True)
        with (destination / "episodes.jsonl").open("x", encoding="utf-8") as stream:
            def emit(index, record):
                if index in record_order:
                    raise RuntimeError("Duplicate LIBERO episode result: {}".format(index))
                record_order[index] = record
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                successes = sum(int(item["success"]) for item in records)
                progress.set_postfix(success="{}/{} ({:.1%})".format(
                    successes, len(records), successes / len(records)), refresh=False)
                progress.update(1)

            options = {
                "seed": args.seed, "save_video": args.save_video,
                "output_dir": str(destination), "max_steps": dict(SUITE_MAX_STEPS),
                "num_trials_per_task": args.num_trials_per_task,
                "replan_steps": args.replan_steps, "server_url": args.server_url,
                "timeout": args.timeout,
            }
            try:
                if args.workers == 1:
                    for plan in task_plans:
                        _evaluate_task(plan, options, client, emit,
                                       env_factory=env_factory, video_factory=video_factory)
                else:
                    _evaluate_parallel(task_plans, options, args.workers, emit,
                                       env_factory=env_factory, video_factory=video_factory)
                if len(records) != planned:
                    raise RuntimeError("LIBERO evaluation ended before all planned episodes completed")
            finally:
                # Completion order varies with scheduling; persisted order does not.
                records[:] = [record_order[index] for index in sorted(record_order)]
                stream.seek(0)
                stream.truncate()
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
        status = "complete"
    except BaseException as error:
        failure = "{}: {}".format(type(error).__name__, error)
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        raise
    finally:
        try:
            if report_started:
                _write_summary(destination / "summary.json", records, protocol,
                               client.metadata, planned, status, failure, time.perf_counter() - started)
        finally:
            if progress is not None:
                progress.close()
            client.close()
    return json.loads((destination / "summary.json").read_text(encoding="utf-8"))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    previous_handler = signal.signal(signal.SIGTERM, _interrupt)
    try:
        try:
            result = evaluate(args)
        except KeyboardInterrupt:
            parser.exit(130, "Evaluation interrupted; any completed episode records were preserved.\n")
        except (ValueError, TypeError, FileNotFoundError, ImportError, RuntimeError, OSError) as error:
            parser.exit(1, "error: {}\n".format(error))
        print("LIBERO success: {}/{} ({:.2%}); report: {}".format(
            result["overall"]["successes"], result["overall"]["episodes"],
            result["overall"]["success_rate"], Path(args.output_dir) / "summary.json"))
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _module_main():
    """Start the CLI with the same CPython hash seed as spawned simulators."""
    parser = build_parser()
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("seed must fit an unsigned 32-bit integer")
    hash_seed = str(args.seed)
    if os.environ.get("PYTHONHASHSEED") != hash_seed:
        os.environ["PYTHONHASHSEED"] = hash_seed
        os.execv(sys.executable, [sys.executable, "-m", "wrobo.sim_envs.LIBERO.eval", *sys.argv[1:]])
        raise RuntimeError("Failed to restart the evaluator with its Python hash seed")
    return main()


if __name__ == "__main__":
    raise SystemExit(_module_main())
