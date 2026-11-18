"""Compare completed serial/parallel LIBERO runs without loading a policy.

Example::

    python -m wrobo.sim_envs.LIBERO.compare_workers \
        --evaluation-dirs Eval/workers1 Eval/workers2 Eval/workers4 \
        --compare-videos --output Eval/worker_comparison.json

The first directory is the reference. Exit status is 0 only when all functional
protocol settings, model metadata and per-episode results match. Wall-clock and
inference timings are reported but do not affect reproducibility comparisons.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


EPISODE_KEY = ("suite", "task_id", "init_state_index")
EPISODE_FIELDS = (
    "task_name", "instruction", "model_seed", "steps", "success", "termination",
    "clipped_action_components", "inference_calls", "action_sha256", "observation_sha256",
)
EXECUTION_PROTOCOL_FIELDS = ("workers", "worker_start_method")


def _read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _identity(record):
    return tuple(record[name] for name in EPISODE_KEY)


def _identity_dict(key):
    return dict(zip(EPISODE_KEY, key))


def _read_run(directory):
    directory = Path(directory).expanduser().resolve()
    summary = _read_json(directory / "summary.json")
    if not isinstance(summary, dict) or summary.get("status") != "complete":
        raise ValueError("Comparison requires a completed evaluation: {}".format(directory))
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Missing evaluation protocol: {}".format(directory))
    workers = protocol.get("workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("Evaluation protocol requires a positive workers count: {}".format(directory))
    metadata = summary.get("policy_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Missing policy metadata: {}".format(directory))
    episodes = {}
    with (directory / "episodes.jsonl").open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = set(EPISODE_KEY + EPISODE_FIELDS)
            if not isinstance(record, dict) or required.difference(record):
                raise ValueError("Incomplete episode at {}:{}".format(directory, number))
            for name in ("task_id", "init_state_index", "model_seed", "steps", "inference_calls",
                         "clipped_action_components"):
                value = record[name]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("Invalid episode {} at {}:{}".format(name, directory, number))
            if (not isinstance(record["suite"], str) or not record["suite"]
                    or not isinstance(record["success"], bool) or record.get("error") is not None):
                raise ValueError("Invalid or failed episode at {}:{}".format(directory, number))
            for name in ("action_sha256", "observation_sha256"):
                if not isinstance(record[name], str) or re.fullmatch(r"[0-9a-f]{64}", record[name]) is None:
                    raise ValueError("Invalid episode digest {} at {}:{}".format(name, directory, number))
            key = _identity(record)
            if key in episodes:
                raise ValueError("Duplicate episode {} in {}".format(key, directory))
            episodes[key] = record
    overall = summary.get("overall", {})
    if (not episodes or summary.get("planned_episodes") != len(episodes)
            or overall.get("episodes") != len(episodes)
            or overall.get("successes") != sum(record["success"] for record in episodes.values())
            or overall.get("success_rate") != overall.get("successes") / len(episodes)
            or overall.get("errors") != 0 or summary.get("error") is not None):
        raise ValueError("Evaluation counts or completion status are inconsistent: {}".format(directory))
    if not isinstance(summary.get("suites"), dict) or not isinstance(summary.get("tasks"), list):
        raise ValueError("Missing per-suite/task summary: {}".format(directory))
    elapsed = summary.get("elapsed_seconds")
    if elapsed is not None and (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                                or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError("Invalid evaluation elapsed_seconds: {}".format(directory))
    return {"directory": directory, "summary": summary, "episodes": episodes}


def _differences(reference, candidate):
    differences = {}
    for name in sorted(set(reference).union(candidate)):
        if name not in reference or name not in candidate or reference[name] != candidate[name]:
            differences[name] = {"reference": reference.get(name), "candidate": candidate.get(name)}
    return differences


def _functional_protocol(run):
    return {key: value for key, value in run["summary"]["protocol"].items()
            if key not in EXECUTION_PROTOCOL_FIELDS}


def _group_results(run):
    summary = run["summary"]
    return {"overall": summary["overall"], "suites": summary["suites"],
            "tasks": sorted(summary["tasks"], key=lambda task: (task["suite"], task["task_id"]))}


def _video_digest(run, record):
    import imageio.v3 as imageio

    relative = record.get("video")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Missing episode video in {}: {}".format(run["directory"], _identity(record)))
    path = (run["directory"] / relative).resolve()
    try:
        path.relative_to(run["directory"])
    except ValueError:
        raise ValueError("Episode video is outside the evaluation directory: {}".format(path))
    digest = hashlib.sha256()
    count = 0
    reader = imageio.imiter(str(path))
    try:
        for frame in reader:
            digest.update(json.dumps({"shape": frame.shape, "dtype": str(frame.dtype)},
                                     sort_keys=True).encode("ascii"))
            digest.update(frame.tobytes(order="C"))
            count += 1
    finally:
        reader.close()
    if not count:
        raise ValueError("Episode video is empty: {}".format(path))
    return {"frames": count, "decoded_sha256": digest.hexdigest()}


def _run_description(run):
    summary = run["summary"]
    return {
        "directory": str(run["directory"]), "workers": summary["protocol"]["workers"],
        "episodes": len(run["episodes"]), "successes": summary["overall"]["successes"],
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "inference_seconds_sum": sum(record.get("inference_seconds", 0.0)
                                     for record in run["episodes"].values()),
    }


def compare_evaluations(evaluation_dirs, *, compare_videos=False):
    """Return explicit mismatches; reject incomplete/missing trace data."""
    if len(evaluation_dirs) < 2:
        raise ValueError("Provide at least two evaluation directories")
    paths = [Path(path).expanduser().resolve() for path in evaluation_dirs]
    if len(set(paths)) != len(paths):
        raise ValueError("Evaluation directories must be distinct")
    runs = [_read_run(path) for path in paths]
    reference = runs[0]
    reference_videos = ({key: _video_digest(reference, record)
                         for key, record in reference["episodes"].items()} if compare_videos else {})
    comparisons = []
    for candidate in runs[1:]:
        protocol = _differences(_functional_protocol(reference), _functional_protocol(candidate))
        metadata = _differences(reference["summary"]["policy_metadata"],
                                candidate["summary"]["policy_metadata"])
        summary_differences = _differences(_group_results(reference), _group_results(candidate))
        missing = sorted(set(reference["episodes"]).difference(candidate["episodes"]))
        extra = sorted(set(candidate["episodes"]).difference(reference["episodes"]))
        episode_differences, video_differences = [], []
        for key in sorted(set(reference["episodes"]).intersection(candidate["episodes"])):
            one, two = reference["episodes"][key], candidate["episodes"][key]
            fields = _differences({name: one[name] for name in EPISODE_FIELDS},
                                  {name: two[name] for name in EPISODE_FIELDS})
            if fields:
                episode_differences.append({"episode": _identity_dict(key), "fields": fields})
            if compare_videos:
                actual = _video_digest(candidate, two)
                if reference_videos[key] != actual:
                    video_differences.append({"episode": _identity_dict(key),
                                              "reference": reference_videos[key], "candidate": actual})
        result = _run_description(candidate)
        result.update({
            "matched": not (protocol or metadata or summary_differences or missing or extra
                            or episode_differences or video_differences),
            "protocol_differences": protocol, "policy_metadata_differences": metadata,
            "summary_differences": summary_differences,
            "missing_episodes": [_identity_dict(key) for key in missing],
            "extra_episodes": [_identity_dict(key) for key in extra],
            "episode_differences": episode_differences, "video_differences": video_differences,
        })
        elapsed, baseline_elapsed = result["elapsed_seconds"], reference["summary"].get("elapsed_seconds")
        if elapsed and baseline_elapsed is not None:
            result["measured_speedup_vs_reference"] = baseline_elapsed / elapsed
        comparisons.append(result)
    return {
        "schema_version": 1, "passed": all(result["matched"] for result in comparisons),
        "reference": _run_description(reference), "comparisons": comparisons,
        "functional_protocol": _functional_protocol(reference),
        "policy_metadata": reference["summary"]["policy_metadata"],
        "episode_fields": list(EPISODE_FIELDS), "videos_compared": bool(compare_videos),
        "reference_video_digests": [{"episode": _identity_dict(key), **value}
                                    for key, value in sorted(reference_videos.items())],
        "ignored_fields": ["protocol." + key for key in EXECUTION_PROTOCOL_FIELDS]
                          + ["elapsed_seconds", "inference_seconds"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--compare-videos", action="store_true", help="Compare decoded frames, not MP4 container bytes.")
    parser.add_argument("--output", required=True, type=Path, help="New JSON comparison report; existing files are preserved.")
    args = parser.parse_args(argv)
    try:
        result = compare_evaluations(args.evaluation_dirs, compare_videos=args.compare_videos)
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except (ValueError, TypeError, KeyError, OSError, ImportError, RuntimeError) as error:
        parser.exit(2, "error: {}\n".format(error))
    for item in result["comparisons"]:
        print("workers={}: {} ({} episodes); elapsed={}s".format(
            item["workers"], "MATCH" if item["matched"] else "MISMATCH",
            item["episodes"], item["elapsed_seconds"]))
    print("Comparison report: {}".format(destination))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
