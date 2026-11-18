"""Render frame-aligned language instructions from a unified task library.

Each episode stores one string in ``attrs["instruction"]``. Examples::

    {"episode":0,"frames":[{"start":0,"end":100,"task_id":1}]}
    {"episode":{"task_id":0,"params":{"color":"red"}}}

An integer task reference selects a ``library`` task. A JSON object selects a task
and supplies parameters for a ``template`` task. Frame ranges are inclusive.
Only structured JSON with episode/frames is accepted. Training selects channels
with instruction_mode.
"""

import json
import os
import zlib
from typing import List, Optional, Tuple, Union

import numpy as np


INSTRUCTION_ATTR = "instruction"
INSTRUCTION_MODES = (
    "episode_only", "frame_only", "episode_plus_frame",
    "frame_with_episode_fallback",
)


def normalize_instruction_modes(modes, count):
    if modes is None:
        return ["episode_only"] * count
    if not isinstance(modes, list) or len(modes) != count:
        raise ValueError("instruction_mode must be a list with one entry per dataset_dir")
    if any(mode not in INSTRUCTION_MODES for mode in modes):
        raise ValueError(f"instruction_mode entries must be one of {INSTRUCTION_MODES}")
    return list(modes)


def format_instruction_metadata(episode=None, frames=None):
    """Serialize independent episode and inclusive frame annotations as JSON."""
    metadata = {}
    if episode is not None:
        metadata["episode"] = _normalize_task_ref(episode)
    if frames is not None:
        metadata["frames"] = frames
    serialized = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    parse_instruction_segments(serialized)
    return serialized


TaskReference = Union[int, dict]
InstructionSegment = Tuple[Optional[int], Optional[int], TaskReference]



def _normalize_task_ref(task_ref: TaskReference) -> TaskReference:
    if isinstance(task_ref, bool):
        raise TypeError("Boolean values are not valid instruction task IDs")
    if isinstance(task_ref, (int, np.integer)):
        return int(task_ref)
    if not isinstance(task_ref, dict):
        raise TypeError(
            "Instruction task reference must be an integer or a JSON object with "
            "'task_id' and optional 'params'"
        )

    unexpected = set(task_ref) - {"task_id", "params"}
    if "task_id" not in task_ref or unexpected:
        raise ValueError(
            "Instruction task object must contain 'task_id' and optional 'params'; "
            f"got keys {sorted(task_ref)}"
        )
    task_id = task_ref["task_id"]
    if isinstance(task_id, bool) or not isinstance(task_id, (int, np.integer)):
        raise TypeError("Instruction 'task_id' must be an integer")
    params = task_ref.get("params")
    if params is not None and not isinstance(params, dict):
        raise TypeError("Instruction 'params' must be a JSON object")
    normalized = {"task_id": int(task_id)}
    if params is not None:
        normalized["params"] = params
    return normalized


class UnifiedInstructionProvider:
    """Resolve episode frame metadata against ``instruction_library.json``."""

    def __init__(self, library_path: str):
        with open(library_path, "r", encoding="utf-8") as file:
            library = json.load(file)
        if not isinstance(library, dict):
            raise TypeError(
                f"Instruction library must be a JSON object keyed by task ID: {library_path}"
            )
        self.library_path = library_path
        self.library = library
        self.task_map = {int(task_id): task for task_id, task in library.items()}

    def get(
        self,
        instruction_str: str,
        frame_idx: int,
        episode_id: str = "",
        pick_random: bool = True,
    ) -> str:
        """Return the natural-language instruction active at ``frame_idx``."""
        segments = self._parse_segments(instruction_str)
        selected = None
        for start, end, task_ref in segments:
            if start is None or start <= frame_idx <= end:
                selected = (start, end, task_ref)
                break
        if selected is None:
            raise ValueError(
                f"No instruction segment covers frame {frame_idx} in {instruction_str!r}"
            )

        start, end, task_ref = selected
        if pick_random:
            # Uses NumPy's worker-seeded global RNG, preserving dataloader reproducibility.
            rng = np.random
        else:
            stable_key = f"{episode_id}:{start}:{end}:{json.dumps(task_ref, sort_keys=True)}"
            rng = np.random.default_rng(zlib.crc32(stable_key.encode("utf-8")))
        return self.sample_instruction(task_ref, rng=rng)

    def get_channels(self, instruction_str, frame_idx, mode, episode_id="", pick_random=True):
        """Return selected channels independently; absent channels are None."""
        if mode not in INSTRUCTION_MODES:
            raise ValueError(f"Unknown instruction mode: {mode}")
        segments = self._parse_segments(instruction_str)
        episode = next((s for s in segments if s[0] is None), None)
        frame = next((s for s in segments if s[0] is not None and s[0] <= frame_idx <= s[1]), None)
        selected = {}
        if mode in ("episode_only", "episode_plus_frame"):
            selected["episode"] = episode
        if mode in ("frame_only", "episode_plus_frame"):
            selected["frame"] = frame
        if mode == "frame_with_episode_fallback":
            selected["frame" if frame is not None else "episode"] = frame or episode
        result = {"episode": None, "frame": None}
        for channel, segment in selected.items():
            if segment is None:
                raise ValueError(f"Missing {channel} instruction for episode {episode_id}, frame {frame_idx}, mode {mode}")
            start, end, task_ref = segment
            rng = np.random
            if not pick_random:
                stable_key = f"{episode_id}:{start}:{end}:{json.dumps(task_ref, sort_keys=True)}"
                rng = np.random.default_rng(zlib.crc32(stable_key.encode("utf-8")))
            result[channel] = self.sample_instruction(task_ref, rng=rng)
        return result

    @staticmethod
    def _parse_segments(instruction_str: str) -> List[InstructionSegment]:
        """Parse only JSON episode/frames metadata into validated task references."""
        if isinstance(instruction_str, (bytes, np.bytes_)):
            instruction_str = instruction_str.decode("utf-8")
        if not isinstance(instruction_str, (str, np.str_)):
            raise TypeError(
                "Episode instruction metadata must be a string in unified format, "
                f"got {type(instruction_str).__name__}"
            )
        instruction_str = str(instruction_str).strip()
        if not instruction_str:
            raise ValueError("Episode instruction metadata is empty")

        try:
            metadata = json.loads(instruction_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Instruction metadata must be a JSON object with episode and/or frames"
            ) from exc
        if not isinstance(metadata, dict):
            raise TypeError("Instruction metadata must be a JSON object")
        if set(metadata) - {"episode", "frames"}:
            raise ValueError("Instruction metadata supports only episode and frames")
        segments = []
        if metadata.get("episode") is not None:
            segments.append((None, None, _normalize_task_ref(metadata["episode"])))
        frames = metadata.get("frames", [])
        if not isinstance(frames, list):
            raise TypeError("frames must be a list")
        for frame in frames:
            if not isinstance(frame, dict) or not {"start", "end", "task_id"} <= set(frame):
                raise ValueError("Each frame annotation needs start, end and task_id")
            start, end = frame["start"], frame["end"]
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)):
                raise TypeError("Frame bounds must be integers")
            if start < 0 or end < start:
                raise ValueError("Invalid inclusive frame range")
            ref = {k: v for k, v in frame.items() if k not in ("start", "end")}
            segments.append((start, end, _normalize_task_ref(ref)))
        if not segments:
            raise ValueError("Episode instruction metadata is empty")
        ranges = sorted((s, e) for s, e, _ in segments if s is not None)
        if any(b <= e for (_, e), (b, _) in zip(ranges, ranges[1:])):
            raise ValueError("Instruction frame ranges must not overlap")
        return segments

    def sample_instruction(self, task_ref: TaskReference, rng=None) -> str:
        """Sample or render one already-resolved task reference."""
        task_ref = _normalize_task_ref(task_ref)
        if isinstance(task_ref, int):
            task_id = task_ref
            params = None
        else:
            task_id = task_ref["task_id"]
            params = task_ref.get("params")

        if task_id not in self.task_map:
            raise KeyError(f"Task ID {task_id} not found in {self.library_path}")
        task = self.task_map[task_id]
        task_type = task.get("type")

        if task_type == "library":
            choices = task.get("instructions", [])
        elif task_type == "template":
            if params is None:
                raise ValueError(f"Template task {task_id} requires params")
            choices = task.get("templates", [])
        else:
            raise ValueError(f"Unknown task type for task {task_id}: {task_type!r}")
        if not choices:
            raise ValueError(f"Task {task_id} has no instruction choices")

        choice = choices[0] if rng is None else str(rng.choice(choices))
        if task_type == "library":
            return choice
        try:
            return choice.format(**params)
        except KeyError as exc:
            raise ValueError(
                f"Template task {task_id} requires missing parameter {exc.args[0]!r}"
            ) from exc

    def get_all_sentences(
        self, task_id: int, params: Optional[dict] = None
    ) -> List[str]:
        """Return all possible renderings for vocabulary construction."""
        task_id = int(task_id)
        if task_id not in self.task_map:
            raise KeyError(f"Task ID {task_id} not found in {self.library_path}")
        task = self.task_map[task_id]
        task_type = task.get("type")
        if task_type == "library":
            return list(task.get("instructions", []))
        if task_type == "template":
            if params is None:
                raise ValueError(f"Template task {task_id} requires params")
            return [template.format(**params) for template in task.get("templates", [])]
        raise ValueError(f"Unknown task type for task {task_id}: {task_type!r}")


def build_instruction_provider(dataset_dir: str) -> UnifiedInstructionProvider:
    """Build a provider, requiring the dataset's unified instruction library."""
    library_path = os.path.join(dataset_dir, "instruction_library.json")
    if not os.path.isfile(library_path):
        raise FileNotFoundError(
            f"Language is enabled but instruction library was not found: {library_path}"
        )
    return UnifiedInstructionProvider(library_path)


def parse_instruction_segments(instruction_str: str) -> List[InstructionSegment]:
    """Parse and validate metadata without loading an instruction library."""
    return UnifiedInstructionProvider._parse_segments(instruction_str)
