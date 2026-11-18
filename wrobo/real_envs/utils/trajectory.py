"""Stream native LeRobot observations and actions into wrobo episode HDF5 files."""

from collections.abc import Mapping
from enum import Enum
import json
import logging
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import tempfile
from threading import Event, Thread

import h5py
import numpy as np

from wrobo.utils.image_codec import ENCODING_ATTR, encode_frame


logger = logging.getLogger(__name__)
_OWNED_ATTRIBUTES = {
    "FPS", "seq_len", "success", "comment", "state_keys", "action_keys",
    "robot_config", "teleop_config", "fps", "interrupted",
}


def _normalize_metadata_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _normalize_metadata_value(value.value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _normalize_metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata_value(item) for item in value]
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        string_type = h5py.check_string_dtype(value.dtype)
        if string_type is not None and string_type.encoding == "utf-8":
            # The shared helper returns UTF8 arrays with object dtype. Convert these
            # back to Unicode before another validation pass; ordinary object arrays
            # still reach the shared helper unchanged and are rejected there.
            items = []
            for item in value.flat:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                if not isinstance(item, str):
                    raise ValueError("UTF8 attribute arrays must contain only strings.")
                items.append(item)
            return np.asarray(items, dtype=str).reshape(value.shape)
    return value


def prepare_episode_metadata(metadata):
    """Validate/serialize root attributes before any hardware is connected."""
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("Episode metadata must be a mapping of attribute names to values.")
    if not metadata:
        return {}
    normalized = {}
    for key, value in metadata.items():
        if key in _OWNED_ATTRIBUTES:
            raise ValueError(f"Episode attribute {key!r} is reserved or no longer supported.")
        normalized[key] = _normalize_metadata_value(value)
    from wrobo.utils.hdf5_metadata import _prepare_attributes

    attributes = _prepare_attributes(normalized)
    dataset_name = attributes.get("dataset")
    if dataset_name is not None and (not isinstance(dataset_name, str) or not dataset_name):
        raise ValueError("The dataset attribute must be a nonempty string.")
    return attributes


def _feature_keys(keys, name):
    if isinstance(keys, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty sequence of feature names.")
    keys = tuple(keys)
    if not keys or any(not isinstance(key, str) or not key or "\0" in key for key in keys):
        raise ValueError(f"{name} must contain nonempty feature names.")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} must not contain duplicate feature names.")
    return keys


def prepare_data_description(*, state_keys, action_keys, robot_config=None, teleop_config=None):
    """Build the shared column ordering and hardware configuration description."""
    for name, config in (("robot_config", robot_config), ("teleop_config", teleop_config)):
        if config is not None and not isinstance(config, Mapping):
            raise ValueError(f"{name} must be a mapping or None.")
    description = {
        "state_keys": list(_feature_keys(state_keys, "state_keys")),
        "action_keys": list(_feature_keys(action_keys, "action_keys")),
        "robot_config": _normalize_metadata_value(robot_config),
        "teleop_config": _normalize_metadata_value(teleop_config),
    }
    try:
        # Round-tripping also takes a private snapshot and normalizes tuples.
        return json.loads(json.dumps(description, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Hardware descriptions must contain finite JSON-serializable values.") from exc


def ensure_data_description(dataset_dir, description, *, create=False):
    """Check or atomically create shared metadata without replacing an existing file.

    ``create=False`` performs a read-only preflight before connecting hardware.
    The first recorder creates the file; subsequent recorders reuse it unchanged.
    """
    directory = Path(dataset_dir).expanduser().resolve()
    path = directory / "data_description.json"

    def check_existing():
        try:
            with path.open(encoding="utf-8") as file:
                existing = json.load(file)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"Invalid JSON in {path}; the existing description was not changed.") from exc
        if existing != description:
            raise ValueError(
                f"{path} does not match this recording's column order or hardware configuration. "
                "Use a different dataset directory."
            )
        return path

    try:
        return check_existing()
    except FileNotFoundError:
        if not create:
            return None
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".data_description_", suffix=".tmp", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(description, file, ensure_ascii=False, allow_nan=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return check_existing()
        return path
    finally:
        temporary.unlink(missing_ok=True)


class EpisodeRecorder:
    """Buffer scalar samples in RAM and write image batches on a background thread.

    ``state_keys`` and ``action_keys`` define vector order. ``image_keys`` maps native
    camera observation names to their expected ``(height, width, 3)`` shapes. Native
    frames must be RGB uint8 HWC; files use the existing RGB CHW image codec.

    The caller supplies observations captured before sending this frame's action,
    and the actual action returned by the robot after its native safety clipping.
    ``timestamp`` is a nonnegative elapsed time in seconds, with the origin chosen
    by the caller. Only ``finish`` publishes a file under ``episodes/``.

    The bounded image queue applies backpressure rather than dropping frames. The
    producer copies camera frames because drivers may reuse their image buffers.
    Only the writer accesses HDF5 during collection; the caller writes scalar
    buffers and final attributes after joining it. There is no concurrent HDF5
    access. This object has one producer, the teleoperation control loop.
    Shared column names and hardware snapshots live in ``data_description.json``.
    """

    def __init__(
        self,
        dataset_dir,
        *,
        state_keys,
        action_keys,
        image_keys=None,
        encoding="jpeg",
        jpeg_quality=95,
        metadata=None,
        robot_config=None,
        teleop_config=None,
        fps=30,
        writer_queue_size=30,
        writer_batch_size=8,
    ):
        self.state_keys = _feature_keys(state_keys, "state_keys")
        self.action_keys = _feature_keys(action_keys, "action_keys")
        if encoding not in ("png", "jpeg", "gzip", "raw"):
            raise ValueError("Image encoding must be png, jpeg, gzip, or raw.")
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int) or not 0 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer between 0 and 100.")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive finite number.")
        for name, value in (("writer_queue_size", writer_queue_size), ("writer_batch_size", writer_batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if image_keys is None:
            image_keys = {}
        if not isinstance(image_keys, Mapping):
            raise ValueError("image_keys must map camera names to RGB HWC shapes.")
        self.image_keys = {}
        for name, shape in image_keys.items():
            if not isinstance(name, str) or not name or "/" in name or "\0" in name:
                raise ValueError("Camera names must be nonempty strings without '/'.")
            shape = tuple(shape)
            if len(shape) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape) or shape[2] != 3:
                raise ValueError(f"Camera {name!r} must have an RGB (height, width, 3) shape.")
            self.image_keys[name] = shape
        attributes = prepare_episode_metadata(metadata)
        dataset_name = attributes.get("dataset")
        if dataset_name is not None and (not isinstance(dataset_name, str) or not dataset_name):
            raise ValueError("The dataset attribute must be a nonempty string.")
        description = prepare_data_description(
            state_keys=self.state_keys, action_keys=self.action_keys,
            robot_config=robot_config, teleop_config=teleop_config,
        )

        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.description_path = ensure_data_description(self.dataset_dir, description, create=True)
        self.encoding = encoding
        self.jpeg_quality = jpeg_quality
        self._count = 0
        self._last_timestamp = None
        self._closed = False
        self._final_path = None
        self._file = None
        self._datasets = {}
        self.writer_queue_size = writer_queue_size
        self.writer_batch_size = writer_batch_size
        self._rows = []
        self._queue = Queue(maxsize=writer_queue_size)
        self._stop_requested = Event()
        self._discard_requested = Event()
        self._writer_done = Event()
        self._writer_error = None
        self._writer_error_reported = False
        self._writer = None
        self._written_count = 0
        incomplete_dir = self.dataset_dir / "_incomplete"
        incomplete_dir.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix="episode_", suffix=".hdf5", dir=incomplete_dir)
        os.close(descriptor)
        self.temp_path = Path(name)
        try:
            self._file = h5py.File(self.temp_path, "w")
            for key, value in attributes.items():
                self._file.attrs[key] = value
            self._file.attrs["dataset"] = dataset_name or self.dataset_dir.name
            self._file.attrs["success"] = False
            self._file.attrs["comment"] = ""
            self._file.attrs["seq_len"] = 0
            self._file.attrs["FPS"] = float(fps)
            observations = self._file.create_group("observations")
            self._datasets["state"] = observations.create_dataset(
                "proprio_state", shape=(0, len(self.state_keys)),
                maxshape=(None, len(self.state_keys)), dtype="float32",
                compression="gzip", compression_opts=1, shuffle=True,
                chunks=(50, len(self.state_keys)),
            )
            self._datasets["action"] = self._file.create_dataset(
                "action_abs", shape=(0, len(self.action_keys)),
                maxshape=(None, len(self.action_keys)), dtype="float32",
                compression="gzip", compression_opts=1, shuffle=True,
                chunks=(50, len(self.action_keys)),
            )
            self._datasets["timestamp"] = self._file.create_dataset(
                "timestamp", shape=(0,), maxshape=(None,), dtype="float64", chunks=(50,),
            )
            self._datasets["timestamp"].attrs["unit"] = "seconds"
            for name, (height, width, channels) in self.image_keys.items():
                if encoding in ("png", "jpeg"):
                    image = observations.create_dataset(
                        f"image_{name}", shape=(0,), maxshape=(None,),
                        dtype=h5py.vlen_dtype(np.uint8), chunks=(1,),
                    )
                else:
                    options = {"compression": "gzip", "compression_opts": 4, "shuffle": True} if encoding == "gzip" else {}
                    image = observations.create_dataset(
                        f"image_{name}", shape=(0, channels, height, width),
                        maxshape=(None, channels, height, width), dtype="uint8",
                        chunks=(1, channels, height, width), **options,
                    )
                image.attrs[ENCODING_ATTR] = encoding
                image.attrs["frame_shape_chw"] = np.array([channels, height, width], dtype=np.int64)
                if encoding == "jpeg":
                    image.attrs["jpeg_quality"] = jpeg_quality
                self._datasets[f"image:{name}"] = image
            if self.image_keys:
                self._writer = Thread(
                    target=self._write_images, name=f"wrobo-hdf5-{self.temp_path.stem}", daemon=True,
                )
                self._writer.start()
        except BaseException:
            self.abort()
            raise

    @property
    def frame_count(self):
        return self._count

    @property
    def final_path(self):
        return self._final_path

    def _require_open(self):
        if self._closed:
            raise RuntimeError("This episode recorder is closed.")

    def _check_writer_error(self):
        if self._writer_error is not None:
            self._writer_error_reported = True
            raise RuntimeError(f"Background trajectory writer failed: {self._writer_error}") from self._writer_error

    def _write_images(self):
        batch = []
        try:
            while not self._discard_requested.is_set():
                try:
                    frames = self._queue.get(timeout=0.05)
                except Empty:
                    if batch:
                        self._write_image_batch(batch)
                        batch.clear()
                    if self._stop_requested.is_set():
                        break
                    continue
                batch.append(frames)
                if len(batch) >= self.writer_batch_size:
                    self._write_image_batch(batch)
                    batch.clear()
        except BaseException as exc:
            self._writer_error = exc
        finally:
            self._writer_done.set()

    def _write_image_batch(self, batch):
        """Encode first, then extend every camera by one synchronized batch."""
        encoded = {}
        for name in self.image_keys:
            frames = [np.transpose(sample[name], (2, 0, 1)) for sample in batch]
            if self.encoding in ("png", "jpeg"):
                values = np.empty(len(batch), dtype=object)
                for index, frame in enumerate(frames):
                    if self._discard_requested.is_set():
                        return
                    values[index] = encode_frame(frame, self.encoding, jpeg_quality=self.jpeg_quality)
            else:
                values = np.stack(frames)
            encoded[name] = values
        if self._discard_requested.is_set():
            return
        start = self._written_count
        end = start + len(batch)
        try:
            for name, values in encoded.items():
                dataset = self._datasets[f"image:{name}"]
                dataset.resize(end, axis=0)
                if self.encoding in ("png", "jpeg"):
                    # h5py interprets equal-length vlen payloads as a 2-D array
                    # on slice assignment. Extend once, then write each payload.
                    for offset, value in enumerate(values):
                        dataset[start + offset] = value
                else:
                    dataset[start:end] = values
        except BaseException:
            # Roll every camera back to the previous complete batch on failure.
            for name in self.image_keys:
                try:
                    self._datasets[f"image:{name}"].resize(start, axis=0)
                except Exception:
                    pass
            raise
        self._written_count = end

    def _stop_writer(self, *, discard=False):
        if discard:
            self._discard_requested.set()
        self._stop_requested.set()
        if self._writer is not None and self._writer.ident is not None:
            # Wait for our event before join: a Ctrl+C during Thread.join on some
            # Python versions can otherwise report a live writer as stopped.
            while not self._writer_done.wait(timeout=0.05):
                pass
            self._writer.join()
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def _persist_scalars(self, count):
        """Finalize a complete prefix after the image writer has stopped."""
        for name in self.image_keys:
            self._datasets[f"image:{name}"].resize(count, axis=0)
        for key, column in (("state", 0), ("action", 1), ("timestamp", 2)):
            dataset = self._datasets[key]
            dataset.resize(count, axis=0)
            if count:
                dataset[:] = np.asarray([row[column] for row in self._rows[:count]], dtype=dataset.dtype)
        self._file.attrs.modify("seq_len", count)
        self._count = count

    def _close_file(self):
        try:
            if self._file is not None:
                self._file.close()
        finally:
            self._file = None
            self._closed = True
            self._rows.clear()

    @staticmethod
    def _vector(values, keys, name):
        if not isinstance(values, Mapping):
            raise ValueError(f"{name} must be a mapping of feature names to scalar values.")
        missing = [key for key in keys if key not in values]
        if missing:
            raise ValueError(f"Missing {name} features: {missing}.")
        vector = np.asarray([values[key] for key in keys])
        if vector.ndim != 1 or vector.dtype.kind not in "iuf":
            raise ValueError(f"{name} features must be finite numeric scalars.")
        with np.errstate(over="ignore", invalid="ignore"):
            vector = vector.astype(np.float32)
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} features must be finite float32 values.")
        return vector

    def append(self, observation, action, *, timestamp):
        """Copy a synchronized sample into scalar RAM and the bounded image queue."""
        self._require_open()
        self._check_writer_error()
        if self._stop_requested.is_set():
            raise RuntimeError("This episode recorder has stopped accepting frames.")
        row = {
            "state": self._vector(observation, self.state_keys, "observation"),
            "action": self._vector(action, self.action_keys, "action"),
        }
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be a nonnegative finite number of seconds.")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("Episode timestamps must be nondecreasing.")
        row["timestamp"] = timestamp
        frames = {}
        for name, shape in self.image_keys.items():
            if name not in observation:
                raise ValueError(f"Missing camera observation {name!r}.")
            frame = np.asarray(observation[name])
            if frame.shape != shape or frame.dtype != np.uint8:
                raise ValueError(f"Camera {name!r} requires an RGB uint8 frame with shape {shape}.")
            frames[name] = np.array(frame, copy=True, order="C")
        if frames:
            while True:
                self._check_writer_error()
                try:
                    self._queue.put(frames, timeout=0.05)
                    break
                except Full:
                    continue
        self._rows.append((row["state"], row["action"], row["timestamp"]))
        self._count += 1
        self._last_timestamp = timestamp

    def finish(self, *, success=True, comment=""):
        """Close and publish ``episode_N.hdf5``; discard an empty episode.

        Files are never replaced. If another writer claims the chosen number during
        publication, ``FileExistsError`` is raised and the temporary file remains.
        A second call after successful publication returns the same path.
        """
        if self._final_path is not None:
            return self._final_path
        self._require_open()
        if not isinstance(success, bool) or not isinstance(comment, str):
            raise ValueError("success must be a boolean and comment must be a string.")
        if not self._count:
            self.abort()
            return None
        self._stop_writer()
        self._check_writer_error()
        if self.image_keys and self._written_count != self._count:
            raise RuntimeError("Image and scalar frame counts differ; the trajectory cannot be published.")
        self._persist_scalars(self._count)
        self._file.attrs.modify("success", success)
        self._file.attrs["comment"] = comment
        self._file.flush()
        self._close_file()
        # Windows os.fsync uses _commit/FlushFileBuffers, which needs a
        # writable handle. Reopen without truncating the completed HDF5 file.
        with self.temp_path.open("r+b") as file:
            os.fsync(file.fileno())
        episodes_dir = self.dataset_dir / "episodes"
        episodes_dir.mkdir(exist_ok=True)
        next_index = 0
        for path in episodes_dir.iterdir():
            match = re.fullmatch(r"episode_([0-9]+)\.hdf5", path.name)
            if match:
                next_index = max(next_index, int(match.group(1)) + 1)
        destination = episodes_dir / f"episode_{next_index}.hdf5"
        os.link(self.temp_path, destination)
        self._final_path = destination
        self.temp_path.unlink()
        return destination

    def abort(self):
        """Close and discard this recorder's unpublished temporary file only."""
        self._stop_writer(discard=True)
        self._close_file()
        if self._final_path is None and hasattr(self, "temp_path"):
            self.temp_path.unlink(missing_ok=True)

    def close(self):
        """Retain an interrupted nonempty episode in ``_incomplete/`` only.

        Recording interruption never commits a training episode. An explicit
        ``abort`` can subsequently discard this recorder's retained temporary file.
        """
        if self._final_path is not None:
            return None
        if self._closed and self._file is None:
            if self.temp_path.exists():
                # A previous close, or a publication failure after closing HDF5.
                with h5py.File(self.temp_path, "r+") as file:
                    file.attrs.modify("success", False)
                return self.temp_path
            return None
        self._stop_writer()
        count = min(self._count, self._written_count) if self.image_keys else self._count
        if self._writer_error is not None and not self._writer_error_reported:
            logger.error("Background trajectory writer failed: %s. Only %d/%d frames are complete.",
                         self._writer_error, count, self._count)
            self._writer_error_reported = True
        if not count:
            self._count = 0
            self.abort()
            return None
        if self._file is not None:
            try:
                self._file.attrs.modify("success", False)
                self._persist_scalars(count)
                self._file.flush()
            finally:
                self._close_file()
        return self.temp_path if self.temp_path.exists() else None

    retain_incomplete = close
