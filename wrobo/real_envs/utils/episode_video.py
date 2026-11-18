"""Record camera videos independently of trajectory recording and review."""

from collections.abc import Mapping
import math
from numbers import Integral, Real
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import shutil
import tempfile
from threading import Event, Thread

import cv2
import numpy as np


_STOP = object()


class CameraVideoRecorder:
    """Copy RGB camera frames into a bounded queue for background MP4 encoding.

    ``image_keys`` maps native camera names to their RGB ``(height, width, 3)``
    shapes.  Width and height must be positive and even, preventing OpenCV from
    silently cropping frames.  A single producer calls :meth:`append` once per
    control tick.  The queue applies backpressure instead of dropping frames.

    :meth:`finish` and :meth:`close` drain the queue, release all writers, and
    publish ``video_dir/episode_N/<camera>.mp4``.  Video numbering depends only
    on existing video directories.  :meth:`abort` discards this round's videos.
    This recorder has no trajectory or HDF5 dependency.
    """

    def __init__(self, video_dir, *, image_keys, fps=30, writer_queue_size=30):
        if not isinstance(image_keys, Mapping):
            raise ValueError("image_keys must map camera names to RGB HWC shapes.")
        shapes = {}
        for name, shape in image_keys.items():
            if not isinstance(name, str) or not name or name in (".", "..") or any(
                character in name for character in ("/", "\\", "\0")
            ):
                raise ValueError("Camera names must be nonempty filenames without path separators.")
            try:
                shape = tuple(shape)
            except TypeError as exc:
                raise ValueError(f"Camera {name!r} must have an RGB HWC shape.") from exc
            if len(shape) != 3 or any(
                isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size <= 0
                for size in shape
            ) or shape[2] != 3 or shape[0] % 2 or shape[1] % 2:
                raise ValueError(f"Camera {name!r} must have a positive even height and width and three RGB channels.")
            shapes[name] = tuple(int(size) for size in shape)
        if isinstance(fps, (bool, np.bool_)) or not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive finite number.")
        if isinstance(writer_queue_size, (bool, np.bool_)) or not isinstance(writer_queue_size, Integral) or writer_queue_size <= 0:
            raise ValueError("writer_queue_size must be a positive integer.")
        self.video_dir = Path(video_dir).expanduser().resolve()
        self.image_keys = shapes
        self.fps = float(fps)
        self._queue = Queue(maxsize=int(writer_queue_size))
        self._abort = Event()
        self._done = Event()
        self._error = None
        self._closed = False
        self._count = 0
        self._paths = {}
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.temp_path = Path(tempfile.mkdtemp(prefix=".episode_", dir=self.video_dir))
        self._worker = Thread(target=self._write, name="wrobo-camera-video", daemon=True)
        try:
            self._worker.start()
        except BaseException:
            self._cleanup_temporary()
            raise

    def _cleanup_temporary(self):
        shutil.rmtree(self.temp_path, ignore_errors=True)
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def _raise_writer_error(self):
        if self._error is not None:
            self._abort.set()
            self._done.wait()
            self._worker.join()
            self._cleanup_temporary()
            self._closed = True
            raise RuntimeError(f"Camera video recording in {self.video_dir} failed: {self._error}") from self._error

    def append(self, observation):
        """Snapshot all cameras; wait for queue capacity instead of losing frames."""
        self._raise_writer_error()
        if self._closed:
            raise RuntimeError("The camera video recorder is closed.")
        if not isinstance(observation, Mapping):
            raise ValueError("Camera observations must be a mapping.")
        frames = {}
        for name, shape in self.image_keys.items():
            if name not in observation:
                raise ValueError(f"Camera observation is missing {name!r}.")
            frame = np.asarray(observation[name])
            if frame.dtype != np.uint8 or frame.shape != shape:
                raise ValueError(f"Camera {name!r} must provide uint8 RGB HWC frames with shape {shape}.")
            # Drivers commonly reuse the same buffer on the next control tick.
            frames[name] = frame.copy()
        if not frames:
            return
        while True:
            self._raise_writer_error()
            try:
                self._queue.put(frames, timeout=0.1)
                self._count += 1
                break
            except Full:
                continue
        self._raise_writer_error()

    def _write(self):
        writers = {}
        try:
            while not self._abort.is_set():
                try:
                    frames = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                if frames is _STOP:
                    break
                if not writers:
                    for name, (height, width, _) in self.image_keys.items():
                        writer = cv2.VideoWriter(
                            str(self.temp_path / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            self.fps, (width, height),
                        )
                        writers[name] = writer
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot open MP4 writer for camera {name!r}.")
                for name, frame in frames.items():
                    writers[name].write(np.ascontiguousarray(frame[:, :, ::-1]))
        except BaseException as exc:
            self._error = exc
        finally:
            for writer in writers.values():
                try:
                    writer.release()
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc
            self._done.set()

    def _drain_writer(self):
        while not self._done.is_set():
            try:
                self._queue.put(_STOP, timeout=0.1)
                break
            except Full:
                continue
        # Waiting on our own event first avoids Thread.join's interrupted-wait
        # behavior marking the thread stopped before its finally block exits.
        self._done.wait()
        self._worker.join()

    def finish(self) -> dict[str, Path]:
        """Drain and publish this round, using independent video numbering."""
        if self._closed:
            self._raise_writer_error()
            return dict(self._paths)
        destination = None
        try:
            self._drain_writer()
            self._raise_writer_error()
            if not self._count:
                return {}
            for name in self.image_keys:
                path = self.temp_path / f"{name}.mp4"
                if not path.is_file() or not path.stat().st_size:
                    raise RuntimeError(f"MP4 writer did not create a nonempty video for camera {name!r}.")
            next_index = 0
            for path in self.video_dir.iterdir():
                match = re.fullmatch(r"episode_([0-9]+)", path.name)
                if match and path.is_dir():
                    next_index = max(next_index, int(match.group(1)) + 1)
            # Claim a directory exclusively; another recorder may publish while
            # this round is encoding.  Never replace existing video directories.
            while True:
                candidate = self.video_dir / f"episode_{next_index}"
                try:
                    candidate.mkdir()
                except FileExistsError:
                    next_index += 1
                else:
                    destination = candidate
                    break
            for name in self.image_keys:
                os.replace(self.temp_path / f"{name}.mp4", destination / f"{name}.mp4")
            self._paths = {name: destination / f"{name}.mp4" for name in self.image_keys}
            return dict(self._paths)
        except BaseException as exc:
            # Publication and interruption errors must not remove files while
            # the encoder still owns them.  Stop and join it before cleanup.
            self._abort.set()
            self._done.wait()
            self._worker.join()
            if destination is not None:
                shutil.rmtree(destination, ignore_errors=True)
            if not isinstance(exc, Exception):
                raise
            raise RuntimeError(f"Cannot save camera videos in {self.video_dir}: {exc}") from exc
        finally:
            self._closed = True
            self._cleanup_temporary()

    def abort(self):
        """Discard only this round's unpublished videos and stop its worker."""
        if self._paths:
            return
        self._abort.set()
        self._done.wait()
        self._worker.join()
        self._closed = True
        self._error = None
        self._cleanup_temporary()

    def close(self) -> dict[str, Path]:
        """Finish videos even when trajectory recording was interrupted/discarded."""
        return self.finish()
