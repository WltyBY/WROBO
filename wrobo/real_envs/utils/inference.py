"""Schedule model action chunks without touching robot hardware.

The async mode predicts in one worker while the caller executes actions.  It
aligns overlapping chunks to control ticks; it is ordinary asynchronous action
chunking, not model-specific Real-Time Chunking (RTC).
"""

from collections.abc import Mapping
from copy import deepcopy
import logging
import math
from numbers import Integral, Real
from threading import Condition, Thread

import numpy as np


logger = logging.getLogger(__name__)


def _metadata_value(metadata, name):
    return metadata[name] if isinstance(metadata, Mapping) else getattr(metadata, name)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number between 0 and 1.")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number between 0 and 1.")
    return value


class ChunkInference:
    """A synchronous FIFO or asynchronous, overlapping action-chunk scheduler.

    ``predictor.metadata`` supplies ``action_chunk_size`` and ``action_dim``
    (or named ``action_keys``);
    ``predictor.predict(window)`` returns finite, denormalized ``[T, D]`` targets.
    The caller invokes :meth:`action` once per control tick.  ``None`` means no
    target is ready; the caller should leave the robot's current target alone.
    Async overlap handling uses ``new_action_fuse="aggregate"`` to blend old
    and new targets with ``new_weight``, or ``"latest"`` to replace each
    overlapping future target with its new prediction.  Both discard expired
    rows and preserve queued targets outside the new chunk's valid range.

    Each :meth:`start` begins a fresh round.  A cancelled inference may finish,
    but its output is discarded.  One persistent worker serializes calls to the
    predictor across rounds; there is at most one pending request.  The caller
    owns the predictor and must cancel any remote request before :meth:`close`
    and close the predictor itself after the worker stops.
    """

    def __init__(
        self,
        predictor,
        *,
        mode="sync",
        chunk_steps=None,
        prefetch_threshold=0.5,
        new_weight=0.7,
        new_action_fuse="aggregate",
        close_timeout=5.0,
    ):
        if mode not in ("sync", "async"):
            raise ValueError("Inference mode must be 'sync' or 'async'.")
        try:
            metadata = predictor.metadata
            horizon = _positive_integer(
                _metadata_value(metadata, "action_chunk_size"), "action_chunk_size"
            )
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError("Predictor metadata must provide action_chunk_size.") from exc
        try:
            keys = _metadata_value(metadata, "action_keys")
        except (AttributeError, KeyError):
            keys = None
        if keys is not None and (
            not isinstance(keys, (list, tuple)) or not keys
            or any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys)
        ):
            raise ValueError("action_keys must be a nonempty sequence of unique names.")
        try:
            action_dim = _metadata_value(metadata, "action_dim")
        except (AttributeError, KeyError):
            action_dim = None
        if action_dim is None and keys is not None:
            action_dim = len(keys)
        self._action_dim = _positive_integer(action_dim, "action_dim")
        if keys is not None and len(keys) != self._action_dim:
            raise ValueError("action_dim does not match action_keys.")
        self.chunk_steps = horizon if chunk_steps is None else _positive_integer(chunk_steps, "chunk_steps")
        if self.chunk_steps > horizon:
            raise ValueError("chunk_steps cannot exceed the model's action_chunk_size.")
        self.prefetch_threshold = _fraction(prefetch_threshold, "prefetch_threshold")
        self.new_weight = _fraction(new_weight, "new_weight")
        if new_action_fuse not in ("latest", "aggregate"):
            raise ValueError("new_action_fuse must be 'latest' or 'aggregate'.")
        self.new_action_fuse = new_action_fuse
        if isinstance(close_timeout, bool) or not isinstance(close_timeout, Real):
            raise ValueError("close_timeout must be a positive finite number.")
        self.close_timeout = float(close_timeout)
        if not math.isfinite(self.close_timeout) or self.close_timeout <= 0:
            raise ValueError("close_timeout must be a positive finite number.")
        self.mode = mode
        self._predictor = predictor
        self._condition = Condition()
        self._worker = None
        self._closed = False
        self._active = False
        self._generation = 0
        self._cursor = 0
        self._executing = False
        self._queue = {}
        self._pending = None
        self._busy_generation = None
        self._error = None

    @property
    def queue_size(self):
        """Number of future targets ready for execution."""
        with self._condition:
            return len(self._queue)

    @property
    def running(self):
        with self._condition:
            return self._active and not self._closed

    def start(self):
        """Clear round state and enable prediction, retaining the same worker."""
        with self._condition:
            if self._closed:
                raise RuntimeError("The inference scheduler is closed.")
            self._generation += 1
            self._active = True
            self._cursor = 0
            self._executing = False
            self._queue.clear()
            self._pending = None
            self._error = None
            if self.mode == "async" and self._worker is None:
                self._worker = Thread(target=self._work, name="wrobo-policy-inference", daemon=True)
                self._worker.start()
            self._condition.notify_all()

    def stop(self):
        """Immediately discard round targets and invalidate outstanding results."""
        with self._condition:
            self._generation += 1
            self._active = False
            self._queue.clear()
            self._pending = None
            self._error = None
            self._executing = False
            self._condition.notify_all()

    def _predict(self, window):
        prediction = np.asarray(self._predictor.predict(window))
        if (
            prediction.ndim != 2
            or prediction.shape[0] < self.chunk_steps
            or prediction.shape[1] != self._action_dim
            or prediction.dtype.kind not in "iuf"
        ):
            raise ValueError(
                f"Policy must return a numeric [T, {self._action_dim}] action chunk "
                f"with at least {self.chunk_steps} steps; got {prediction.shape}."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = prediction[: self.chunk_steps].astype(np.float32, copy=True)
        if not np.isfinite(prediction).all():
            raise ValueError("Policy action chunk contains nonfinite targets.")
        return prediction

    def _merge(self, chunk, capture_cursor):
        for offset, new_action in enumerate(chunk):
            index = capture_cursor + offset
            if index < self._cursor:
                continue
            old_action = self._queue.get(index)
            self._queue[index] = (
                new_action.copy()
                if old_action is None or self.new_action_fuse == "latest"
                else old_action * (1 - self.new_weight) + new_action * self.new_weight
            )

    def _check_action(self):
        if self._closed:
            raise RuntimeError("The inference scheduler is closed.")
        if self._error is not None:
            raise RuntimeError("Asynchronous policy inference failed.") from self._error
        return self._active

    def action(self, window):
        """Return an owned ``[D]`` target or ``None`` when inactive/not ready."""
        if self.mode == "sync":
            return self._sync_action(window)
        with self._condition:
            if not self._check_action():
                return None
            capture_cursor = self._cursor
            action = self._queue.pop(capture_cursor, None)
            if action is not None:
                self._executing = True
            # Freeze time during initial warmup only.  Once motion has begun,
            # even an empty queue consumes a control tick, expiring late targets.
            if self._executing:
                self._cursor += 1
            if (
                len(self._queue) <= self.chunk_steps * self.prefetch_threshold
                and self._busy_generation != self._generation
                and self._pending is None
            ):
                self._pending = (self._generation, capture_cursor, deepcopy(window))
                self._condition.notify_all()
            return None if action is None else action.copy()

    def _sync_action(self, window):
        with self._condition:
            if not self._check_action():
                return None
            action = self._queue.pop(self._cursor, None)
            if action is not None:
                self._cursor += 1
                return action.copy()
            if self._busy_generation is not None:
                raise RuntimeError("Another synchronous policy prediction is still running.")
            generation, capture_cursor = self._generation, self._cursor
            self._busy_generation = generation
        try:
            chunk = self._predict(window)
        finally:
            with self._condition:
                self._busy_generation = None
                self._condition.notify_all()
        with self._condition:
            if self._closed or not self._active or generation != self._generation:
                return None
            self._merge(chunk, capture_cursor)
            action = self._queue.pop(self._cursor)
            self._cursor += 1
            return action.copy()

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                generation, capture_cursor, window = self._pending
                self._pending = None
                self._busy_generation = generation
            chunk, error = None, None
            try:
                chunk = self._predict(window)
            except BaseException as exc:
                error = exc
            with self._condition:
                self._busy_generation = None
                if not self._closed and self._active and generation == self._generation:
                    if error is not None:
                        self._error = error
                        self._pending = None
                    else:
                        self._merge(chunk, capture_cursor)
                self._condition.notify_all()

    def close(self):
        """Stop and join the worker; timeout if the predictor fails to return.

        A timed-out worker is a daemon and has no robot references.  Its results
        remain invalidated; a later close may join it after remote cancellation.
        """
        with self._condition:
            self._closed = True
            self._active = False
            self._generation += 1
            self._queue.clear()
            self._pending = None
            self._condition.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout=self.close_timeout)
            if worker.is_alive():
                raise TimeoutError(
                    "Policy inference did not stop before close_timeout; cancel any pending remote request."
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except TimeoutError:
            if exc_type is None:
                raise
            logger.exception("Inference worker did not stop while handling another error.")
        return False
