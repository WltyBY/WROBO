"""Per-client inference random streams, owned by the serial model service.

Torch is imported only when a model server constructs this manager. Simulation
clients therefore keep their NumPy-only transport dependency.
"""

from contextlib import contextmanager
import importlib.util
import random

import numpy as np


class SessionRandomStates:
    """Preserve the serial seed-once stream independently for each client."""

    def __init__(self, reset_callback=None):
        self.reset_callback = reset_callback
        self._states = {}
        self.torch = None
        if importlib.util.find_spec("torch") is not None:
            import torch

            self.torch = torch
            # Capture CUDA generators before any session is seeded, including
            # generators first used by a predictor on its first request.
            if torch.cuda.is_available():
                torch.cuda.get_rng_state_all()

    @property
    def session_count(self):
        return len(self._states)

    def _capture(self):
        state = {"python": random.getstate(), "numpy": np.random.get_state()}
        if self.torch is not None:
            state["torch"] = self.torch.get_rng_state()
            if self.torch.cuda.is_available():
                state["cuda"] = self.torch.cuda.get_rng_state_all()
        return state

    def _restore(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        if self.torch is not None:
            self.torch.set_rng_state(state["torch"])
            if "cuda" in state:
                self.torch.cuda.set_rng_state_all(state["cuda"])

    def _seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        if self.torch is not None:
            self.torch.manual_seed(seed)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(seed)
        if self.reset_callback is not None:
            self.reset_callback(seed)

    def reset(self, session_id, seed):
        outside = self._capture()
        try:
            self._seed(seed)
            self._states[session_id] = self._capture()
        finally:
            self._restore(outside)

    @contextmanager
    def activate(self, session_id):
        if session_id not in self._states:
            raise ValueError("Unknown inference session; reset the client before prediction")
        outside = self._capture()
        self._restore(self._states[session_id])
        try:
            yield
        finally:
            self._states[session_id] = self._capture()
            self._restore(outside)

    def release(self, session_id):
        self._states.pop(session_id, None)

    def clear(self):
        self._states.clear()
