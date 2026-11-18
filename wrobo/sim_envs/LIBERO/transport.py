"""Small synchronous inference transport shared by Python 3.8 and model Python.

Only NumPy and the standard library are needed on the simulation side. Images
remain lossless uint8 arrays; numeric NPZ payloads never deserialize pickle.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
import math
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np


PROTOCOL = "wrobo_libero_http_v2"
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
SESSION_HEADER = "X-Wrobo-Session"


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _json_loads(value):
    def reject_constant(constant):
        raise ValueError("Nonfinite JSON constant: " + constant)

    return json.loads(value.decode("utf-8"), parse_constant=reject_constant)


def _metadata(value):
    if not isinstance(value, dict):
        raise ValueError("Policy metadata must be a JSON object")
    for key in ("history_width", "state_dim", "action_dim", "action_chunk_size"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError("Policy metadata requires a positive " + key)
    cameras = value.get("camera_keys")
    if (not isinstance(cameras, list) or not cameras
            or any(not isinstance(name, str) or not name for name in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError("Policy camera_keys must be unique nonempty names")
    for key in ("proprio_key", "action_key"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError("Policy metadata requires " + key)
    if not isinstance(value.get("requires_instruction"), bool):
        raise ValueError("Policy metadata requires boolean requires_instruction")
    return value


def _window(value, metadata):
    if not isinstance(value, dict):
        raise ValueError("Observation must be a dictionary")
    state = np.asarray(value.get("state"))
    expected = (metadata["history_width"], metadata["state_dim"])
    if state.shape != expected or state.dtype.kind not in "fiu":
        raise ValueError("Observation state must have shape {!r}".format(expected))
    with np.errstate(over="ignore", invalid="ignore"):
        state = state.astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Observation state must contain finite float32 values")
    images = value.get("images")
    if not isinstance(images, dict) or set(images) != set(metadata["camera_keys"]):
        raise ValueError("Observation cameras must match policy camera_keys")
    for name in metadata["camera_keys"]:
        frame = np.asarray(images[name])
        if (frame.dtype != np.uint8 or frame.ndim != 4
                or frame.shape[0] != expected[0] or frame.shape[-1] != 3
                or min(frame.shape[1:3]) < 1):
            raise ValueError("Camera {} must contain RGB uint8 [history,H,W,3]".format(name))
    instruction = value.get("instruction", metadata.get("instruction"))
    if instruction is None:
        instruction = metadata.get("instruction")
    if instruction is not None and (not isinstance(instruction, str) or not instruction.strip()):
        raise ValueError("Instruction must be a nonempty string")
    if metadata["requires_instruction"] and instruction is None:
        raise ValueError("This policy requires an instruction")
    return {"state": state, "images": images, "instruction": instruction}


def _pack_window(value, metadata):
    value = _window(value, metadata)
    arrays = {
        "state": value["state"],
        "description": np.frombuffer(_json_bytes({
            "camera_keys": metadata["camera_keys"], "instruction": value["instruction"],
        }), dtype=np.uint8),
    }
    arrays.update({"image_{}".format(index): value["images"][name]
                   for index, name in enumerate(metadata["camera_keys"])})
    stream = BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _unpack_window(payload, metadata):
    with np.load(BytesIO(payload), allow_pickle=False) as arrays:
        expected = {"state", "description"} | {
            "image_{}".format(index) for index in range(len(metadata["camera_keys"]))
        }
        if set(arrays.files) != expected:
            raise ValueError("Unexpected observation arrays")
        description_array = arrays["description"]
        if description_array.dtype != np.uint8 or description_array.ndim != 1:
            raise ValueError("Observation description must contain UTF-8 bytes")
        description = _json_loads(description_array.tobytes())
        if not isinstance(description, dict) or description.get("camera_keys") != metadata["camera_keys"]:
            raise ValueError("Observation camera ordering does not match policy")
        return _window({
            "state": arrays["state"],
            "images": {name: arrays["image_{}".format(index)]
                       for index, name in enumerate(metadata["camera_keys"])},
            "instruction": description.get("instruction"),
        }, metadata)


def _actions(value, metadata):
    actions = np.asarray(value)
    expected = (metadata["action_chunk_size"], metadata["action_dim"])
    if actions.shape != expected or actions.dtype.kind not in "fiu":
        raise ValueError("Policy actions must have shape {!r}".format(expected))
    with np.errstate(over="ignore", invalid="ignore"):
        actions = actions.astype(np.float32)
    if not np.isfinite(actions).all():
        raise ValueError("Policy actions must contain finite float32 values")
    return np.ascontiguousarray(actions)


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return value


def _session_id(value):
    if (not isinstance(value, str) or len(value) != 32
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("session_id must contain 32 lowercase hexadecimal characters")
    return value


class PolicyClient:
    """Synchronous client; receives complete, inverse-normalized action chunks."""

    def __init__(self, server_url="http://127.0.0.1:8000", timeout=120.0):
        parsed = urlsplit(server_url)
        if (parsed.scheme != "http" or not parsed.hostname or parsed.path not in ("", "/")
                or parsed.query or parsed.fragment or parsed.username or parsed.password):
            raise ValueError("server_url must be an HTTP origin, e.g. http://127.0.0.1:8000")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.server_url = server_url.rstrip("/")
        self.timeout = float(timeout)
        self._closed = False
        self._session_id = uuid.uuid4().hex
        self._session_active = False
        reply = _json_loads(self._request("/metadata"))
        if not isinstance(reply, dict) or reply.get("protocol") != PROTOCOL:
            raise ValueError("Server returned an unsupported inference protocol")
        self.metadata = _metadata(reply.get("metadata"))

    def _request(self, route, payload=None, content_type="application/json", timeout=None):
        if self._closed:
            raise RuntimeError("Policy client is closed")
        if payload is not None and len(payload) > MAX_PAYLOAD_BYTES:
            raise ValueError("Inference request exceeds the 64 MiB limit")
        request = Request(self.server_url + route, data=payload,
                          headers={"Content-Type": content_type, SESSION_HEADER: self._session_id})
        try:
            with urlopen(request, timeout=self.timeout if timeout is None else timeout) as response:
                result = response.read(MAX_PAYLOAD_BYTES + 1)
        except HTTPError as error:
            try:
                detail = error.read(8192).decode("utf-8", errors="replace")
            finally:
                error.close()
            raise RuntimeError("Policy server {} failed (HTTP {}): {}".format(route, error.code, detail)) from error
        except (URLError, OSError) as error:
            raise RuntimeError("Policy server {} failed: {}".format(route, error)) from error
        if len(result) > MAX_PAYLOAD_BYTES:
            raise RuntimeError("Inference response exceeds the 64 MiB limit")
        return result

    def predict(self, window):
        if self._closed:
            raise RuntimeError("Policy client is closed")
        observation = _pack_window(window, self.metadata)
        if not self._session_active:
            raise RuntimeError("Policy client must reset an episode seed before prediction")
        payload = self._request("/predict", observation, "application/x-npz")
        with np.load(BytesIO(payload), allow_pickle=False) as arrays:
            if arrays.files != ["actions"]:
                raise ValueError("Unexpected action arrays from policy server")
            return _actions(arrays["actions"], self.metadata)

    def reset(self, seed):
        options = {"session_id": self._session_id, "seed": _seed(seed)}
        reply = _json_loads(self._request("/reset", _json_bytes(options)))
        if reply != options:
            raise ValueError("Policy server did not acknowledge episode seed")
        self._session_active = True

    def close(self):
        if self._closed:
            return
        try:
            if self._session_active:
                self._request("/close", _json_bytes({"session_id": self._session_id}),
                              timeout=min(self.timeout, 5.0))
        except (RuntimeError, ValueError):
            # Closing a dead service must not hide the original rollout error.
            pass
        finally:
            self._closed = True
            self._session_active = False


def make_server(predictor, *, host="127.0.0.1", port=8000, reset_callback=None):
    """Build a single-request-at-a-time model server without owning the model."""
    metadata = _metadata(predictor.metadata)
    _json_bytes(metadata)
    from .rng import SessionRandomStates

    random_states = SessionRandomStates(reset_callback=reset_callback)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(120.0)

        def log_message(self, *args):
            pass

        def _respond(self, status, payload, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != "/metadata":
                self._respond(404, _json_bytes({"error": "Unknown endpoint"}))
                return
            self._respond(200, _json_bytes({"protocol": PROTOCOL, "metadata": metadata}))

        def do_POST(self):
            try:
                if self.path not in ("/predict", "/reset", "/close"):
                    self._respond(404, _json_bytes({"error": "Unknown endpoint"}))
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_PAYLOAD_BYTES:
                    raise ValueError("Request requires Content-Length in [1, 64 MiB]")
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise ValueError("Incomplete inference request")
                if self.path == "/reset":
                    options = _json_loads(payload)
                    if not isinstance(options, dict) or set(options) != {"seed", "session_id"}:
                        raise ValueError("Reset requires a session_id and seed")
                    session_id = _session_id(options["session_id"])
                    seed = _seed(options["seed"])
                    random_states.reset(session_id, seed)
                    self._respond(200, _json_bytes({"session_id": session_id, "seed": seed}))
                elif self.path == "/close":
                    options = _json_loads(payload)
                    if not isinstance(options, dict) or set(options) != {"session_id"}:
                        raise ValueError("Close requires a session_id")
                    session_id = _session_id(options["session_id"])
                    random_states.release(session_id)
                    self._respond(200, _json_bytes({"session_id": session_id, "closed": True}))
                else:
                    window = _unpack_window(payload, metadata)
                    session_id = _session_id(self.headers.get(SESSION_HEADER))
                    with random_states.activate(session_id):
                        actions = _actions(predictor.predict(window), metadata)
                    stream = BytesIO()
                    np.savez(stream, actions=actions)
                    self._respond(200, stream.getvalue(), "application/x-npz")
            except (ValueError, TypeError, KeyError) as error:
                self._respond(400, _json_bytes({"error": str(error)}))
            except Exception as error:
                self._respond(500, _json_bytes({"error": type(error).__name__ + ": " + str(error)}))

    class PolicyHTTPServer(HTTPServer):
        request_queue_size = 64

        def server_close(self):
            try:
                super().server_close()
            finally:
                random_states.clear()

    server = PolicyHTTPServer((host, port), Handler)
    server.random_states = random_states
    return server
