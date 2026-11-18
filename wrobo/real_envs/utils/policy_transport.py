"""Paired wrobo inference RPCs using LeRobot's protobuf and chunk transport.

The wire payload is numeric NPZ plus UTF-8 JSON. Neither end deserializes
Python objects or loads models selected by a client.
"""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import math
from pathlib import Path
import threading

import numpy as np

from .device import TargetDevice, canonical_compute_device


_SERVICE = "wrobo.PolicyInference"
_PROTOCOL = 1
_MAX_PAYLOAD_BYTES = 512 * 1024 * 1024


def _transport():
    try:
        import grpc
        from lerobot.transport import services_pb2
        from lerobot.transport.utils import (
            grpc_channel_options,
            receive_bytes_in_chunks,
            send_bytes_in_chunks,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Remote inference requires the installed lerobot[grpcio-dep] package."
        ) from exc
    return grpc, services_pb2, grpc_channel_options, receive_bytes_in_chunks, send_bytes_in_chunks


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _json_bytes(value):
    return json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _json_loads(data):
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(bytes(data).decode("utf-8"), parse_constant=invalid_constant)


def _metadata(value):
    if not isinstance(value, Mapping):
        raise ValueError("Policy metadata must be a JSON object.")
    result = _json_loads(_json_bytes(value))
    for key in ("state_keys", "action_keys", "camera_keys"):
        names = result.get(key)
        if key != "camera_keys" and names is None:
            continue
        if (
            not isinstance(names, list)
            or not all(isinstance(name, str) and name for name in names)
            or len(names) != len(set(names))
            or (key != "camera_keys" and not names)
        ):
            raise ValueError(f"Policy metadata {key} must contain unique column names.")
    for key, names_key in (("state_dim", "state_keys"), ("action_dim", "action_keys")):
        names = result.get(names_key)
        dimension = result.get(key, len(names) if names else None)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Policy metadata {key} must be a positive integer.")
        if names and dimension != len(names):
            raise ValueError(f"Policy metadata {key} does not match {names_key}.")
        result[key] = dimension
    for key in ("history_width", "action_chunk_size"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Policy metadata {key} must be a positive integer.")
    if not isinstance(result.get("requires_instruction"), bool):
        raise ValueError("Policy metadata requires_instruction must be boolean.")
    if not isinstance(result.get("method"), str) or not result["method"]:
        raise ValueError("Policy metadata method must be a nonempty string.")
    if result.get("instruction") is not None and not isinstance(result["instruction"], str):
        raise ValueError("Policy metadata instruction must be a string or null.")
    return result


def _window(window, metadata, instruction=None):
    if not isinstance(window, Mapping):
        raise ValueError("An observation window must be a mapping.")
    state = np.asarray(window.get("state"))
    expected_shape = (metadata["history_width"], metadata["state_dim"])
    if state.dtype.kind not in "fiu" or state.shape != expected_shape:
        raise ValueError(f"Observation state must be a numeric array with shape {expected_shape}.")
    state = np.asarray(state, dtype=np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Observation state must contain finite float32 values.")
    images = window.get("images")
    if not isinstance(images, Mapping) or set(images) != set(metadata["camera_keys"]):
        raise ValueError("Observation images must match the model's camera names exactly.")
    frames = {}
    for key in metadata["camera_keys"]:
        image = np.asarray(images[key])
        if (
            image.dtype != np.uint8
            or image.ndim != 4
            or image.shape[0] != metadata["history_width"]
            or image.shape[-1] != 3
            or min(image.shape[1:3]) <= 0
        ):
            raise ValueError(f"Camera {key!r} must contain RGB uint8 images with shape (H, height, width, 3).")
        frames[key] = image
    language = window.get("instruction")
    if language is None:
        language = instruction
    if language is not None and not isinstance(language, str):
        raise ValueError("Instruction must be a string or null.")
    if metadata["requires_instruction"] and (language is None or not language.strip()):
        raise ValueError("This policy requires a nonempty instruction.")
    return {"state": state, "images": frames, "instruction": language}


def _pack_window(window, metadata):
    payload = {
        "state": window["state"],
        "metadata": np.frombuffer(
            _json_bytes({"camera_keys": metadata["camera_keys"], "instruction": window["instruction"]}),
            dtype=np.uint8,
        ),
    }
    payload.update({f"image_{index}": window["images"][name] for index, name in enumerate(metadata["camera_keys"])})
    stream = BytesIO()
    np.savez(stream, **payload)
    result = stream.getvalue()
    if len(result) > _MAX_PAYLOAD_BYTES:
        raise ValueError("Observation payload exceeds the 512 MiB transport limit.")
    return result


def _unpack_window(payload, metadata, instruction):
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        description = _json_loads(archive["metadata"])
        if not isinstance(description, dict) or description.get("camera_keys") != metadata["camera_keys"]:
            raise ValueError("Observation camera names do not match the served policy.")
        expected_keys = {"state", "metadata"} | {f"image_{index}" for index in range(len(metadata["camera_keys"]))}
        if set(archive.files) != expected_keys:
            raise ValueError("Unexpected arrays in the observation payload.")
        window = {
            "state": archive["state"],
            "images": {name: archive[f"image_{index}"] for index, name in enumerate(metadata["camera_keys"])},
            "instruction": description.get("instruction"),
        }
    return _window(window, metadata, instruction)


def _actions(value, metadata):
    actions = np.asarray(value)
    if (
        actions.dtype.kind not in "fiu"
        or actions.ndim != 2
        or actions.shape[1] != metadata["action_dim"]
        or not 1 <= actions.shape[0] <= metadata["action_chunk_size"]
    ):
        raise ValueError("Policy must return a numeric (T, action_dim) action chunk.")
    actions = np.asarray(actions, dtype=np.float32)
    if not np.isfinite(actions).all():
        raise ValueError("Policy returned nonfinite actions.")
    return actions


def _pack_actions(value, metadata):
    stream = BytesIO()
    np.savez(stream, actions=_actions(value, metadata))
    return stream.getvalue()


def _unpack_actions(payload, metadata):
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        if archive.files != ["actions"]:
            raise ValueError("Unexpected arrays in the action payload.")
        return _actions(archive["actions"], metadata)


def _receive(iterator, stop_event, receive_bytes):
    def limited_chunks():
        size = 0
        for chunk in iterator:
            size += len(chunk.data)
            if size > _MAX_PAYLOAD_BYTES:
                raise ValueError("RPC payload exceeds the 512 MiB transport limit.")
            yield chunk

    result = receive_bytes(limited_chunks(), None, stop_event, "wrobo inference")
    if result is None:
        raise ValueError("Incomplete or cancelled RPC payload.")
    return result


class RemotePolicyClient:
    """A cancellable client for a separately loaded wrobo policy server."""

    def __init__(self, target: TargetDevice, *, timeout=30.0, instruction=None):
        if not isinstance(target, TargetDevice) or not target.is_remote:
            raise ValueError("RemotePolicyClient requires a HOST:PORT:DEVICE target.")
        if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Remote timeout must be a positive finite number.")
        if instruction is not None and not isinstance(instruction, str):
            raise ValueError("Instruction must be a string or null.")
        self._grpc, self._messages, options, self._receive_bytes, self._send_bytes = _transport()
        self._timeout = float(timeout)
        self._instruction = instruction
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._calls = set()
        self._channel = self._grpc.insecure_channel(target.address, options=options(enable_retries=False))
        self._handshake = self._channel.unary_unary(
            f"/{_SERVICE}/Handshake",
            request_serializer=self._messages.PolicySetup.SerializeToString,
            response_deserializer=self._messages.PolicySetup.FromString,
        )
        self._predict = self._channel.stream_stream(
            f"/{_SERVICE}/Predict",
            request_serializer=self._messages.Observation.SerializeToString,
            response_deserializer=self._messages.Observation.FromString,
        )
        try:
            response = self._track(
                self._handshake.future(
                    self._messages.PolicySetup(data=_json_bytes({"protocol": _PROTOCOL, "compute_device": target.compute_device})),
                    timeout=self._timeout,
                    wait_for_ready=True,
                )
            )
            try:
                reply = _json_loads(response.result().data)
            finally:
                self._untrack(response)
            if not isinstance(reply, dict) or reply.get("protocol") != _PROTOCOL:
                raise ValueError("Server returned an unsupported inference protocol.")
            if reply.get("compute_device") != target.compute_device:
                raise ValueError("Server inference device does not match the requested device.")
            self.metadata = _metadata(reply.get("metadata"))
            self.metadata["compute_device"] = reply["compute_device"]
        except self._grpc.RpcError as exc:
            self.close()
            raise RuntimeError(f"Remote policy handshake failed ({exc.code().name}): {exc.details()}") from exc
        except BaseException:
            self.close()
            raise

    def _track(self, call):
        with self._lock:
            if self._stop.is_set():
                call.cancel()
                raise RuntimeError("Remote policy client is closed.")
            self._calls.add(call)
        return call

    def _untrack(self, call):
        with self._lock:
            self._calls.discard(call)

    def predict(self, window):
        if self._stop.is_set():
            raise RuntimeError("Remote policy client is closed.")
        default_instruction = self._instruction if self._instruction is not None else self.metadata.get("instruction")
        value = _window(window, self.metadata, default_instruction)
        payload = _pack_window(value, self.metadata)
        call = None
        try:
            call = self._track(
                self._predict(
                    self._send_bytes(payload, self._messages.Observation),
                    timeout=self._timeout,
                )
            )
            return _unpack_actions(_receive(call, self._stop, self._receive_bytes), self.metadata)
        except self._grpc.RpcError as exc:
            raise RuntimeError(f"Remote policy prediction failed ({exc.code().name}): {exc.details()}") from exc
        finally:
            if call is not None:
                call.cancel()
                self._untrack(call)

    def close(self):
        with self._lock:
            self._stop.set()
        self.cancel()
        self._channel.close()

    def cancel(self):
        """Cancel outstanding requests without discarding readable metadata."""
        with self._lock:
            calls = tuple(self._calls)
        for call in calls:
            call.cancel()


class PolicyRPCServer:
    """Serve an already loaded predictor, serializing inference across clients.

    Constructing this object starts no hardware. ``port=0`` is supported for
    local tests. The predictor remains owned by the caller.
    """

    def __init__(self, predictor, *, compute_device, host="127.0.0.1", port=8080, instruction=None):
        if not isinstance(host, str) or not host.strip():
            raise ValueError("Server host must be nonempty.")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("Server port must be between 0 and 65535.")
        if instruction is not None and not isinstance(instruction, str):
            raise ValueError("Instruction must be a string or null.")
        self.compute_device = canonical_compute_device(compute_device)
        self.metadata = _metadata(predictor.metadata)
        actual_device = self.metadata.get("compute_device")
        if actual_device is not None and canonical_compute_device(actual_device) != self.compute_device:
            raise ValueError("Predictor device does not match the server device.")
        self.metadata["compute_device"] = self.compute_device
        self._instruction = instruction if instruction is not None else self.metadata.get("instruction")
        self.metadata["instruction"] = self._instruction
        self._predictor = predictor
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._grpc, self._messages, options, self._receive_bytes, self._send_bytes = _transport()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wrobo-policy-rpc")
        self._server = self._grpc.server(self._executor, options=options(enable_retries=False))
        handlers = {
            "Handshake": self._grpc.unary_unary_rpc_method_handler(
                self._handle_handshake,
                request_deserializer=self._messages.PolicySetup.FromString,
                response_serializer=self._messages.PolicySetup.SerializeToString,
            ),
            "Predict": self._grpc.stream_stream_rpc_method_handler(
                self._handle_predict,
                request_deserializer=self._messages.Observation.FromString,
                response_serializer=self._messages.Observation.SerializeToString,
            ),
        }
        self._server.add_generic_rpc_handlers((self._grpc.method_handlers_generic_handler(_SERVICE, handlers),))
        address = f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"
        self.port = self._server.add_insecure_port(address)
        if self.port == 0:
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError(f"Could not bind policy server to {address}.")
        self._server.start()

    def _handle_handshake(self, request, context):
        try:
            specs = _json_loads(request.data)
            if not isinstance(specs, dict) or specs.get("protocol") != _PROTOCOL:
                raise ValueError("Client inference protocol is unsupported.")
            if specs.get("compute_device") != self.compute_device:
                raise ValueError(f"Requested device {specs.get('compute_device')!r} does not match server device {self.compute_device!r}.")
            return self._messages.PolicySetup(data=_json_bytes({
                "protocol": _PROTOCOL,
                "compute_device": self.compute_device,
                "metadata": self.metadata,
            }))
        except (ValueError, TypeError) as exc:
            context.abort(self._grpc.StatusCode.INVALID_ARGUMENT, str(exc))

    def _handle_predict(self, iterator, context):
        cancelled = threading.Event()
        context.add_callback(cancelled.set)
        try:
            payload = _receive(iterator, cancelled, self._receive_bytes)
            window = _unpack_window(payload, self.metadata, self._instruction)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            context.abort(self._grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        with self._lock:
            if self._stop.is_set() or not context.is_active():
                context.abort(self._grpc.StatusCode.CANCELLED, "Inference request cancelled.")
            try:
                response = _pack_actions(self._predictor.predict(window), self.metadata)
            except Exception as exc:
                context.abort(self._grpc.StatusCode.INTERNAL, f"Policy prediction failed: {type(exc).__name__}: {exc}")
        if not context.is_active():
            context.abort(self._grpc.StatusCode.CANCELLED, "Inference request cancelled.")
        yield from self._send_bytes(response, self._messages.Observation)

    def wait(self):
        self._server.wait_for_termination()

    def close(self):
        self._stop.set()
        self._server.stop(grace=0).wait()
        self._executor.shutdown(wait=True, cancel_futures=True)
