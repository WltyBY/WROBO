"""Nonblocking arrow-key edges, filtering native keyboard auto-repeat."""

from collections import deque
import sys
import threading
from time import monotonic


class ArrowKeys:
    def __init__(self, clock=monotonic, *, debug=False):
        self._lock = threading.Lock()
        self._held = set()
        self._released = {}
        self._native_timestamps = {}
        self._clock = clock
        self._started = clock()
        self._debug_events = deque(maxlen=256) if debug else None
        self._debug_dropped = 0
        self._events = deque(maxlen=1)
        self._listener = None
        self._mapping = {}

    def _trace(self, action, key, outcome, timestamp=None, repeat=False):
        # Called with the lock held. Printing stays on the control thread, so
        # a slow terminal cannot delay the native keyboard callback.
        if self._debug_events is not None:
            if len(self._debug_events) == self._debug_events.maxlen:
                self._debug_dropped += 1
            self._debug_events.append((
                self._clock() - self._started, action, self._mapping[key],
                outcome, timestamp, repeat,
            ))

    def _flush_debug(self):
        if self._debug_events is None:
            return
        with self._lock:
            events = list(self._debug_events)
            self._debug_events.clear()
            dropped, self._debug_dropped = self._debug_dropped, 0
        if dropped:
            print(f"[keyboard] {dropped} older arrow debug events omitted.", flush=True)
        for elapsed, action, name, outcome, timestamp, repeat in events:
            print(
                f"[keyboard +{elapsed:.3f}s] {action} {name}: {outcome}; "
                f"repeat={repeat}, native_time={timestamp}", flush=True,
            )

    def _accept_timestamp(self, key, action, timestamp, repeat=False):
        # Called with the lock held, for Darwin's monotonic native timestamps.
        # Untimestamped events retain the ordinary held/repeat filtering.
        if timestamp is not None and timestamp > 0:
            previous = self._native_timestamps.get(key, 0)
            if timestamp <= previous:
                self._trace(
                    action, key, f"ignored stale/replayed event (last native_time={previous})",
                    timestamp, repeat,
                )
                return False
            self._native_timestamps[key] = timestamp
        return True

    def _press(self, key, *, repeat=False, timestamp=None, reject_stale=False):
        name = self._mapping.get(key)
        if name is not None:
            with self._lock:
                if reject_stale and not self._accept_timestamp(key, "down", timestamp, repeat):
                    return
                if repeat:
                    self._trace("down", key, "ignored auto-repeat", timestamp, repeat)
                    return
                released = self._released.pop(key, None)
                # X11 synthesizes release/press pairs with the same server
                # timestamp. Callback scheduling delays do not make them a
                # new physical press. Other platforms release held immediately.
                if released is not None and timestamp != released:
                    self._held.discard(key)
                if key not in self._held:
                    self._held.add(key)
                    self._events.append(name)
                    self._trace("down", key, "accepted", timestamp)
                else:
                    self._trace("down", key, "ignored held key", timestamp)

    def _release(self, key, *, timestamp=None, reject_stale=False):
        if key not in self._mapping:
            return
        with self._lock:
            if reject_stale and not self._accept_timestamp(key, "up", timestamp):
                return
            # Darwin has explicit auto-repeat flags, so a fresh release can
            # rearm immediately. X11 still needs its release/press pair check.
            if timestamp is None or reject_stale:
                self._held.discard(key)
                self._released.pop(key, None)
            else:
                self._released[key] = timestamp
            self._trace("up", key, "released", timestamp)

    def clear(self):
        # Never clear held keys: changing state must still require a release.
        with self._lock:
            self._events.clear()

    def poll(self):
        self._flush_debug()
        if self._listener is not None and not self._listener.is_alive():
            raise RuntimeError("Keyboard listener stopped; stopping robot control")
        with self._lock:
            return self._events.popleft() if self._events else None

    def _make_listener(self, keyboard):
        if sys.platform == "darwin":
            from Quartz import (
                CGEventGetIntegerValueField,
                CGEventGetTimestamp,
                kCGEventKeyDown,
                kCGEventKeyUp,
                kCGKeyboardEventAutorepeat,
                kCGKeyboardEventKeycode,
            )

            codes = {key.value.vk: key for key in self._mapping}

            def intercept(event_type, event):
                if event_type in (kCGEventKeyDown, kCGEventKeyUp):
                    code = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
                    key = codes.get(code)
                    if key is not None:
                        timestamp = CGEventGetTimestamp(event) or None
                        if event_type == kCGEventKeyDown:
                            self._press(key, repeat=bool(CGEventGetIntegerValueField(
                                event, kCGKeyboardEventAutorepeat,
                            )), timestamp=timestamp, reject_stale=True)
                        else:
                            self._release(key, timestamp=timestamp, reject_stale=True)
                return event  # Leave terminal and other apps' input intact.

            # pynput runs ordinary callbacks BEFORE darwin_intercept. Register
            # only the intercept, so unfiltered repeats never reach _press.
            return keyboard.Listener(darwin_intercept=intercept)

        if keyboard.Listener.__module__.endswith("._xorg"):
            from Xlib import X

            owner = self

            class TimestampedListener(keyboard.Listener):
                def _handle_message(self, display, event, injected=False):
                    try:
                        key = self._event_to_key(display, event)
                    except IndexError:
                        return
                    if event.type == X.KeyPress:
                        owner._press(key, timestamp=event.time)
                    elif event.type == X.KeyRelease:
                        owner._release(key, timestamp=event.time)

            return TimestampedListener()

        # Keep callback signatures positional: pynput 1.8 counts keyword-only
        # parameters when adapting callbacks to its optional injected argument.
        return keyboard.Listener(
            on_press=lambda key: self._press(key),
            on_release=lambda key: self._release(key),
        )

    def __enter__(self):
        try:
            from lerobot.utils.keyboard_input import (
                pynput_can_capture,
                pynput_listener_is_trusted,
            )

            if not pynput_can_capture():
                raise RuntimeError(
                    "Collection needs keyboard release events: use a local X11/Windows/macOS session, not headless SSH or Wayland"
                )
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError(
                "Arrow-key collection requires pynput and a local graphical keyboard session"
            ) from exc
        self._mapping = {
            keyboard.Key.right: "right",
            keyboard.Key.left: "left",
            keyboard.Key.up: "up",
            keyboard.Key.down: "down",
        }
        self._listener = self._make_listener(keyboard)
        try:
            self._listener.start()
            self._listener.wait()
            if not pynput_listener_is_trusted(self._listener):
                message = "Keyboard capture permission denied; grant input/accessibility permission"
                if sys.platform == "darwin":
                    message += (
                        "; macOS: open System Settings > Privacy & Security > Accessibility "
                        "and enable the app running this command (Terminal, iTerm, or your IDE). "
                        "Also check Input Monitoring. Fully quit and reopen that app, "
                        "then rerun the command. If permission is still denied, "
                        f"check the Python executable: {sys.executable}"
                    )
                raise RuntimeError(message)
        except BaseException:
            self._listener.stop()
            raise
        return self

    def __exit__(self, *exc):
        self._listener.stop()
        self._listener.join(timeout=1)
        self._flush_debug()
