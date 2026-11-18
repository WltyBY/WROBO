"""Display camera frames in a shared grid without changing capture images."""

from collections.abc import Mapping
import math
import os
import re
import sys
from typing import Any

import numpy as np


def _tile_dimensions(tile_width: int, tile_height: int) -> None:
    for name, value in (("tile_width", tile_width), ("tile_height", tile_height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")


def _wrap_text(cv2, text: str, max_width: int, scale: float) -> list[str]:
    """Wrap even long device paths, using the actual OpenCV text width."""
    lines = []
    for paragraph in text.splitlines() or [""]:
        while paragraph:
            low, high = 1, len(paragraph)
            while low < high:
                middle = (low + high + 1) // 2
                width = cv2.getTextSize(
                    paragraph[:middle], cv2.FONT_HERSHEY_SIMPLEX, scale, 1,
                )[0][0]
                if width <= max_width:
                    low = middle
                else:
                    high = middle - 1
            lines.append(paragraph[:low])
            paragraph = paragraph[low:]
    return lines or [""]


def _annotate(cv2, tile: np.ndarray, labels: list[str]) -> None:
    height, width = tile.shape[:2]
    margin = min(8, max(1, min(width, height) // 12))
    padding = min(5, margin)
    available_width = max(1, width - 2 * (margin + padding))
    scale = min(0.5, available_width / 160, height / 100)
    # A very long path can require more lines. Shrink the complete annotation
    # until it fits, retaining every setting instead of drawing beyond a tile.
    for _ in range(20):
        lines = [line for label in labels
                 for line in _wrap_text(cv2, label, available_width, scale)]
        (_, text_height), baseline = cv2.getTextSize(
            "Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, 1,
        )
        line_height = max(1, text_height + baseline + 3)
        if len(lines) * line_height + 2 * padding <= height - 2 * margin:
            break
        scale *= 0.85

    max_lines = max(1, (height - 2 * (margin + padding)) // line_height)
    if len(lines) > max_lines:
        # Extremely small user-specified tiles still keep all pixels in bounds.
        lines = lines[:max_lines]
        lines[-1] = _wrap_text(cv2, "...", available_width, scale)[0]
    text_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
                  for line in lines]
    panel_width = min(width - margin, max(size[0] for size in text_sizes) + 2 * padding)
    panel_height = min(height - margin, len(lines) * line_height + 2 * padding)
    right = width - margin
    left = max(0, right - panel_width)
    top = margin
    bottom = min(height, top + panel_height)
    panel = tile[top:bottom, left:right]
    panel[:] = (panel.astype(np.float32) * 0.3).astype(np.uint8)
    for index, (line, size) in enumerate(zip(lines, text_sizes)):
        x = max(left, right - padding - size[0])
        y = top + padding + text_height + index * line_height
        cv2.putText(tile, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)


def compose_camera_grid(
    frames: Mapping[str, np.ndarray],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    tile_width: int = 640,
    tile_height: int = 480,
) -> np.ndarray:
    """Return a BGR grid of equal-size tiles, letterboxed and annotated.

    Frames are RGB unless their metadata specifies ``color_mode: bgr``. The
    input arrays remain untouched so a caller can record their original frames.
    Tiles follow mapping order, with at most two columns; an unused last tile is
    black. This function uses no GUI APIs and also works with headless OpenCV.
    """
    _tile_dimensions(tile_width, tile_height)
    if not frames:
        raise ValueError("Camera preview requires at least one frame.")

    import cv2

    columns = min(2, len(frames))
    rows = math.ceil(len(frames) / columns)
    grid = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, (name, frame) in enumerate(frames.items()):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or \
                frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0:
            raise ValueError(f"Camera {name!r} must provide a nonempty uint8 HxWx3 image.")
        settings = metadata.get(name, {})
        color_mode = settings.get("color_mode", "rgb")
        color_mode = str(getattr(color_mode, "value", color_mode)).lower()
        if color_mode not in ("rgb", "bgr"):
            raise ValueError(f"Camera {name!r} color_mode must be 'rgb' or 'bgr'.")
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if color_mode == "rgb" else frame
        source_height, source_width = frame.shape[:2]
        ratio = min(tile_width / source_width, tile_height / source_height)
        target_width = max(1, min(tile_width, round(source_width * ratio)))
        target_height = max(1, min(tile_height, round(source_height * ratio)))
        interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
        tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        x = (tile_width - target_width) // 2
        y = (tile_height - target_height) // 2
        tile[y:y + target_height, x:x + target_width] = resized
        fps = settings.get("fps")
        labels = [str(name), f"device={settings.get('device', '?')}",
                  f"{source_width}x{source_height} | rotation={settings.get('rotation', 0)}"
                  f" | fps={'auto' if fps is None else fps}"]
        _annotate(cv2, tile, labels)
        row, column = divmod(index, columns)
        grid[row * tile_height:(row + 1) * tile_height,
             column * tile_width:(column + 1) * tile_width] = tile
    return grid


class CameraPreview:
    """Own one GUI window, stopping on q, Escape, or the window close button."""

    def __init__(
        self,
        window_name: str = "WROBO cameras",
        *,
        tile_width: int = 640,
        tile_height: int = 480,
    ) -> None:
        _tile_dimensions(tile_width, tile_height)
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Camera preview requires GUI OpenCV (opencv-python).") from exc
        gui = re.search(r"^\s*GUI:\s*(\S+)", cv2.getBuildInformation(), re.MULTILINE)
        if gui and gui.group(1).upper() in ("NONE", "NO"):
            raise RuntimeError(
                "Camera preview requires GUI OpenCV. Uninstall opencv-python-headless "
                "and reinstall opencv-python."
            )
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            raise RuntimeError("Camera preview requires GUI OpenCV and a graphical display.")
        self._cv2 = cv2
        self.window_name = window_name
        self.tile_width = tile_width
        self.tile_height = tile_height
        self._created = False
        self._stopped = False

    def _window_visible(self) -> bool:
        try:
            return self._cv2.getWindowProperty(
                self.window_name, self._cv2.WND_PROP_VISIBLE,
            ) >= 1
        except self._cv2.error:
            # Cocoa raises when querying a window the user has already closed.
            return False

    def show(
        self,
        frames: Mapping[str, np.ndarray],
        metadata: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        """Show one composite frame; return False when the preview is stopped."""
        if not frames:
            raise ValueError("Camera preview requires at least one frame.")
        if self._stopped:
            return False
        if self._created and not self._window_visible():
            self._stopped = True
            return False
        grid = compose_camera_grid(frames, metadata, tile_width=self.tile_width,
                                   tile_height=self.tile_height)
        try:
            if not self._created:
                self._cv2.namedWindow(self.window_name, self._cv2.WINDOW_NORMAL)
                self._created = True
            self._cv2.imshow(self.window_name, grid)
            key = self._cv2.waitKey(1) & 0xFF
        except self._cv2.error as exc:
            raise RuntimeError(
                "Could not display camera preview. Use GUI OpenCV (opencv-python) "
                "in a graphical desktop session."
            ) from exc
        if key in (ord("q"), 27) or not self._window_visible():
            self._stopped = True
        return not self._stopped

    def close(self) -> None:
        """Release only this preview's window; repeated calls are harmless."""
        self._stopped = True
        if self._created:
            self._created = False
            try:
                self._cv2.destroyWindow(self.window_name)
            except self._cv2.error:
                pass
