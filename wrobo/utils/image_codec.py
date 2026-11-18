"""
Shared image (de)coding for episode HDF5 files.

Images can be stored in three ways inside an episode file:

1. "raw"  : a dense uint8 array of shape (T, C, H, W), no compression.
2. "gzip" : a dense uint8 array of shape (T, C, H, W), gzip-compressed. (Historically
   this layout was tagged "raw"; old files keep reading since both are dense arrays
   and gzip is transparent on read.)
3. "png" / "jpeg" : a variable-length array of length T, where each element is the
   1-D uint8 byte string of one PNG/JPEG-encoded frame. This trades a small amount
   of CPU (decode on read) for a large reduction on disk (typically 10-30x+ vs the
   gzip'd dense layout), and is the layout used by most embodied datasets.

Conventions (kept identical on the write and read sides, so producers and the
dataset/stats readers never disagree):
- In-memory frames are RGB, channel-first (C, H, W), uint8 -- the same layout the
  dense layouts use and that the rest of the pipeline expects.
- On disk the encoded bytes are standard PNG/JPEG, which OpenCV handles in BGR; we
  convert RGB<->BGR around encode/decode so the decoded frame matches the original.

The chosen encoding is recorded on the HDF5 dataset via the "img_encoding" attribute
("raw" / "gzip" / "png" / "jpeg"). Readers branch via is_encoded() -- only the
per-frame byte-string layouts (png/jpeg) need decoding; dense layouts (raw/gzip) are
read directly. Old files keep working: a missing attribute is treated as a dense
layout, and the legacy "jpg" tag is normalized to "jpeg".
"""

from typing import List

import cv2
import h5py
import numpy as np

# attribute name used to tag how an image dataset is encoded
ENCODING_ATTR = "img_encoding"
RAW = "raw"  # dense uint8 array, uncompressed
GZIP = "gzip"  # dense uint8 array, gzip-compressed (was historically "raw")
PNG = "png"  # per-frame PNG byte strings
JPEG = "jpeg"  # per-frame JPEG byte strings (was "jpg")

_EXT = {PNG: ".png", JPEG: ".jpg"}
_ENCODED = set(_EXT)  # layouts stored as per-frame byte strings (need decode on read)
_DENSE = {RAW, GZIP}  # layouts stored as a dense 4-D array (read directly)

# legacy encoding tags -> current names, applied on read so old datasets keep working.
_ALIASES = {"jpg": JPEG}


def is_encoded(encoding: str) -> bool:
    """True if the layout stores per-frame byte strings (png/jpeg) that must be decoded;
    False for dense array layouts (raw/gzip), which are read directly."""
    return encoding in _ENCODED


def encode_frame(frame_chw: np.ndarray, encoding: str, jpeg_quality: int = 95) -> np.ndarray:
    """
    Encode a single (C, H, W) uint8 RGB frame to a 1-D uint8 byte string.

    Args:
        frame_chw: (C, H, W) uint8 RGB frame.
        encoding: "png" or "jpeg".
        jpeg_quality: JPEG quality in [0, 100] (ignored for png).
    Returns:
        1-D uint8 np.ndarray of the encoded bytes.
    """
    if encoding not in _EXT:
        raise ValueError(f"encode_frame only supports {list(_EXT)}, got '{encoding}'")
    hwc_rgb = np.transpose(frame_chw, (1, 2, 0))  # (H, W, C)
    bgr = cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR)
    if encoding == JPEG:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    else:
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    ok, buf = cv2.imencode(_EXT[encoding], bgr, params)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for encoding '{encoding}'")
    return buf.reshape(-1).astype(np.uint8)


def decode_frame(buf: np.ndarray) -> np.ndarray:
    """
    Decode a single encoded byte string back to a (C, H, W) uint8 RGB frame.
    """
    buf = np.frombuffer(np.asarray(buf, dtype=np.uint8), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed: corrupt or unsupported image bytes")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.transpose(rgb, (2, 0, 1))  # (C, H, W)


def get_encoding(dataset: h5py.Dataset) -> str:
    """Return the encoding tag of an image dataset, normalized to current names.
    Missing attr -> 'gzip' (legacy dense layout); legacy 'jpg' -> 'jpeg'."""
    enc = dataset.attrs.get(ENCODING_ATTR, GZIP)
    if isinstance(enc, bytes):
        enc = enc.decode()
    enc = str(enc)
    return _ALIASES.get(enc, enc)


def decode_frames(frames) -> np.ndarray:
    """
    Decode a sequence of encoded byte strings to a dense (n, C, H, W) uint8 array.
    `frames` is the object/vlen array read from an encoded image dataset.
    """
    decoded: List[np.ndarray] = [decode_frame(f) for f in frames]
    return np.stack(decoded, axis=0)


def create_image_dataset(
    group: h5py.Group,
    name: str,
    frames_chw,
    encoding: str,
    jpeg_quality: int = 95,
    gzip_opts: int = 4,
):
    """
    Create an image dataset under `group`, encoded per `encoding`.

    Args:
        group: target h5py group (e.g. the "observations" group).
        name: dataset name (e.g. "image_top").
        frames_chw: sequence/array of T frames, each (C, H, W) uint8 RGB.
        encoding: "raw" (dense, uncompressed), "gzip" (dense, gzip'd) or
            "png"/"jpeg" (per-frame byte strings).
        jpeg_quality: JPEG quality (only used for "jpeg").
        gzip_opts: gzip level for the "gzip" layout (unused otherwise; variable-length
            data lives in the global heap where filters don't apply).
    Returns:
        The created h5py.Dataset.
    """
    frames_chw = [np.asarray(f) for f in frames_chw]
    T = len(frames_chw)
    C, H, W = frames_chw[0].shape

    if encoding in _DENSE:
        arr = np.stack(frames_chw, axis=0)  # (T, C, H, W)
        if encoding == GZIP:
            ds = group.create_dataset(
                name,
                data=arr,
                dtype="uint8",
                compression="gzip",
                compression_opts=gzip_opts,
                shuffle=True,
                chunks=(1, C, H, W),
            )
        else:  # RAW: dense, no compression
            ds = group.create_dataset(
                name,
                data=arr,
                dtype="uint8",
                chunks=(1, C, H, W),
            )
    elif encoding in _EXT:
        # variable-length uint8 byte strings, one chunk per frame for fast random
        # single-frame reads (the dataset/stats access pattern).
        vlen = h5py.vlen_dtype(np.uint8)
        ds = group.create_dataset(name, shape=(T,), dtype=vlen, chunks=(1,))
        for i, frame in enumerate(frames_chw):
            ds[i] = encode_frame(frame, encoding, jpeg_quality=jpeg_quality)
    else:
        raise ValueError(
            f"Unknown encoding '{encoding}' (expected raw/gzip/png/jpeg)"
        )

    # tag encoding + original geometry so readers can branch and sanity-check.
    ds.attrs[ENCODING_ATTR] = encoding
    ds.attrs["frame_shape_chw"] = np.array([C, H, W], dtype=np.int64)
    return ds
