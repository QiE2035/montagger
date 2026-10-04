"""In-memory image preprocessing, ported from monbooru's tagger/preprocess.go.

Everything runs on bytes → numpy; no file ever touches the disk. Go's
image.RGBA stores alpha-premultiplied pixels and the tensor builder reads
those RGB bytes while ignoring alpha, so the port premultiplies explicitly -
that (not alpha compositing) is what keeps transparent PNG art scoring the
same in both programs.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

from .profile import Profile

# OpenAI CLIP values, kept verbatim from joytag's preprocess.
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# camieDefaultFill: the padding colour of Camie's onnx_inference.py.
_CAMIE_FILL = np.array([124, 116, 104], dtype=np.uint8)


class NotAnImage(ValueError):
    """The bytes do not decode as any image PIL knows."""


def decode(data: bytes) -> Image.Image:
    """Decode bytes to RGBA with the EXIF orientation applied. Animated
    formats land on their first frame (seek(0) is the default frame)."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGBA")
    except Exception as err:  # PIL raises a zoo of decoder errors
        raise NotAnImage(str(err)) from err


def pad_and_resize(rgba: Image.Image, size: int, profile: Profile) -> np.ndarray:
    """Aspect-preserving bilinear resize, centered on a square canvas.

    The canvas starts filled with the pad colour; a white pad doubles as the
    alpha==0 → white rule of the Go original, because the source is
    premultiplied first and pasted without compositing.
    """
    arr = np.asarray(rgba, dtype=np.float32)
    alpha = arr[..., 3:4] / 255.0
    premul = np.rint(arr[..., :3] * alpha)
    premul = premul.clip(0, 255).astype(np.uint8)

    w, h = rgba.width, rgba.height
    scale_w, scale_h = size, size
    if w >= h:
        scale_h = max(1, h * size // w)
    else:
        scale_w = max(1, w * size // h)
    # Truncating division matches Go's integer arithmetic; PIL's BILINEAR is
    # the same family of filter as x/image/draw.BiLinear.
    scaled = np.asarray(
        Image.fromarray(premul).resize((scale_w, scale_h), Image.BILINEAR),
        dtype=np.uint8,
    )

    if profile.pad == "mean_color_aspect":
        fill = np.array(profile.fill_color, dtype=np.uint8)
        if not fill.any():
            fill = _CAMIE_FILL
    else:
        fill = np.array([255, 255, 255], dtype=np.uint8)
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:] = fill

    off_x = (size - scale_w) // 2
    off_y = (size - scale_h) // 2
    canvas[off_y : off_y + scale_h, off_x : off_x + scale_w] = scaled
    return canvas


def build_tensor(canvas: np.ndarray, size: int, profile: Profile) -> np.ndarray:
    """Canvas → float32 input tensor in the profile's layout/channels/normalize."""
    if profile.layout == "nhwc":
        if profile.normalize not in ("", "none"):
            raise ValueError(f"buildTensor: NHWC + normalize={profile.normalize!r} is not implemented")
        channels = canvas[..., ::-1] if profile.channels == "bgr" else canvas
        return channels.astype(np.float32)[np.newaxis, ...]

    if profile.layout == "nchw":
        rgb = canvas.astype(np.float32)
        if profile.channels == "bgr":
            rgb = rgb[..., ::-1]
        if profile.normalize in ("imagenet", "clip"):
            mean = _IMAGENET_MEAN if profile.normalize == "imagenet" else _CLIP_MEAN
            std = _IMAGENET_STD if profile.normalize == "imagenet" else _CLIP_STD
            rgb = (rgb / 255.0 - mean) / std
        elif profile.normalize == "div255":
            rgb = rgb / 255.0
        # Channel planes in the final (post-swap) order, matching the Go
        # builder's per-channel writes.
        planes = [np.ascontiguousarray(rgb[..., i]) for i in range(3)]
        return np.stack(planes)[np.newaxis, ...]

    raise ValueError(f"buildTensor: unsupported layout {profile.layout!r}")


def preprocess(data: bytes, size: int, profile: Profile) -> np.ndarray:
    """Bytes → model-ready float32 tensor of shape (1, ...) per the profile."""
    return build_tensor(pad_and_resize(decode(data), size, profile), size, profile)


def infer_input_size(dims, layout: str) -> int:
    """Port of inferInputSize: the spatial edge from a 4-D input shape.
    Symbolic dimensions (strings) read as unknown."""
    try:
        dims = list(dims)
    except TypeError:
        return 0
    if len(dims) != 4:
        return 0
    d = dims[1] if layout == "nhwc" else dims[2] if layout == "nchw" else 0
    if not isinstance(d, int) or d <= 0:
        return 0
    return d
