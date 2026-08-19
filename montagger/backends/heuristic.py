"""Zero-dependency backend for smoke tests and hardware-less demos.

Derives simple general-category tags from colour and geometry statistics so
the whole chain (download -> preprocess -> infer -> write tags -> WebUI) can
be exercised without an ONNX model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from montagger.backends import Backend, register


@register("heuristic")
class HeuristicBackend(Backend):
    name = "heuristic"
    _SIZE = 64

    def preprocess(self, image: Image.Image) -> tuple[np.ndarray, int, int]:
        width, height = image.size
        small = image.convert("RGB").resize((self._SIZE, self._SIZE), Image.BILINEAR)
        return np.asarray(small, dtype=np.float32) / 255.0, width, height

    def tag(self, prepared: Any, runtime: Any) -> list[str]:
        del runtime
        arr, width, height = prepared
        tags: list[str] = []

        aspect = width / max(height, 1)
        if aspect < 0.8:
            tags.append("portrait")
        elif aspect > 1.25:
            tags.append("landscape")
        else:
            tags.append("square")

        brightness = float(arr.mean())
        if brightness > 0.6:
            tags.append("bright")
        elif brightness < 0.25:
            tags.append("dark")

        saturation = float(np.std(arr, axis=2).mean())
        if saturation < 0.08:
            tags.append("grayscale")
        elif saturation > 0.3:
            tags.append("colorful")

        if width >= 2560 or height >= 2560:
            tags.append("large")
        elif width < 800 and height < 800:
            tags.append("small")

        return tags