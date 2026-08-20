"""ONNX backend, WD14-family compatible (wd-swinv2, camie-v2, joytag, ...).

A model folder holds one .onnx plus a label file: either a wd14-style csv
(tags carry rating_/general_/character_/copyright_/artist_ prefixes) or a
plain txt (one tag per line, e.g. joytag). The tag list is aligned with the
model's logits by count - a stray csv header row is dropped automatically.

Execution providers are hot-swappable: reload() rebuilds the InferenceSession
for the provider named by the runtime (cpu/cuda/directml/openvino/coreml),
falling back to CPU when the requested provider is not installed.
"""

from __future__ import annotations

import csv
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - the package ships onnxruntime
    ort = None  # type: ignore[assignment]

from montagger.backends import Backend, register
from montagger.settings import DEFAULT_CATEGORY_MAP, EP_ALIASES

log = logging.getLogger(__name__)

DEFAULT_INPUT_SIZE = 448
WD14_PREFIXES = ("rating_", "general_", "character_", "copyright_", "artist_")


@dataclass
class _Label:
    tag: str
    category: str  # "" = general, else a monbooru category name
    monbooru_tag: str  # precomputed: "tag" or "category:tag"


def _parse_labels(path: Path, output_n: int) -> list[_Label] | None:
    """Read csv/txt labels and align them with output_n logits."""
    rows: list[str] = []
    try:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8", errors="replace") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if row and row[0].strip():
                        rows.append(row[0].strip())
        else:
            rows = [
                line.strip()
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
    except OSError as exc:
        log.warning("cannot read tag file %s: %s", path, exc)
        return None

    labels = rows
    if len(labels) > output_n:
        # A csv header row (e.g. "tag,count") pushed the count past the
        # logits; drop the first row and re-check.
        labels = labels[1:]
    if len(labels) != output_n:
        log.warning(
            "tag count mismatch for %s: model outputs %d but %s declares %d",
            path, output_n, path.name, len(rows),
        )
        return None
    return [_Label(tag=t, category="", monbooru_tag=t) for t in labels]


def _apply_categories(labels: list[_Label], valid: set[str]) -> None:
    """Tag wd14 prefixes with their monbooru category when that category
    exists; everything else stays general."""
    for label in labels:
        for prefix in WD14_PREFIXES:
            if label.tag.startswith(prefix):
                category = prefix[:-1]  # "general" -> ""
                mapped = DEFAULT_CATEGORY_MAP.get(category, "")
                if not mapped or mapped in valid:
                    rest = label.tag[len(prefix):]
                    label.category = mapped
                    label.monbooru_tag = rest if not mapped else f"{mapped}:{rest}"
                break


@register("onnx")
def _make_onnx(runtime: Any, deps: Any) -> "OnnxBackend":
    return OnnxBackend(runtime, deps)


class OnnxBackend(Backend):
    name = "onnx"

    def __init__(self, runtime: Any, deps: Any) -> None:
        if ort is None:  # pragma: no cover
            raise RuntimeError("onnxruntime is not installed")
        self.runtime = runtime
        self.client = deps.get("client")
        self.model_dir = Path(getattr(runtime, "model_dir", "."))
        self.activation = getattr(runtime, "activation", "sigmoid_in_model")
        self._lock = threading.RLock()
        self._session: Any = None
        self._labels: list[_Label] = []
        self._input_name = ""
        self._input_layout = "nchw"
        self._input_size = DEFAULT_INPUT_SIZE
        self._output_index = 0
        self._valid_categories = set(DEFAULT_CATEGORY_MAP.values())
        self.load()

    # ---- model loading --------------------------------------------------

    def _find_model(self) -> Path:
        candidates = sorted(self.model_dir.glob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"no .onnx model in {self.model_dir}")
        return candidates[0]

    def _find_tags(self) -> Path | None:
        for ext in ("csv", "txt"):
            found = sorted(self.model_dir.glob(f"*.{ext}"))
            if found:
                return found[0]
        return None

    def load(self) -> None:
        with self._lock:
            self._close_unlocked()
            model = self._find_model()
            providers = self._providers()
            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3
            self._session = ort.InferenceSession(str(model), sess_opts, providers=providers)
            used = self._session.get_providers()
            log.info("onnx backend %s loaded with providers: %s", model.name, used)

            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            shape = list(inp.shape)
            if len(shape) == 4:
                if shape[1] in (1, 3) and shape[1] != shape[-1]:
                    self._input_layout = "nchw"
                    size = shape[2]
                else:
                    self._input_layout = "nhwc"
                    size = shape[1]
                self._input_size = size if isinstance(size, int) and size > 0 else DEFAULT_INPUT_SIZE
            out_shape = self._session.get_outputs()[0].shape
            output_n = int(out_shape[-1]) if out_shape and out_shape[-1] not in (-1, None) else 0

            tags_path = self._find_tags()
            if tags_path is None:
                raise FileNotFoundError(f"no .csv/.txt tag file in {self.model_dir}")
            labels = _parse_labels(tags_path, output_n)
            if labels is None:
                raise RuntimeError(f"cannot align tag file {tags_path.name} with model output")
            self._refresh_categories()
            _apply_categories(labels, self._valid_categories)
            self._labels = labels

    def _refresh_categories(self) -> None:
        if self.client is None:
            return
        try:
            names = set(self.client.categories())
            if names:
                self._valid_categories = names
        except Exception:
            log.debug("category refresh failed; keeping defaults", exc_info=True)

    def _providers(self) -> list[str]:
        wanted = EP_ALIASES.get(self.runtime.ep, "CPUExecutionProvider")
        available = list(ort.get_available_providers())
        if wanted in available:
            return [wanted, "CPUExecutionProvider"]
        if wanted != "CPUExecutionProvider":
            log.warning(
                "provider %s is not available (installed: %s); using CPU",
                wanted, available,
            )
        return ["CPUExecutionProvider"]

    def reload(self, runtime: Any) -> None:
        """Hot-reload: provider, model folder and activation all come from
        the runtime, so a settings save that touches any of them rebuilds
        the session (falling back to CPU on an unknown provider)."""
        self.runtime = runtime
        self.model_dir = Path(getattr(runtime, "model_dir", self.model_dir))
        self.activation = getattr(runtime, "activation", self.activation)
        self.load()

    # ---- preprocessing & inference ---------------------------------------

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB")
        width, height = img.size
        side = max(width, height)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(img, ((side - width) // 2, (side - height) // 2))
        canvas = canvas.resize((self._input_size, self._input_size), Image.BILINEAR)
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        if self._input_layout == "nchw":
            arr = arr.transpose(2, 0, 1)
        return arr[None]  # batch of 1

    def tag(self, prepared: Any, runtime: Any) -> list[str]:
        with self._lock:
            session = self._session
            labels = self._labels
        logits = np.asarray(session.run(None, {self._input_name: prepared})[self._output_index])
        probs = logits.reshape(-1)
        if self.activation == "logits" or float(probs.max()) > 1.0:
            probs = 1.0 / (1.0 + np.exp(-probs))

        picked: list[tuple[float, _Label]] = []
        for index, label in enumerate(labels):
            if index >= probs.size:
                break
            floor = (
                runtime.character_threshold
                if label.category == "character"
                else runtime.threshold
            )
            if float(probs[index]) >= floor:
                picked.append((float(probs[index]), label))
        picked.sort(key=lambda item: item[0], reverse=True)
        topk = getattr(runtime, "general_topk", 40)
        general_seen = 0
        result: list[str] = []
        for score, label in picked:
            if not label.category:
                general_seen += 1
                if general_seen > topk:
                    continue
            result.append(label.monbooru_tag)
        return result

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        if self._session is not None:
            try:
                self._session = None
            except Exception:  # pragma: no cover
                pass
            self._session = None