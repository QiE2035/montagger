"""montagger.toml: loaded once, swapped atomically on save, persisted in full.

The file lives next to the checkout (repo root). A missing file is created
with defaults on first start; an existing file keeps its values and gains
nothing - every read fills blanks with defaults so an old config upgrades
silently.
"""

from __future__ import annotations

import os
import secrets
import threading
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

import tomli_w

from . import logx

log = logx.get("config")

DEFAULT_HF_ENDPOINT = "https://huggingface.co"

# Execution providers montagger knows how to configure, mapped to the
# onnxruntime provider names they request. Anything the installed
# onnxruntime build offers beyond this map is still selectable by its short
# name (the part before "ExecutionProvider", lowercased).
PROVIDER_NAMES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "azure": "AzureExecutionProvider",
    "nnapi": "NnapiExecutionProvider",
    "snpe": "SnpeExecutionProvider",
    "xnnpack": "XnnpackExecutionProvider",
}


@dataclass
class ServerConfig:
    bind_address: str = "0.0.0.0:8457"
    base_url: str = ""


@dataclass
class AuthConfig:
    # Empty disables the login screen; montagger is a LAN tool.
    password: str = ""
    session_days: int = 30


@dataclass
class ModelsConfig:
    # Empty resolves to <repo>/models. May point at monbooru's
    # paths.model_path to share downloaded models with it.
    path: str = ""
    default: str = "wd-swinv2"
    # Models every upload tags with, in order; empty falls back to
    # [default]. The UI can override per upload.
    default_models: list[str] = field(default_factory=list)
    # Explicit, never auto: the provider montagger must configure, matched
    # against the installed onnxruntime build at startup. A value the build
    # does not offer is a hard error, not a fallback.
    execution_provider: str = "cpu"
    device_id: int = 0
    # 0 leaves the onnxruntime default threading in place.
    intra_op_threads: int = 0
    max_upload_mb: int = 100
    # Minutes a loaded model sits unused before its session is dropped
    # (RAM/VRAM back to the system). 0 keeps models loaded forever.
    idle_unload_min: int = 0
    # Run all ONNX sessions in a child process. ORT never returns the CUDA
    # context to the OS within a process, so this is the only way
    # 立即释放内存 can actually shrink RSS. Turn off only to shave the
    # ~1s worker startup per respawn; requires a restart when changed.
    isolated: bool = True
    # Categories the tagger never emits (e.g. ["meta"]).
    disabled_categories: list[str] = field(default_factory=list)


@dataclass
class QueueConfig:
    # 0 = unbounded: relay pushes of an unknown count never bounce.
    max_pending: int = 32
    history_days: int = 7


@dataclass
class HFConfig:
    # For gated repos (animetimm-eva02). HF_TOKEN / HUGGING_FACE_HUB_TOKEN
    # win over the file value.
    token: str = ""
    endpoint: str = DEFAULT_HF_ENDPOINT


@dataclass
class MonbooruConfig:
    api_url: str = ""
    web_url: str = ""
    token: str = ""
    # Two independent switches. push_tags: finished results are written back
    # (enrich for images already in monbooru). push_images: direct uploads
    # become new monbooru posts; tags ride along only when push_tags is on.
    # Off by default: montagger never creates posts unless asked.
    push_tags: bool = True
    push_images: bool = False
    gallery: str = ""


@dataclass
class LogConfig:
    debug: bool = False


def normalize_thresholds(raw: dict) -> dict:
    """Per-model threshold overrides, nested form:
    {model: {global, categories:{cat:thr}, top_k:{cat:n}, disabled:[cat]}}.

    The historical flat form {model: {global, <category>: thr}} upgrades
    silently: any non-reserved key is a category threshold.
    """
    out: dict = {}
    for model, over in (raw or {}).items():
        if not isinstance(over, dict):
            continue
        entry: dict = {"global": over.get("global"), "categories": {}, "top_k": {}, "disabled": []}
        cats = over.get("categories")
        if isinstance(cats, dict):
            entry["categories"] = dict(cats)
        for key, value in over.items():
            if key in ("global", "categories", "top_k", "disabled"):
                continue
            entry["categories"][str(key)] = value  # flat-form upgrade
        top_k = over.get("top_k")
        if isinstance(top_k, dict):
            entry["top_k"] = dict(top_k)
        disabled = over.get("disabled")
        if isinstance(disabled, list):
            entry["disabled"] = [str(c) for c in disabled]
        out[str(model)] = entry
    return out


@dataclass
class Config:
    setup_done: bool = False
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    hf: HFConfig = field(default_factory=HFConfig)
    monbooru: MonbooruConfig = field(default_factory=MonbooruConfig)
    log: LogConfig = field(default_factory=LogConfig)
    # Per-model threshold overrides: {"wd-swinv2": {"global": 0.35,
    # "character": 0.5}}. "global" renames the catalog default; a category
    # name overrides just that category.
    thresholds: dict = field(default_factory=dict)

    def model_dir(self, repo_root: Path) -> Path:
        if self.models.path.strip():
            return Path(self.models.path).expanduser()
        return repo_root / "models"

    def hf_token(self) -> str:
        return (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or self.hf.token
        )

    def hf_endpoint(self) -> str:
        env = os.environ.get("HF_ENDPOINT")
        return (env or self.hf.endpoint or DEFAULT_HF_ENDPOINT).rstrip("/")

    def threshold_overrides(self, model: str) -> dict:
        """Normalized per-model overrides: {global, categories, top_k,
        disabled} with every key present."""
        raw = self.thresholds.get(model)
        if not isinstance(raw, dict):
            return {"global": None, "categories": {}, "top_k": {}, "disabled": []}
        return normalize_thresholds({model: raw})[model]

    def validate(self) -> None:
        if not self.models.execution_provider:
            raise ValueError("models.execution_provider must not be empty (explicit choice, no auto)")
        if not (0 < self.models.max_upload_mb <= 4096):
            raise ValueError("models.max_upload_mb must be within 1..4096")
        if not (0 <= self.queue.max_pending <= 1024):
            raise ValueError("queue.max_pending must be 0 (unbounded) or within 1..1024")
        if not (0 <= self.models.idle_unload_min <= 24 * 60):
            raise ValueError("models.idle_unload_min must be within 0..1440")
        if not isinstance(self.monbooru.push_tags, bool) or not isinstance(
            self.monbooru.push_images, bool
        ):
            raise ValueError("monbooru.push_tags / push_images must be booleans")
        seen = set()
        for name in self.models.default_models:
            name = str(name).strip()
            if not name or name in seen:
                raise ValueError(f"models.default_models holds a blank or duplicate entry: {name!r}")
            seen.add(name)
        for model, over in self.thresholds.items():
            if not isinstance(over, dict):
                raise ValueError(f"thresholds.{model} must be a table")
            g = over.get("global")
            if g is not None and not (isinstance(g, (int, float)) and 0.0 < float(g) < 1.0):
                raise ValueError(f"thresholds.{model}.global must be within (0,1)")
            cats = over.get("categories") or {}
            if not isinstance(cats, dict):
                raise ValueError(f"thresholds.{model}.categories must be a table")
            for cat, value in cats.items():
                if not isinstance(value, (int, float)) or not (0.0 < float(value) < 1.0):
                    raise ValueError(f"thresholds.{model}.categories.{cat} must be within (0,1)")
            top_k = over.get("top_k") or {}
            if not isinstance(top_k, dict):
                raise ValueError(f"thresholds.{model}.top_k must be a table")
            for cat, value in top_k.items():
                if not isinstance(value, int) or not (0 < value <= 100):
                    raise ValueError(f"thresholds.{model}.top_k.{cat} must be an int within 1..100")
            disabled = over.get("disabled") or []
            if not isinstance(disabled, list) or not all(isinstance(c, str) and c for c in disabled):
                raise ValueError(f"thresholds.{model}.disabled must be a list of category names")


_SECTIONS = {
    "server": ServerConfig,
    "auth": AuthConfig,
    "models": ModelsConfig,
    "queue": QueueConfig,
    "hf": HFConfig,
    "monbooru": MonbooruConfig,
    "log": LogConfig,
}


def _apply(config: Config, data: dict) -> None:
    for key, value in data.items():
        if key in _SECTIONS and isinstance(value, dict):
            section = getattr(config, key)
            for f in fields(section):
                if f.name in value and value[f.name] is not None:
                    setattr(section, f.name, value[f.name])
        elif key == "thresholds" and isinstance(value, dict):
            config.thresholds = normalize_thresholds(value)
        elif key == "setup_done":
            config.setup_done = bool(value)
        # Unknown keys are ignored: an older config never breaks a newer build.


def generate_secret() -> str:
    return secrets.token_hex(32)


class Provider:
    """Holds the current config; swaps are atomic and persisted."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._config = Config()
        self.load()

    def load(self) -> Config:
        data = {}
        if self.path.exists():
            try:
                data = tomllib.loads(self.path.read_text(encoding="utf-8"))
            except (tomllib.TOMLDecodeError, OSError) as err:
                log.warning("config %s unreadable (%s); using defaults", self.path, err)
        config = Config()
        _apply(config, data)
        config.validate()
        with self._lock:
            self._config = config
        return config

    def current(self) -> Config:
        with self._lock:
            return self._config

    def update(self, mutate) -> Config:
        """mutate(config) edits a deep copy; the result is validated, written,
        then swapped in. A raising mutator leaves everything untouched."""
        with self._lock:
            snapshot = self._config
        import copy

        candidate = copy.deepcopy(snapshot)
        mutate(candidate)
        candidate.validate()
        self._write(candidate)
        with self._lock:
            self._config = candidate
        return candidate

    def _write(self, config: Config) -> None:
        doc: dict = {"setup_done": config.setup_done}
        for name, cls in _SECTIONS.items():
            section = getattr(config, name)
            doc[name] = {f.name: getattr(section, f.name) for f in fields(cls)}
        doc["thresholds"] = config.thresholds
        text = tomli_w.dumps(doc)
        tmp = self.path.with_suffix(".toml.part")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)
