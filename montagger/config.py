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
from pathlib import Path
from typing import Annotated

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

# pydantic constraint aliases keep the threshold tables declarative.
ThresholdValue = Annotated[float, Field(gt=0, lt=1)]
TopKValue = Annotated[int, Field(gt=0, le=100)]
CategoryName = Annotated[str, Field(min_length=1)]


class _Section(BaseModel):
    """Unknown keys are ignored: an older config never breaks a newer build.
    Assignment is revalidated so a saving mutator cannot park a bad value."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)


class ServerConfig(_Section):
    bind_address: str = "0.0.0.0:8457"
    base_url: str = ""


class AuthConfig(_Section):
    # Empty disables the login screen; montagger is a LAN tool.
    password: str = ""
    session_days: int = 30


class ModelsConfig(_Section):
    # Empty resolves to <repo>/models. May point at monbooru's
    # paths.model_path to share downloaded models with it.
    path: str = ""
    default: str = "wd-swinv2"
    # Models every upload tags with, in order; empty falls back to
    # [default]. The UI can override per upload.
    default_models: list[str] = Field(default_factory=list)
    # Explicit, never auto: the provider montagger must configure, matched
    # against the installed onnxruntime build at startup. A value the build
    # does not offer is a hard error, not a fallback.
    execution_provider: str = Field("cpu", min_length=1)
    device_id: int = 0
    # 0 leaves the onnxruntime default threading in place.
    intra_op_threads: int = 0
    max_upload_mb: int = Field(100, gt=0, le=4096)
    # Minutes a loaded model sits unused before its session is dropped
    # (RAM/VRAM back to the system). 0 keeps models loaded forever.
    idle_unload_min: int = Field(0, ge=0, le=1440)
    # Models that may stay resident at once; loading one more evicts the
    # least recently used. Keeps multi-model batches from exhausting the
    # accelerator - 1 means one model runs its whole slice of the queue
    # before the next one loads. 0 disables the cap.
    max_loaded: int = Field(1, ge=0, le=64)
    # Unload every model the moment the queue runs dry (no queued, no
    # running work). Independent of idle_unload_min; on a small-VRAM box
    # this hands the accelerator back after every batch.
    unload_on_drain: bool = False
    # Run all ONNX sessions in a child process. ORT never returns the CUDA
    # context to the OS within a process, so this is the only way
    # 立即释放内存 can actually shrink RSS. Turn off only to shave the
    # ~1s worker startup per respawn; requires a restart when changed.
    isolated: bool = True
    # Categories the tagger never emits (e.g. ["meta"]).
    disabled_categories: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_blank_or_duplicate_models(self) -> "ModelsConfig":
        seen = set()
        for name in self.default_models:
            if not name.strip() or name.strip() in seen:
                raise ValueError(f"default_models holds a blank or duplicate entry: {name!r}")
            seen.add(name.strip())
        return self


class QueueConfig(_Section):
    # 0 = unbounded: relay pushes of an unknown count never bounce.
    max_pending: int = Field(32, ge=0, le=1024)
    history_days: int = 7


class HFConfig(_Section):
    # For gated repos (animetimm-eva02). HF_TOKEN / HUGGING_FACE_HUB_TOKEN
    # win over the file value.
    token: str = ""
    endpoint: str = DEFAULT_HF_ENDPOINT


class MonbooruConfig(_Section):
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


class LogConfig(_Section):
    debug: bool = False


class ThresholdOverride(_Section):
    """Per-model threshold overrides, nested form:
    {global, categories:{cat:thr}, top_k:{cat:n}, disabled:[cat]}."""

    model_config = ConfigDict(populate_by_name=True)

    # TOML has no null: unset means "keep the catalog default".
    global_: ThresholdValue | None = Field(default=None, alias="global")
    categories: dict[str, ThresholdValue] = Field(default_factory=dict)
    top_k: dict[str, TopKValue] = Field(default_factory=dict)
    disabled: list[CategoryName] = Field(default_factory=list)


def normalize_thresholds(raw: dict) -> dict:
    """Per-model threshold overrides, nested form:
    {model: {global, categories:{cat:thr}, top_k:{cat:n}, disabled:[cat]}}.

    The historical flat form {model: {global, <category>: thr}} upgrades
    silently: any non-reserved key is a category threshold. Entries already
    normalized to ThresholdOverride pass through untouched.
    """
    out: dict = {}
    for model, over in (raw or {}).items():
        if isinstance(over, ThresholdOverride):
            out[str(model)] = over
            continue
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


_SECTIONS = ("server", "auth", "models", "queue", "hf", "monbooru", "log")


class Config(_Section):
    setup_done: bool = False
    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    hf: HFConfig = Field(default_factory=HFConfig)
    monbooru: MonbooruConfig = Field(default_factory=MonbooruConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    # Per-model threshold overrides: {"wd-swinv2": {"global": 0.35,
    # "character": 0.5}}. "global" renames the catalog default; a category
    # name overrides just that category.
    thresholds: dict[str, ThresholdOverride] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _drop_malformed_sections(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        cleaned = {}
        for key, value in data.items():
            # Values are plain dicts when validating parsed TOML, but model
            # instances when validate_assignment re-runs this on a live
            # config - both are well-formed, anything else is dropped.
            if (key in _SECTIONS or key == "thresholds") and not isinstance(
                value, (dict, BaseModel)
            ):
                continue
            cleaned[key] = value
        return cleaned

    @field_validator("setup_done", mode="before")
    @classmethod
    def _coerce_bool(cls, value: object) -> object:
        return bool(value)

    @field_validator("thresholds", mode="before")
    @classmethod
    def _normalize_thresholds(cls, value: object) -> object:
        return normalize_thresholds(value)  # upgrades the historical flat form

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

    def threshold_overrides(self, model: str) -> ThresholdOverride:
        """Normalized per-model overrides with every key present."""
        return self.thresholds.get(model) or ThresholdOverride()


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
        config = Config.model_validate(data)
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
        candidate = snapshot.model_copy(deep=True)
        mutate(candidate)
        # Assignment validation catches scalar edits already; this final pass
        # also re-validates in-place edits (dict/list mutation escapes it).
        candidate = Config.model_validate(candidate.model_dump(by_alias=True))
        self._write(candidate)
        with self._lock:
            self._config = candidate
        return candidate

    def _write(self, config: Config) -> None:
        # exclude_none: TOML has no null, an unset "global" is simply absent
        # and reads back as None (keep the catalog default).
        doc = config.model_dump(by_alias=True, exclude_none=True)
        text = tomli_w.dumps(doc)
        tmp = self.path.with_suffix(".toml.part")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)
