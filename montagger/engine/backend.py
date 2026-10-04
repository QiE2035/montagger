"""The tagger facade: discovery, thresholds, one-call tagging.

Ties catalog + model folders + onnxruntime sessions together. The public
surface is tag_bytes(): bytes in, a TagResult out - no file involved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import logx
from ..config import Provider
from .catalog import find_entry, load_catalog
from .categories import RATING_TAGS
from .scoring import ScoredTag, aggregate, sort_tags
from .session import (
    EngineError,
    ModelRuntime,
    SessionManager,
    has_tagger_files,
    model_insights,
    resolve_tagger_files,
)

log = logx.get("engine")


@dataclass
class ModelStatus:
    name: str
    description: str = ""
    gated: bool = False
    in_catalog: bool = False
    discovered: bool = False  # a folder with model+label files exists
    available: bool = False
    reason: str = ""
    size_bytes: int = 0
    default_threshold: float = 0.35
    default_thresholds: dict = field(default_factory=dict)
    default_top_k: dict = field(default_factory=dict)
    emitted_categories: list[str] = field(default_factory=list)


@dataclass
class TagResult:
    model: str
    provider: str
    width: int
    height: int
    elapsed_ms: int
    tags: list[ScoredTag]
    rating: str | None


class Engine:
    """Holds the session manager; the EP comes from config at construction
    and after an explicit reload (settings save), never per request."""

    def __init__(self, cfg: Provider, repo_root: Path):
        self._cfg = cfg
        self._repo_root = repo_root
        self._emitted_cache: dict[str, list[str]] = {}
        self._sessions = self._build_sessions()

    def _build_sessions(self) -> SessionManager:
        from .session import WorkerSessionManager

        m = self._cfg.current().models
        cls = WorkerSessionManager if m.isolated else SessionManager
        return cls(m.execution_provider, m.device_id, m.intra_op_threads)

    def child_rss_mb(self) -> float:
        """Worker-process RSS when inference is isolated; else 0."""
        child = getattr(self._sessions, "child_rss_mb", None)
        return child() if child is not None else 0.0

    def reload_sessions(self) -> None:
        """Called when EP/device/threads change; drops every loaded model."""
        self._sessions = self._build_sessions()

    @property
    def provider(self) -> str:
        return self._sessions.provider

    @property
    def provider_short(self) -> str:
        return self._sessions.provider_short

    @property
    def model_root(self) -> Path:
        return self._cfg.current().model_dir(self._repo_root)

    # -- discovery ---------------------------------------------------------

    def statuses(self) -> list[ModelStatus]:
        cfg = self._cfg.current()
        root = self.model_root
        entries = {e.name: e for e in load_catalog(root)}

        discovered: set[str] = set()
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if child.is_dir() and has_tagger_files(child):
                    discovered.add(child.name)

        out: list[ModelStatus] = []
        seen: set[str] = set()
        for name in [*entries, *discovered]:
            if name in seen:
                continue
            seen.add(name)
            entry = entries.get(name)
            status = ModelStatus(
                name=name,
                description=entry.description if entry else "",
                gated=bool(entry.gated) if entry else False,
                in_catalog=entry is not None,
                discovered=name in discovered,
                default_threshold=entry.default_threshold if entry else 0.35,
                default_thresholds=dict(entry.default_thresholds) if entry else {},
                default_top_k=dict(entry.default_top_k) if entry else {},
            )
            if name in discovered:
                model_file, tags_file = resolve_tagger_files(root / name)
                missing = [
                    f
                    for f, p in (
                        (model_file, root / name / model_file),
                        (tags_file, root / name / tags_file),
                    )
                    if not p.exists()
                ]
                if missing:
                    status.reason = "missing " + ", ".join(missing)
                else:
                    status.available = True
                    try:
                        status.size_bytes = sum(
                            p.stat().st_size
                            for p in (root / name).iterdir()
                            if p.is_file()
                        )
                    except OSError:
                        pass
            elif entry is not None:
                status.reason = "not installed"
            out.append(status)

        # The configured default sorts first so the UI can land on it.
        default = cfg.models.default
        out.sort(key=lambda s: (s.name != default, s.name))
        return out

    def is_available(self, name: str) -> str:
        """'' when the model can run, else the reason it cannot."""
        for status in self.statuses():
            if status.name == name:
                return status.reason
        return "unknown model"

    # -- memory ------------------------------------------------------------

    def loaded_models(self) -> list[dict]:
        """What is resident right now, for /health and the settings UI."""
        out = []
        for name in self._sessions.loaded():
            idle = self._sessions.idle_seconds(name)
            out.append({"name": name, "idle_s": int(idle) if idle is not None else None})
        return out

    def unload(self, name: str) -> bool:
        """Drop one model's session. True if it was resident. When the last
        one goes, the isolated worker shuts down too - that is the only
        state that keeps the CUDA context alive."""
        if name not in self._sessions.loaded():
            return False
        self._sessions.invalidate(name)
        if not self._sessions.loaded() and hasattr(self._sessions, "child_rss_mb"):
            self._sessions.invalidate(None)
        log.info("model %s unloaded", name)
        return True

    def unload_all(self) -> int:
        names = self._sessions.loaded()
        self._sessions.invalidate(None)
        if names:
            log.info("unloaded %d model session(s)", len(names))
        return len(names)

    # -- category metadata ---------------------------------------------------

    def emitted_categories(self, name: str) -> list[str]:
        """Categories the model can emit; GPU-free, cached per name."""
        cached = self._emitted_cache.get(name)
        if cached is not None:
            return cached
        cats: list[str] = []
        if (self.model_root / name).is_dir():
            try:
                cats = model_insights(self.model_root / name, name)
            except EngineError:
                cats = []
        self._emitted_cache[name] = cats
        return cats

    # -- tagging -----------------------------------------------------------

    def resolve_model_name(self, model: str | None = None) -> str:
        name = (model or self._cfg.current().models.default).strip()
        if not name:
            raise EngineError("no model configured: set models.default or pass ?model=")
        return name

    def resolve_model_list(self, model_param: str | None = None) -> list[str]:
        """Models one upload tags with: an explicit ?model=a,b wins, else
        the configured default_models, else [default]. Order preserved,
        duplicates dropped; empty means nothing is configured."""
        cfg = self._cfg.current()
        if model_param and model_param.strip():
            names: list[str] = []
            for part in model_param.split(","):
                name = part.strip()
                if name and name not in names:
                    names.append(name)
            if names:
                return names
        names = [m.strip() for m in cfg.models.default_models if m.strip()]
        if names:
            return names
        default = cfg.models.default.strip()
        return [default] if default else []

    def thresholds(self, name: str) -> tuple[float, dict, dict, list[str]]:
        """Effective (global, per-category thresholds, per-category top-k,
        disabled categories) for one model: catalog defaults with the
        config's nested per-model overrides on top."""
        cfg = self._cfg.current()
        overrides = cfg.threshold_overrides(name)
        entry = find_entry(load_catalog(self.model_root), name)
        global_threshold = float(
            overrides["global"]
            if overrides["global"] is not None
            else (entry.default_threshold if entry else 0.35)
        )
        category_thresholds = dict(entry.default_thresholds) if entry else {}
        category_thresholds.update({k: float(v) for k, v in overrides["categories"].items()})
        top_k = dict(entry.default_top_k) if entry else {}
        top_k.update({k: int(v) for k, v in overrides["top_k"].items()})
        disabled = list(overrides["disabled"])
        return global_threshold, category_thresholds, top_k, disabled

    def tag_bytes(self, data: bytes, model: str | None = None) -> TagResult:
        cfg = self._cfg.current()
        name = self.resolve_model_name(model)
        reason = self.is_available(name)
        if reason:
            raise EngineError(f"model {name}: {reason}")

        runtime: ModelRuntime = self._sessions.get(self.model_root / name, name)
        started = time.perf_counter()
        scores, width, height = runtime.infer(data)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        global_threshold, category_thresholds, top_k, disabled = self.thresholds(name)
        # Global never-emit categories plus this model's own disabled list.
        disabled = tuple({*cfg.models.disabled_categories, *disabled})
        tags = aggregate(
            scores,
            runtime.candidates,
            global_threshold,
            category_thresholds,
            top_k,
            disabled,
        )
        # A dispatch rule can name anything; only the four levels rate.
        tags = [t for t in tags if t.category != "rating" or t.name in RATING_TAGS]
        rating = next((t.name for t in tags if t.category == "rating"), None)
        return TagResult(
            model=name,
            provider=self.provider,
            width=width,
            height=height,
            elapsed_ms=elapsed_ms,
            tags=sort_tags(tags),
            rating=rating,
        )

    # -- installation ------------------------------------------------------

    def install(self, name: str, progress=None) -> Path:
        from .downloader import DownloadError
        from .downloader import install as download

        entry = find_entry(load_catalog(self.model_root), name)
        if entry is None:
            raise EngineError(f"model {name}: not in the catalog")
        if not (entry.files or []):
            raise EngineError(f"model {name}: catalog entry lists no files")
        try:
            model_dir = download(
                self._cfg.current(), self.model_root, entry, progress=progress
            )
        except DownloadError:
            raise
        self._sessions.invalidate(name)
        self._emitted_cache.pop(name, None)
        log.info("model %s installed at %s", name, model_dir)
        return model_dir

    def remove(self, name: str) -> None:
        """Drop one model folder. Refuses names outside the model root and
        any folder that is not a model folder."""
        import shutil

        root = self.model_root.resolve()
        target = (root / name).resolve()
        if target.parent != root or target == root:
            raise EngineError(f"refusing to remove {name}")
        if not target.is_dir() or not has_tagger_files(target):
            raise EngineError(f"model {name}: no model folder to remove")
        shutil.rmtree(target)
        self._sessions.invalidate(name)
        self._emitted_cache.pop(name, None)
        log.info("model %s removed", name)
