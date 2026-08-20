"""Configuration for montagger.

Three channels, all first-class pydantic-settings sources:

    CLI (built-in CliSettingsSource) > init kwargs > env (MONTAGGER_*,
    MONBOORU_URL as an ecosystem alias) > montagger.toml > defaults

The TOML file is the single source of truth that the WebUI writes back to
via tomlkit (comments and formatting are preserved). Runtime-tunable values
(ep, thresholds, window, thread counts, skip_tagged) are mirrored into a
thread-safe RuntimeState; the pipeline and backends read from that, so the
WebUI can hot-apply them.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# EP names used across settings, CLI and WebUI, mapped to onnxruntime
# provider names. The provider may not be built into the installed
# onnxruntime package; get_available_providers() decides at runtime.
EP_ALIASES: dict[str, str] = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}

# WD14-style tag prefix -> monbooru category. An empty category means the
# monbooru default (general). Verified against /api/v1/categories at startup
# when the backend loads; these are the fallback names.
DEFAULT_CATEGORY_MAP: dict[str, str] = {
    "rating": "rating",
    "general": "",
    "character": "character",
    "copyright": "copyright",
    "artist": "artist",
}

# Relay buttons offered at pairing. media="image" keeps videos, animations
# and comic archives out - they have no single frame to tag, and monbooru
# drops those ids out of a batch selection anyway.
BUTTONS: list[dict[str, str]] = [
    {"slot": "detail-actions", "label": "tag with montagger", "mode": "relay", "path": "/relay/tag", "media": "image"},
    {"slot": "batch-bar", "label": "tag with montagger", "mode": "relay", "path": "/relay/tag", "media": "image"},
]

# montagger.toml is organised in sections (like monbooru.toml). The settings
# model stays flat; this map routes each field to its (section, key) pair so
# [server].url and [monbooru].url can both exist without colliding.
FIELD_MAP: dict[str, tuple[str, str]] = {
    "addr": ("server", "addr"),
    "url": ("server", "url"),
    "monbooru": ("monbooru", "url"),
    "via": ("monbooru", "via"),
    "state": ("paths", "state"),
    "config": ("paths", "config"),
    "model_dir": ("paths", "model_dir"),
    "backend": ("tagging", "backend"),
    "ep": ("tagging", "ep"),
    "threshold": ("tagging", "threshold"),
    "character_threshold": ("tagging", "character_threshold"),
    "activation": ("tagging", "activation"),
    "general_topk": ("tagging", "general_topk"),
    "window": ("pipeline", "window"),
    "prefetch_threads": ("pipeline", "prefetch_threads"),
    "workers": ("pipeline", "workers"),
    "skip_tagged": ("pipeline", "skip_tagged"),
    "resume": ("pipeline", "resume"),
    "webui_token": ("webui", "webui_token"),
    "log_level": ("log", "level"),
}
TOML_TO_FIELD = {
    (section, key): field for field, (section, key) in FIELD_MAP.items()
}


def resolve_config_path() -> Path:
    """Locate montagger.toml: --config/-c on the command line wins, then
    MONTAGGER_CONFIG, then ./montagger.toml. Scanned before Settings is
    built because the TOML source itself needs the path."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg in ("--config", "-c"):
            if i + 1 < len(argv):
                return Path(argv[i + 1])
        elif arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    env = os.environ.get("MONTAGGER_CONFIG")
    if env:
        return Path(env)
    return Path("montagger.toml")


class MontaggerTomlSource(PydanticBaseSettingsSource):
    """Toml source pinned to the resolved config path. Section tables are
    flattened onto the flat settings model (e.g. [pipeline].window ->
    field window)."""

    def __init__(self, settings_cls: type[BaseSettings], toml_file: Path) -> None:
        super().__init__(settings_cls)
        self.toml_file = Path(toml_file)

    def __call__(self) -> dict[str, Any]:
        if not self.toml_file.exists():
            return {}
        import tomllib

        doc = tomllib.loads(self.toml_file.read_text(encoding="utf-8"))
        flat: dict[str, Any] = {}
        for key, value in doc.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    field = TOML_TO_FIELD.get((key, sub_key))
                    if field:
                        flat[field] = sub_value
            else:
                if key in self.settings_cls.model_fields:
                    flat[key] = value  # top-level keys (flat legacy files)
        return flat

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # The whole document is collected in __call__; per-field hooks are
        # not used.
        return None, field_name, False


class Settings(BaseSettings):
    """All configuration. Fields are read from CLI > init > env > TOML >
    defaults; WebUI changes are written back to the TOML file."""

    model_config = SettingsConfigDict(
        env_prefix="MONTAGGER_",
        cli_parse_args=True,
        cli_implicit_flags=True,
        cli_show_env_vars=True,
        populate_by_name=True,  # source dicts may use field names, not only aliases
        extra="ignore",
    )

    # --- listening / pairing ------------------------------------------
    addr: str = "127.0.0.1:8301"  # what we serve on
    monbooru: str = Field(  # where monbooru answers
        default="http://127.0.0.1:8080",
        validation_alias=AliasChoices("MONTAGGER_MONBOORU", "MONBOORU_URL"),
    )
    url: str = ""  # address monbooru should call us back at; default http://<addr>
    state: Path = Path(".")  # credentials, database and montagger.toml live here
    config: Path = Path("montagger.toml")  # config file used (--config/-c)

    # --- tagging -------------------------------------------------------
    model_dir: Path = Path(".")  # ONNX model folder (may point into monbooru's model_path)
    backend: str = "heuristic"  # heuristic | onnx
    ep: str = "cpu"  # cpu | cuda | directml | openvino | coreml
    threshold: float = 0.35  # general tag score floor
    character_threshold: float = 0.5  # category-specific floor for character
    activation: str = "sigmoid_in_model"  # sigmoid_in_model | logits
    general_topk: int = Field(default=40, ge=1, le=200)  # cap on general tags per image

    # --- pipeline ------------------------------------------------------
    window: int = Field(default=16, ge=1, le=1024)  # inflight images (fetch+prep+ready+infer)
    prefetch_threads: int = Field(default=2, ge=1, le=64)
    workers: int = Field(default=2, ge=1, le=64)  # inference workers; forced to 1 on DirectML
    skip_tagged: bool = False  # skip images that already have auto_tagged_at
    resume: bool = True  # re-enqueue pending/failed/processing tasks at startup

    # --- webui ---------------------------------------------------------
    webui_token: str = ""  # optional token for direct WebUI access

    # --- misc ----------------------------------------------------------
    log_level: str = "info"
    via: str = "montagger"  # source string written into monbooru tags

    _source_order: list[tuple[str, PydanticBaseSettingsSource]] = []

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        cli = CliSettingsSource(settings_cls)
        toml = MontaggerTomlSource(settings_cls, resolve_config_path())
        # Record the chain so WebUI can show where a value came from.
        cls._source_order = [
            ("cli", cli),
            ("init", init_settings),
            ("env", env_settings),
            ("toml", toml),
        ]
        return (cli, init_settings, env_settings, toml)

    @property
    def self_url(self) -> str:
        return self.url or "http://" + self.addr

    @property
    def state_dir(self) -> Path:
        return self.state

    def source_of(self, field_name: str) -> str:
        """Which channel supplied the current value of a field."""
        for name, source in type(self)._source_order:
            try:
                values = source(self)
            except TypeError:
                values = source()
            if field_name in values or field_name.upper() in values:
                return name
        return "default"

    def write_back(self, values: dict[str, Any]) -> None:
        """Persist WebUI setting changes into montagger.toml, preserving
        comments and formatting (tomlkit). Creates the file if absent."""
        import tomlkit

        path = resolve_config_path()
        if path.exists():
            doc = tomlkit.parse(path.read_text(encoding="utf-8"))
        else:
            doc = tomlkit.document()
        for key, value in values.items():
            mapped = FIELD_MAP.get(key)
            if mapped:
                section, toml_key = mapped
                table = doc.get(section)
                if table is None:
                    table = tomlkit.table()
                    doc[section] = table
                table[toml_key] = value
            else:
                doc[key] = value
        path.write_text(tomlkit.dumps(doc), encoding="utf-8")

    def effective_workers(self) -> int:
        if self.ep == "directml":
            return 1  # onnxruntime DirectML EP is not thread-safe
        return self.workers


@dataclass
class RuntimeState:
    """Hot-applyable settings shared between pipeline, backends and WebUI.
    Reads are lock-free; updates are atomic per field."""

    ep: str = "cpu"
    threshold: float = 0.35
    character_threshold: float = 0.5
    general_topk: int = 40
    backend: str = "heuristic"
    model_dir: str = "."
    activation: str = "sigmoid_in_model"
    window: int = 16
    prefetch_threads: int = 2
    workers: int = 2
    skip_tagged: bool = False

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def from_settings(cls, settings: Settings) -> "RuntimeState":
        return cls(
            ep=settings.ep,
            threshold=settings.threshold,
            character_threshold=settings.character_threshold,
            general_topk=settings.general_topk,
            backend=settings.backend,
            model_dir=str(settings.model_dir),
            activation=settings.activation,
            window=settings.window,
            prefetch_threads=settings.prefetch_threads,
            workers=settings.workers,
            skip_tagged=settings.skip_tagged,
        )

    def update(self, **values: Any) -> None:
        with self._lock:
            for key, value in values.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def effective_workers(self) -> int:
        with self._lock:
            return 1 if self.ep == "directml" else self.workers

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ep": self.ep,
                "threshold": self.threshold,
                "character_threshold": self.character_threshold,
                "general_topk": self.general_topk,
                "backend": self.backend,
                "model_dir": self.model_dir,
                "activation": self.activation,
                "window": self.window,
                "prefetch_threads": self.prefetch_threads,
                "workers": self.workers,
                "effective_workers": self.effective_workers(),
                "skip_tagged": self.skip_tagged,
            }