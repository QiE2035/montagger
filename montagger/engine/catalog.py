"""The model catalog: what can be installed and from where.

Port of tagger/catalog.go, one behavioral addition: montagger ships a real
downloader, so entries carry everything it needs (direct HF resolve URLs,
the gated flag). A models.json in the model path overrides or extends the
embedded entries, same as monbooru.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_DATA = Path(__file__).parent / "data"


@dataclass
class CatalogFile:
    url: str
    filename: str


@dataclass
class CatalogEntry:
    name: str
    description: str
    files: list[CatalogFile]
    gated: bool = False
    default_threshold: float = 0.35
    default_thresholds: dict = field(default_factory=dict)
    default_top_k: dict = field(default_factory=dict)


def _parse(doc: dict) -> list[CatalogEntry]:
    if doc.get("version") != 1:
        raise ValueError(f"catalog: unsupported version {doc.get('version')}")
    out = []
    for model in doc.get("models", []):
        out.append(
            CatalogEntry(
                name=model["name"],
                description=model.get("description", ""),
                files=[CatalogFile(url=f["url"], filename=f["filename"]) for f in model.get("files", [])],
                gated=bool(model.get("gated", False)),
                default_threshold=float(model.get("default_threshold", 0.35)),
                default_thresholds=dict(model.get("default_thresholds", {})),
                default_top_k=dict(model.get("default_top_k", {})),
            )
        )
    return out


def load_catalog(model_path: Path | None = None) -> list[CatalogEntry]:
    entries = _parse(json.loads((_DATA / "catalog.json").read_text(encoding="utf-8")))
    if model_path is not None:
        override = model_path / "models.json"
        if override.exists():
            try:
                extra = _parse(json.loads(override.read_text(encoding="utf-8")))
            except (ValueError, json.JSONDecodeError) as err:
                raise ValueError(f"{override}: {err}") from err
            by_name = {e.name: i for i, e in enumerate(entries)}
            for e in extra:
                if e.name in by_name:
                    entries[by_name[e.name]] = e
                else:
                    entries.append(e)
    return entries


def find_entry(catalog: list[CatalogEntry], name: str) -> CatalogEntry | None:
    for e in catalog:
        if e.name == name:
            return e
    return None
