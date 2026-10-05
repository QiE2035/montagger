"""The model catalog: what can be installed and from where.

Port of tagger/catalog.go, one behavioral addition: montagger ships a real
downloader, so entries carry everything it needs (direct HF resolve URLs,
the gated flag). A models.json in the model path overrides or extends the
embedded entries, same as monbooru.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

_DATA = Path(__file__).parent / "data"


class CatalogFile(BaseModel):
    url: str
    filename: str


class CatalogEntry(BaseModel):
    name: str
    description: str = ""
    files: list[CatalogFile] = Field(default_factory=list)
    gated: bool = False
    default_threshold: float = 0.35
    default_thresholds: dict[str, float] = Field(default_factory=dict)
    default_top_k: dict[str, int] = Field(default_factory=dict)


class CatalogDoc(BaseModel):
    version: Literal[1]
    models: list[CatalogEntry] = Field(default_factory=list)


def _parse(doc: dict) -> list[CatalogEntry]:
    return CatalogDoc.model_validate(doc).models


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
