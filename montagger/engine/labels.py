"""Model label files: wd14_csv, joytag_txt and camie_json parsers.

Ported from tagger/labels.go. Labels are indexed by output channel: an
unusable label becomes a placeholder that inference skips rather than a
hole that would shift every later channel.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

from .names import sanitize_label


@dataclass
class Label:
    name: str
    category_id: int = 0  # wd14 numeric category, 0 elsewhere
    category_name: str = ""  # camie's own category string, "" elsewhere
    placeholder: bool = False


def load_labels(path: Path, label_format: str) -> list[Label]:
    raw = path.read_bytes()
    if label_format == "wd14_csv":
        return _load_csv(io.StringIO(raw.decode("utf-8-sig")))
    if label_format == "joytag_txt":
        return _load_text(raw.decode("utf-8"))
    if label_format == "camie_json":
        return _load_camie_json(raw)
    raise ValueError(f"loadLabels: unsupported label_format {label_format!r}")


def _load_csv(f) -> list[Label]:
    reader = csv.reader(f)
    try:
        next(reader)  # header
    except StopIteration:
        raise ValueError("wd14 csv: empty file") from None
    labels: list[Label] = []
    for rec in reader:
        if len(rec) < 3:
            continue
        try:
            cat_id = int(rec[2].strip())
        except ValueError:
            cat_id = 0
        name, ok = sanitize_label(rec[1], len(labels))
        labels.append(Label(name=name, category_id=cat_id, placeholder=not ok))
    return labels


def _load_text(text: str) -> list[Label]:
    labels: list[Label] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        name, ok = sanitize_label(line, len(labels))
        labels.append(Label(name=name, category_id=0, placeholder=not ok))
    return labels


def _load_camie_json(raw: bytes) -> list[Label]:
    doc = json.loads(raw.decode("utf-8"))
    mapping = doc.get("dataset_info", {}).get("tag_mapping", {})
    idx_to_tag: dict = mapping.get("idx_to_tag", {})
    if not idx_to_tag:
        raise ValueError("camie metadata: dataset_info.tag_mapping.idx_to_tag is empty")
    max_idx = -1
    for k in idx_to_tag:
        i = int(k)
        if i < 0:
            raise ValueError(f"camie metadata: bad idx {k!r}")
        max_idx = max(max_idx, i)
    labels: list[Label | None] = [None] * (max_idx + 1)
    tag_to_cat: dict = mapping.get("tag_to_category", {})
    for k, raw_name in idx_to_tag.items():
        i = int(k)
        name, ok = sanitize_label(raw_name, i)
        labels[i] = Label(name=name, category_id=0, category_name=tag_to_cat.get(raw_name, ""), placeholder=not ok)
    return [
        lbl if lbl is not None else Label(name=f"_unsupported_{i}", category_id=0, placeholder=True)
        for i, lbl in enumerate(labels)
    ]
