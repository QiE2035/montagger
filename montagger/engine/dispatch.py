"""Dispatch overlays: per-label category/name overrides, ported from
tagger/dispatch.go. Embedded defaults load first, then the model folder's
own dispatch.json rewrites them; a rule that fails to compile leaves the
previous rule for its source in place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .names import sanitize_label
from .categories import KNOWN_CATEGORIES

_DISPATCH_SCHEMA_VERSION = 1

_DATA = Path(__file__).parent / "data"


@dataclass(frozen=True)
class DispatchRule:
    drop: bool
    category: str  # "" exactly when drop
    name: str  # "" keeps the source label as the tag name


def _compile(rule: dict) -> DispatchRule | None:
    category = rule.get("category", "")
    if category == "":
        compiled = DispatchRule(drop=True, category="", name="")
    elif category in KNOWN_CATEGORIES:
        compiled = DispatchRule(drop=False, category=category, name="")
    else:
        return None  # unknown target category: keep the prior rule
    name = rule.get("name", "")
    if name:
        fixed, ok = sanitize_label(name, 0)
        if not ok:
            return None  # unsupported target name: keep the prior rule
        compiled = DispatchRule(drop=compiled.drop, category=compiled.category, name=fixed)
    return compiled


def _embedded_rules(name: str) -> list[dict]:
    path = _DATA / "dispatch" / f"{name}.json"
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("version") != _DISPATCH_SCHEMA_VERSION:
        raise ValueError(f"embedded dispatch {name}: unsupported version {doc.get('version')}")
    return doc.get("rules", [])


def _overlay_rules(model_dir: Path) -> list[dict]:
    path = model_dir / "dispatch.json"
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("version") != _DISPATCH_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported dispatch version {doc.get('version')}")
    return doc.get("rules", [])


class DispatchTable:
    def __init__(self, rules: dict[str, DispatchRule]):
        self.rules = rules

    def lookup(self, source: str) -> DispatchRule | None:
        return self.rules.get(source)


def load_dispatch(model_dir: Path, name: str) -> DispatchTable:
    rules: dict[str, DispatchRule] = {}
    for entry in _embedded_rules(name):
        compiled = _compile(entry)
        if compiled is not None:
            rules[entry["source"]] = compiled
    for entry in _overlay_rules(model_dir):
        compiled = _compile(entry)
        if compiled is not None:
            rules[entry["source"]] = compiled
    return DispatchTable(rules)
