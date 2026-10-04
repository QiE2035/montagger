"""Tag name normalization, ported from monbooru internal/tags so a label the
Go tagger keeps is a label montagger keeps (and spells the same way).
"""

from __future__ import annotations

import unicodedata

# Port of monbooru's tagDecorationClass.
_DECORATION = set("_()!@#$.~+:-")
_CONTROL_CATEGORIES = ("Cc", "Cf", "Co")


def _fold(name: str) -> str:
    """Port of buildTagName(name, fold_reserved=true): lowercase, controls
    dropped, whitespace and reserved characters folded to underscores, runs
    collapsed, no leading underscore.

    Go checks unicode.IsSpace; Python's isspace matches on every realistic
    label character set (the stray divergence is C0 controls inside a tag
    name, which no shipped model label contains).
    """
    name = name.lower()
    out: list[str] = []
    pending = False
    for r in name:
        if unicodedata.category(r) in _CONTROL_CATEGORIES:
            continue
        if r.isspace() or r in ('"', "*"):
            pending = True
            continue
        if pending and out:
            out.append("_")
        pending = False
        out.append(r)
    return "".join(out)


def normalize_name(name: str) -> str:
    """Port of tags.NormalizeName, for imported model labels."""
    return _fold(name).strip("_")


def has_tag_content(name: str) -> bool:
    """Port of tags.HasTagContent: at least one character that is not pure
    decoration, so "_", "()" and friends never become tags."""
    return any(r not in _DECORATION for r in name)


def sanitize_label(raw: str, idx: int) -> tuple[str, bool]:
    """Port of tagger.sanitizeLabel: an unusable label becomes a placeholder
    that inference must skip, keeping output channel indices intact."""
    name = normalize_name(raw)
    chars = list(name)
    if len(chars) > 200:
        name = "".join(chars[:200])
    if not name or not has_tag_content(name):
        return f"_unsupported_{idx}", False
    return name, True
