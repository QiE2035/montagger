"""Category resolution, ported from tagger/categories.go.

montagger keeps no tag database, so a resolution answers with a category
NAME only; the ordering rules (dispatch > canonical rating names >
category scheme) are identical to the Go original.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .labels import Label
from .profile import Profile

if TYPE_CHECKING:
    from .dispatch import DispatchTable

# Category 9, WD14's rating family, is left out on purpose: the canonical
# rating labels are routed by name, and any other label in 9 falls to
# general.
_WD14_CATEGORY = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}

# Only the four canonical names: anything else routed to rating is dropped.
RATING_TAGS = ("general", "sensitive", "questionable", "explicit")
_RATING_SET = frozenset(RATING_TAGS)

# monbooru's category universe: camie's name_string scheme files its own
# labels into these, anything unfamiliar falls to general (the Go original
# checks its tag DB's category table; these are the same rows).
KNOWN_CATEGORIES = frozenset(
    ("general", "artist", "copyright", "character", "meta", "rating", "medium", "person", "species", "year")
)


@dataclass
class Resolution:
    category: str
    skip: bool = False
    override: bool = False  # true when a dispatch rule produced this


def resolve_category(profile: Profile, label: Label, dispatch: DispatchTable | None) -> Resolution:
    if dispatch is not None:
        rule = dispatch.lookup(label.name)
        if rule is not None:
            if rule.drop:
                return Resolution(category="", skip=True, override=True)
            return Resolution(category=rule.category, override=True)
    if label.name in _RATING_SET:
        return Resolution(category="rating")
    if profile.category_scheme == "wd14_numeric":
        return Resolution(category=_WD14_CATEGORY.get(label.category_id, "general"))
    if profile.category_scheme == "single_general":
        return Resolution(category="general")
    if profile.category_scheme == "name_string":
        name = label.category_name
        if name not in KNOWN_CATEGORIES:
            name = "general"
        return Resolution(category=name)
    return Resolution(category="general")
