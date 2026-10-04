"""Score aggregation, ported from tagger/aggregation.go.

Tuned for the WD14, JoyTag and Camie distributions: attribution categories
sit low so a noisy run can't pile them onto one image, and rating and year
take a single tag each.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PER_CATEGORY_TOP_K = {
    "character": 8,
    "copyright": 4,
    "artist": 4,
    "general": 25,
    "rating": 1,
    "medium": 4,
    "person": 8,
    "species": 8,
    "year": 1,
}
TOP_K_FALLBACK = 10

# montagger tags single images; the multi-frame min-hits rule of the Go
# original degenerates to one hit per frame count of 1.
_MIN_SCORE = 0.001


def resolve_top_k(overrides: dict[str, int] | None, category: str) -> int:
    """0 (uncapped) when the override is an explicit 0."""
    if overrides and category in overrides:
        return overrides[category]
    if category in DEFAULT_PER_CATEGORY_TOP_K:
        return DEFAULT_PER_CATEGORY_TOP_K[category]
    return TOP_K_FALLBACK


@dataclass
class ScoredTag:
    name: str
    category: str
    confidence: float


def aggregate(
    scores,
    candidates,
    global_threshold: float,
    category_thresholds: dict[str, float] | None = None,
    top_k_overrides: dict[str, int] | None = None,
    disabled_categories: tuple[str, ...] = (),
) -> list[ScoredTag]:
    """One frame of scores → surviving tags: per-category threshold, then a
    per-category top-k with a name tiebreak so equal runs emit the same set."""
    cat_thresholds = category_thresholds or {}
    disabled = frozenset(disabled_categories)

    by_category: dict[str, list[ScoredTag]] = {}
    for idx, score in enumerate(scores):
        if score < _MIN_SCORE or idx >= len(candidates):
            continue
        cand = candidates[idx]
        if cand.placeholder or cand.category in disabled:
            continue
        threshold = cat_thresholds.get(cand.category, global_threshold)
        if score < threshold:
            continue
        by_category.setdefault(cand.category, []).append(
            ScoredTag(name=cand.name, category=cand.category, confidence=round(float(score), 4))
        )

    out: list[ScoredTag] = []
    for category, tags in by_category.items():
        tags.sort(key=lambda t: (-t.confidence, t.name))
        k = resolve_top_k(top_k_overrides, category)
        if k > 0:
            tags = tags[:k]
        out.extend(tags)
    return out


# Fixed display order; within a category tags stay sorted by confidence.
CATEGORY_ORDER = (
    "artist",
    "copyright",
    "character",
    "general",
    "meta",
    "medium",
    "person",
    "species",
    "year",
    "rating",
)


def sort_tags(tags: list[ScoredTag]) -> list[ScoredTag]:
    rank = {name: i for i, name in enumerate(CATEGORY_ORDER)}
    return sorted(tags, key=lambda t: (rank.get(t.category, len(CATEGORY_ORDER)), -t.confidence, t.name))
