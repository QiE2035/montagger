"""Score aggregation: thresholds, top-k, rating and disabled categories."""

import numpy as np

from montagger.engine.scoring import ScoredTag, aggregate, resolve_top_k, sort_tags
from montagger.engine.session import CandidateLabel


def cands(*pairs):
    return [CandidateLabel(name=n, category=c, placeholder=False) for n, c in pairs]


def test_threshold_filters():
    candidates = cands(("1girl", "general"), ("solo", "general"))
    scores = np.array([0.9, 0.2], dtype=np.float32)
    tags = aggregate(scores, candidates, global_threshold=0.35)
    assert [t.name for t in tags] == ["1girl"]


def test_per_category_threshold():
    candidates = cands(("miku", "character"), ("1girl", "general"))
    scores = np.array([0.45, 0.45], dtype=np.float32)
    tags = aggregate(scores, candidates, 0.35, {"character": 0.5})
    assert [t.name for t in tags] == ["1girl"]


def test_top_k_with_name_tiebreak():
    candidates = cands(("b_tag", "general"), ("a_tag", "general"), ("c_tag", "general"), ("miku", "character"))
    scores = np.array([0.9, 0.9, 0.8, 0.7], dtype=np.float32)
    tags = aggregate(scores, candidates, 0.1, {}, {"general": 2})
    general = sorted(t.name for t in tags if t.category == "general")
    assert general == ["a_tag", "b_tag"]  # tie broken by name, c_tag cut by k


def test_sub_threshold_score_skipped_early():
    candidates = cands(("tiny", "general"))
    scores = np.array([0.0005], dtype=np.float32)
    assert aggregate(scores, candidates, 0.0001) == []


def test_placeholder_skipped():
    from montagger.engine.session import CandidateLabel as C

    candidates = [C(name="_unsupported_0", category="", placeholder=True)]
    scores = np.array([0.99], dtype=np.float32)
    assert aggregate(scores, candidates, 0.1) == []


def test_sort_tags_category_order():
    tags = sort_tags(
        [
            ScoredTag(name="1girl", category="general", confidence=0.9),
            ScoredTag(name="miku", category="character", confidence=0.9),
            ScoredTag(name="explicit", category="rating", confidence=0.9),
            ScoredTag(name="foo", category="artist", confidence=0.9),
        ]
    )
    assert [t.category for t in tags] == ["artist", "character", "general", "rating"]


def test_resolve_top_k():
    assert resolve_top_k(None, "character") == 8
    assert resolve_top_k(None, "whatever") == 10
    assert resolve_top_k({"general": 0}, "general") == 0  # explicit 0 = uncapped
