"""Category resolution and dispatch overlays."""

from pathlib import Path

from conftest import make_profile

from montagger.engine.categories import resolve_category
from montagger.engine.dispatch import load_dispatch
from montagger.engine.labels import Label


def lbl(name, cat=0, cat_name=""):
    return Label(name=name, category_id=cat, category_name=cat_name)


def test_wd14_numeric():
    profile = make_profile()
    assert resolve_category(profile, lbl("1girl", 0), None).category == "general"
    assert resolve_category(profile, lbl("hatsune_miku", 4), None).category == "character"
    assert resolve_category(profile, lbl("artist:foo", 1), None).category == "artist"
    assert resolve_category(profile, lbl("unknown", 7), None).category == "general"
    # category 9 is deliberately not mapped; only canonical rating names route.
    assert resolve_category(profile, lbl("whatever", 9), None).category == "general"


def test_rating_by_name():
    profile = make_profile()
    for name in ("general", "sensitive", "questionable", "explicit"):
        assert resolve_category(profile, lbl(name, 0), None).category == "rating"


def test_single_general():
    profile = make_profile(category_scheme="single_general")
    assert resolve_category(profile, lbl("1girl"), None).category == "general"


def test_name_string():
    profile = make_profile(category_scheme="name_string")
    assert resolve_category(profile, lbl("x", cat_name="character"), None).category == "character"
    assert resolve_category(profile, lbl("x", cat_name="weird"), None).category == "general"
    assert resolve_category(profile, lbl("x"), None).category == "general"


def test_dispatch_override_beats_scheme():
    profile = make_profile()
    dispatch = load_dispatch(Path("/nonexistent"), "wd-swinv2")  # embedded rules load
    # wd-swinv2's embedded table re-files monochrome under medium.
    res = resolve_category(profile, lbl("monochrome", 0), dispatch)
    assert res.category == "medium" and res.override

    # Canonical rating names still route to rating regardless.
    assert resolve_category(profile, lbl("sensitive", 0), dispatch).category == "rating"


def test_dispatch_drop(tmp_path):
    (tmp_path / "dispatch.json").write_text(
        '{"version": 1, "rules": [{"source": "bad_tag", "category": ""},'
        ' {"source": "renamed", "category": "meta", "name": "Renamed Tag"},'
        ' {"source": "unknown_cat", "category": "nope"}]}'
    )
    dispatch = load_dispatch(tmp_path, "nosuchmodel")
    profile = make_profile()
    assert resolve_category(profile, lbl("bad_tag"), dispatch).skip
    res = resolve_category(profile, lbl("renamed"), dispatch)
    assert res.category == "meta" and res.override
    # The renamed rule also renames the tag; ModelRuntime applies that.
    assert dispatch.lookup("renamed").name == "renamed_tag"
    # Unknown target category: rule fails to compile, label falls through.
    assert resolve_category(profile, lbl("unknown_cat"), dispatch).category == "general"


def test_overlay_beats_embedded(tmp_path):
    (tmp_path / "dispatch.json").write_text('{"version": 1, "rules": [{"source": "monochrome", "category": "meta"}]}')
    dispatch = load_dispatch(tmp_path, "wd-swinv2")
    assert resolve_category(make_profile(), lbl("monochrome"), dispatch).category == "meta"
