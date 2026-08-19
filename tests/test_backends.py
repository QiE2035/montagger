"""Backends: heuristic output shape, label parsing, category mapping."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from montagger.backends.heuristic import HeuristicBackend
from montagger.backends.onnx import _apply_categories, _parse_labels
from montagger.settings import DEFAULT_CATEGORY_MAP


def test_heuristic_tags_are_general() -> None:
    backend = HeuristicBackend()
    image = Image.new("RGB", (3200, 1000), (240, 240, 240))  # wide, bright, low saturation
    prepared = backend.preprocess(image)
    tags = backend.tag(prepared, None)
    assert "landscape" in tags
    assert "bright" in tags
    assert "grayscale" in tags
    assert "large" in tags
    assert all(":" not in tag for tag in tags)  # nothing prefixed -> general


def test_parse_labels_csv_with_header(tmp_path: Path) -> None:
    csv_file = tmp_path / "tags.csv"
    csv_file.write_text(
        "tag,count\nrating_safe,10\ngeneral_1girl,20\ncharacter_miku,5\n",
        encoding="utf-8",
    )
    labels = _parse_labels(csv_file, 3)  # output_n == 3, header must be dropped
    assert labels is not None
    assert [label.tag for label in labels] == ["rating_safe", "general_1girl", "character_miku"]


def test_parse_labels_txt(tmp_path: Path) -> None:
    txt = tmp_path / "tags.txt"
    txt.write_text("quality_lineart\ngeneral_1girl\n", encoding="utf-8")
    labels = _parse_labels(txt, 2)
    assert labels is not None
    assert len(labels) == 2


def test_parse_labels_mismatch(tmp_path: Path) -> None:
    txt = tmp_path / "tags.txt"
    txt.write_text("a\nb\nc\n", encoding="utf-8")
    assert _parse_labels(txt, 5) is None  # cannot align -> refuse to guess


def test_apply_categories_maps_wd14_prefixes() -> None:
    labels = [
        type("L", (), {"tag": "rating_safe", "category": "", "monbooru_tag": "rating_safe"})(),
        type("L", (), {"tag": "general_1girl", "category": "", "monbooru_tag": "general_1girl"})(),
        type("L", (), {"tag": "character_miku", "category": "", "monbooru_tag": "character_miku"})(),
        type("L", (), {"tag": "copyright_touhou", "category": "", "monbooru_tag": "copyright_touhou"})(),
    ]
    valid = set(DEFAULT_CATEGORY_MAP.values())
    _apply_categories(labels, valid)
    assert labels[0].monbooru_tag == "rating:safe"
    assert labels[1].monbooru_tag == "1girl"
    assert labels[2].monbooru_tag == "character:miku"
    assert labels[3].monbooru_tag == "copyright:touhou"


def test_apply_categories_unknown_category_stays_general() -> None:
    labels = [
        type("L", (), {"tag": "epic_thing", "category": "", "monbooru_tag": "epic_thing"})(),
    ]
    _apply_categories(labels, {"general", "character"})  # no "epic" category
    assert labels[0].monbooru_tag == "epic_thing"
    assert labels[0].category == ""