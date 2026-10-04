"""Label file parsers: wd14_csv, joytag_txt, camie_json."""

import json

import pytest

from montagger.engine.labels import load_labels

WD14 = "tag_id,name,category\n0,1girl,0\n1,Solo,0\n2,hatsune miku,4\n,sensitive,9\n,____,0\n"
JOYTAG = "1girl\n\nsolo\ntagged/bad\n"
CAMIE = {
    "dataset_info": {
        "tag_mapping": {
            "idx_to_tag": {"0": "1girl", "2": "solo", "5": "hatsune miku"},
            "tag_to_category": {"1girl": "general", "solo": "general", "hatsune miku": "character"},
        }
    }
}


def write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_wd14_csv(tmp_path):
    labels = load_labels(write(tmp_path, "tags.csv", WD14), "wd14_csv")
    assert [l.name for l in labels] == ["1girl", "solo", "hatsune_miku", "sensitive", "_unsupported_4"]
    assert labels[2].category_id == 4
    assert labels[4].placeholder  # decoration-only label


def test_joytag_txt(tmp_path):
    labels = load_labels(write(tmp_path, "tags.txt", JOYTAG), "joytag_txt")
    # A slash is not whitespace or a reserved char: it survives, exactly as
    # monbooru's normalize keeps it.
    assert [l.name for l in labels] == ["1girl", "solo", "tagged/bad"]
    assert all(l.category_id == 0 for l in labels)


def test_camie_json(tmp_path):
    labels = load_labels(write(tmp_path, "meta.json", json.dumps(CAMIE)), "camie_json")
    assert len(labels) == 6  # indices dense up to max
    assert labels[0].name == "1girl"
    assert labels[1].placeholder  # gap in idx_to_tag
    assert labels[5].category_name == "character"


def test_camie_empty_mapping_raises(tmp_path):
    p = write(tmp_path, "meta.json", json.dumps({"dataset_info": {"tag_mapping": {}}}))
    with pytest.raises(ValueError):
        load_labels(p, "camie_json")
