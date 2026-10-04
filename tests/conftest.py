"""Shared fixtures: config provider and model profile factory."""

from __future__ import annotations

import pytest

from montagger.config import Config, Provider
from montagger.engine.profile import Profile


def make_profile(**over) -> Profile:
    base = dict(
        input_size=4,
        layout="nchw",
        channels="rgb",
        normalize="none",
        pad="white_square",
        activation="sigmoid_in_model",
        label_format="wd14_csv",
        category_scheme="wd14_numeric",
        output_index=0,
    )
    base.update(over)
    return Profile(**base)


@pytest.fixture
def cfg_path(tmp_path):
    return tmp_path / "montagger.toml"


@pytest.fixture
def provider(cfg_path):
    return Provider(cfg_path)


@pytest.fixture
def config() -> Config:
    return Config()
