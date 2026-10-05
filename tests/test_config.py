"""Config: pydantic models, validation, and TOML round-trips."""

from __future__ import annotations

from montagger.config import Config, Provider, normalize_thresholds


def test_toml_round_trip(provider: Provider):
    def mutate(config: Config):
        config.setup_done = True
        config.server.bind_address = "127.0.0.1:9000"
        config.models.max_upload_mb = 2048
        config.models.default_models = ["wd-swinv2", "camie"]
        config.monbooru.api_url = "http://lan:8080"
        config.thresholds = normalize_thresholds({"stub": {"global": 0.4, "character": 0.6}})

    saved = provider.update(mutate)
    reloaded = provider.load()
    assert reloaded == saved
    assert reloaded.setup_done is True
    assert reloaded.server.bind_address == "127.0.0.1:9000"
    assert reloaded.models.max_upload_mb == 2048
    assert reloaded.models.default_models == ["wd-swinv2", "camie"]
    assert reloaded.monbooru.api_url == "http://lan:8080"
    # the flat form {"character": 0.6} upgraded silently to categories
    overrides = reloaded.threshold_overrides("stub")
    assert overrides.global_ == 0.4
    assert overrides.categories == {"character": 0.6}


def test_threshold_without_global_survives_write(provider: Provider):
    """global=None must not reach tomli_w (TOML has no null); it is omitted
    on write and reads back as None (keep the catalog default)."""

    def mutate(config: Config):
        config.thresholds = {"stub": {"categories": {"character": 0.6}}}

    saved = provider.update(mutate)
    reloaded = provider.load()
    assert saved == reloaded
    overrides = reloaded.threshold_overrides("stub")
    assert overrides.global_ is None
    assert overrides.categories == {"character": 0.6}


def test_malformed_sections_dropped(provider: Provider):
    """A wrong-typed section is dropped rather than fatal - an old or
    corrupted config never breaks a newer build."""
    provider.path.write_text(
        'server = 5\nthresholds = "oops"\n[models]\npath = "/tmp/m"\n',
        encoding="utf-8",
    )
    config = provider.load()
    assert config.server.bind_address == "0.0.0.0:8457"  # defaults
    assert config.thresholds == {}
    assert config.models.path == "/tmp/m"


def test_out_of_range_rejected(provider: Provider):
    import pytest
    from pydantic import ValidationError

    def mutate(config: Config):
        config.models.max_upload_mb = 0

    with pytest.raises(ValidationError):
        provider.update(mutate)
    # the raising mutator leaves everything untouched
    assert provider.current().models.max_upload_mb == 100
