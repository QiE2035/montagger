"""Settings sources: TOML < env < CLI, MONBOORU_URL alias, write-back."""

from __future__ import annotations

import sys

import pytest

from montagger.settings import Settings


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(sys.modules["os"].environ):
        if key.startswith("MONTAGGER_") or key == "MONBOORU_URL" or key == "MONTAGGER_CONFIG":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(sys, "argv", ["montagger"])


def test_defaults(clean_env: None) -> None:
    settings = Settings()
    assert settings.window == 16
    assert settings.backend == "heuristic"
    assert settings.monbooru == "http://127.0.0.1:8080"


def test_toml_is_lowest_source(clean_env: None, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "montagger.toml"
    config.write_text("window = 8\nthreshold = 0.5\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = Settings()
    assert settings.window == 8
    assert settings.threshold == 0.5

    # env beats toml
    monkeypatch.setenv("MONTAGGER_WINDOW", "12")
    settings = Settings()
    assert settings.window == 12

    # CLI beats env
    monkeypatch.setattr(sys, "argv", ["montagger", "--window", "20"])
    settings = Settings()
    assert settings.window == 20


def test_monbooru_url_alias(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONBOORU_URL", "http://example.test:9999")
    assert Settings().monbooru == "http://example.test:9999"

    # the prefixed form wins
    monkeypatch.setenv("MONTAGGER_MONBOORU", "http://other.test")
    assert Settings().monbooru == "http://other.test"


def test_cli_implicit_flag(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["montagger", "--skip_tagged"])
    assert Settings().skip_tagged is True


def test_write_back_roundtrip(clean_env: None, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "montagger.toml"
    config.write_text(
        "# keep me\nwindow = 16  # inflight\nthreshold = 0.35\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = Settings()
    settings.write_back({"window": 24, "ep": "directml"})

    raw = config.read_text(encoding="utf-8")
    assert "# keep me" in raw          # comments survive
    assert "# inflight" in raw         # inline comments survive
    assert "window = 24" in raw
    assert "ep = \"directml\"" in raw

    reloaded = Settings()
    assert reloaded.window == 24
    assert reloaded.ep == "directml"


def test_source_of(clean_env: None, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "montagger.toml"
    config.write_text("window = 8\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = Settings()
    assert settings.source_of("window") == "toml"
    assert settings.source_of("backend") == "default"
    monkeypatch.setenv("MONTAGGER_WINDOW", "12")
    assert Settings().source_of("window") == "env"