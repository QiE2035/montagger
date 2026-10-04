"""Provider resolution (explicit, fail-fast) and model folder file selection."""

import numpy as np
import pytest

import montagger.engine.session as session_mod
from montagger.engine.session import EngineError, resolve_provider, resolve_tagger_files, has_tagger_files


def test_cpu_always_available():
    assert resolve_provider("cpu") == "CPUExecutionProvider"


def test_unknown_provider_rejected():
    with pytest.raises(EngineError, match="unknown execution_provider"):
        resolve_provider("warp-drive")


def test_unavailable_provider_is_hard_error(monkeypatch):
    monkeypatch.setattr(session_mod.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(EngineError, match="not in the installed"):
        resolve_provider("cuda")


def test_provider_accepted_when_build_offers_it(monkeypatch):
    monkeypatch.setattr(
        session_mod.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert resolve_provider("cuda") == "CUDAExecutionProvider"


def test_resolve_tagger_files(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    (model / "model.onnx").write_bytes(b"x")
    (model / "tags.csv").write_bytes(b"y")
    assert resolve_tagger_files(model) == ("model.onnx", "tags.csv")


def test_resolve_tagger_files_txt_variant(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    (model / "model.onnx").write_bytes(b"x")
    (model / "tags.txt").write_bytes(b"y")
    (model / "tagger.json").write_bytes(b"{}")  # sidecar never becomes a label file
    assert resolve_tagger_files(model) == ("model.onnx", "tags.txt")


def test_resolve_tagger_files_single_onnx_nonstandard(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    (model / "camie-tagger-v2.onnx").write_bytes(b"x")
    (model / "camie-tagger-v2-metadata.json").write_bytes(b"y")
    assert resolve_tagger_files(model) == ("camie-tagger-v2.onnx", "camie-tagger-v2-metadata.json")


def test_has_tagger_files(tmp_path):
    model = tmp_path / "m"
    assert not has_tagger_files(model)
    model.mkdir()
    assert not has_tagger_files(model)
    (model / "model.onnx").write_bytes(b"x")
    assert has_tagger_files(model)
