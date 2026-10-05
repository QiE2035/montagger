"""Engine-level recovery: accelerator OOM releases every session and
retries once; anything else propagates untouched."""

from __future__ import annotations

import numpy as np
import pytest

from montagger.config import Provider
from montagger.engine.backend import Engine, TagResult
from montagger.engine.session import CandidateLabel, EngineError

_OOM = (
    "[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running "
    "FusedMatMul node. Failed to allocate memory for requested buffer of size 67240192"
)
_CANDIDATES = [CandidateLabel(name="1girl", category="general", placeholder=False)]


class FakeRuntime:
    candidates = _CANDIDATES

    def __init__(self, outcome):
        self.outcome = outcome  # scores array or an exception to raise

    def infer(self, data: bytes):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome, 2, 2


class StubSessions:
    """Hands out queued runtimes in order and records invalidations."""

    provider = "CPUExecutionProvider"
    provider_short = "cpu"

    def __init__(self, runtimes):
        self._runtimes = list(runtimes)
        self.invalidations: list[str | None] = []

    def get(self, model_dir, name):
        return self._runtimes.pop(0)

    def invalidate(self, name=None):
        self.invalidations.append(name)

    def loaded(self):
        return []

    def idle_seconds(self, name):
        return None


def make_engine(tmp_path, runtimes) -> tuple[Engine, StubSessions]:
    # a discovered model folder is enough: availability checks file presence
    stub = tmp_path / "models" / "stub"
    stub.mkdir(parents=True)
    (stub / "model.onnx").write_bytes(b"")
    (stub / "tags.csv").write_text("name,category\n1girl,0\n", encoding="utf-8")
    engine = Engine(Provider(tmp_path / "montagger.toml"), tmp_path)
    sessions = StubSessions(runtimes)
    engine._sessions = sessions
    return engine, sessions


def test_oom_releases_all_and_retries_once(tmp_path):
    scores = np.array([0.9], dtype=np.float32)
    engine, sessions = make_engine(tmp_path, [FakeRuntime(EngineError(_OOM)), FakeRuntime(scores)])
    result = engine.tag_bytes(b"fake", "stub")
    assert isinstance(result, TagResult) and result.tags[0].name == "1girl"
    assert sessions.invalidations == [None]  # everything released, once


def test_persistent_oom_fails_with_actionable_error(tmp_path):
    engine, _ = make_engine(tmp_path, [FakeRuntime(EngineError(_OOM)), FakeRuntime(EngineError(_OOM))])
    with pytest.raises(EngineError, match="still out of memory"):
        engine.tag_bytes(b"fake", "stub")


def test_non_oom_errors_propagate_without_invalidating(tmp_path):
    boom = EngineError("model stub: missing model.onnx")
    engine, sessions = make_engine(tmp_path, [FakeRuntime(boom), FakeRuntime(np.array([0.9]))])
    with pytest.raises(EngineError, match="missing model.onnx"):
        engine.tag_bytes(b"fake", "stub")
    assert sessions.invalidations == []
