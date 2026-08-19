"""Pipeline: window bound, dedup, failure isolation, pause/resume.

Uses a real Store (tmp file) with fake client/backend so the threading is
exercised honestly.
"""

from __future__ import annotations

import threading
import time

import pytest
from PIL import Image

from montagger.pipeline import Pipeline
from montagger.settings import RuntimeState
from montagger.store import Store


class FakeClient:
    def __init__(self) -> None:
        self.fetched: list[int] = []
        self.tagged: list[tuple[int, list[str]]] = []
        self.fail_fetch: set[int] = set()
        self.fail_tags: set[int] = set()
        self.fetch_delay = 0.0
        self.lock = threading.Lock()

    def fetch_image(self, image_id: int) -> bytes:
        with self.lock:
            self.fetched.append(image_id)
        if image_id in self.fail_fetch:
            raise RuntimeError("download failed")
        if self.fetch_delay:
            time.sleep(self.fetch_delay)
        img = Image.new("RGB", (8, 8), (200, 100, 50))
        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def add_tags(self, image_id: int, tags: list[str], via: str) -> None:
        if image_id in self.fail_tags:
            raise RuntimeError("write failed")
        with self.lock:
            self.tagged.append((image_id, list(tags)))

    def image_status(self, image_id: int) -> dict:
        return {"auto_tagged_at": None}


class FakeBackend:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0

    def preprocess(self, image):
        return image

    def tag(self, prepared, runtime):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return ["tag_x"]


def wait_until(condition, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def make_pipeline(tmp_path: pytest.TempPathFactory) -> Pipeline:
    store = Store(tmp_path / "p.db")
    runtime = RuntimeState(window=3, prefetch_threads=2, workers=2)
    client = FakeClient()
    backend = FakeBackend()
    events: list[tuple[str, dict]] = []

    pipe = Pipeline(runtime, store, client, backend, "montagger", lambda kind, data: events.append((kind, data)))
    pipe.start()
    yield pipe
    pipe.stop(drain_timeout=5.0)
    store.close()


def test_basic_flow(make_pipeline: Pipeline) -> None:
    pipe = make_pipeline
    new, known = pipe.submit([1, 2, 3])
    assert new == 3
    assert known == 0
    assert wait_until(lambda: pipe.stats()["done"] == 3)
    assert len(pipe.client.tagged) == 3  # type: ignore[attr-defined]
    stats = pipe.stats()
    assert stats["pending"] == 0
    assert stats["failed"] == 0


def test_dedup(make_pipeline: Pipeline) -> None:
    pipe = make_pipeline
    pipe.submit([1, 2])
    new, known = pipe.submit([2, 3])
    assert new == 1
    assert known == 1
    assert wait_until(lambda: pipe.stats()["done"] == 3)


def test_window_bound(make_pipeline: Pipeline) -> None:
    pipe = make_pipeline
    pipe.runtime.window = 2  # type: ignore[attr-defined]
    pipe.backend.delay = 0.05  # type: ignore[attr-defined]
    pipe.submit(list(range(1, 40)))
    max_processing = 0
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        max_processing = max(max_processing, pipe.stats()["processing"])
        time.sleep(0.005)
        if pipe.stats()["pending"] == 0 and pipe.stats()["processing"] == 0 \
                and pipe.stats()["done"] == 39:
            break
    assert max_processing <= 2, f"inflight exceeded the window: {max_processing}"
    assert pipe.stats()["done"] == 39


def test_failure_isolation(make_pipeline: Pipeline) -> None:
    pipe = make_pipeline
    pipe.client.fail_fetch = {2}  # type: ignore[attr-defined]
    pipe.client.fail_tags = {4}  # type: ignore[attr-defined]
    pipe.submit([1, 2, 3, 4, 5])
    assert wait_until(lambda: pipe.stats()["done"] + pipe.stats()["failed"] == 5)
    stats = pipe.stats()
    assert stats["done"] == 3
    assert stats["failed"] == 2
    rows, _ = pipe.store.results(1, 50, "failed")
    errors = {r["image_id"]: r["error"] for r in rows}
    assert "download failed" in errors[2]
    assert "write failed" in errors[4]


def test_pause_resume(make_pipeline: Pipeline) -> None:
    pipe = make_pipeline
    pipe.submit(list(range(1, 20)))
    pipe.pause()
    time.sleep(0.3)
    frozen_fetches = len(pipe.client.fetched)  # type: ignore[attr-defined]
    time.sleep(0.3)
    assert len(pipe.client.fetched) == frozen_fetches  # type: ignore[attr-defined]
    pipe.resume()
    assert wait_until(lambda: pipe.stats()["done"] == 19)


def test_retry_failed(make_pipeline: Pipeline) -> None:
    pipe = make_pipeline
    pipe.client.fail_tags = {1}  # type: ignore[attr-defined]
    pipe.submit([1, 2])
    assert wait_until(lambda: pipe.stats()["failed"] == 1)
    pipe.client.fail_tags = set()  # type: ignore[attr-defined]
    assert pipe.retry_failed() == 1
    assert wait_until(lambda: pipe.stats()["done"] == 2 and pipe.stats()["failed"] == 0)


def test_skip_tagged(make_pipeline: Pipeline) -> None:
    pipe = make_pipeline
    pipe.runtime.skip_tagged = True  # type: ignore[attr-defined]

    class SkippedClient(FakeClient):
        def image_status(self, image_id: int) -> dict:
            return {"auto_tagged_at": "2026-01-01" if image_id == 7 else None}

    pipe.client = SkippedClient()  # type: ignore[attr-defined]
    pipe.submit([7, 8])
    assert wait_until(lambda: pipe.stats()["done"] == 2)
    assert 7 not in pipe.client.fetched  # type: ignore[attr-defined]
    assert 8 in pipe.client.fetched  # type: ignore[attr-defined]