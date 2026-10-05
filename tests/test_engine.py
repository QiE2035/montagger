"""Residency cap (LRU eviction) and model-affinity queue scheduling."""

from __future__ import annotations

import asyncio

from montagger import store
from montagger.engine.backend import TagResult
from montagger.engine.session import WorkerSessionManager
from montagger.queue import Job, Runner

# ---------------------------------------------------------------- residency cap


class _FakeChild:
    """Stands in for the worker process: records ops, answers loads."""

    def __init__(self):
        self.ops: list[str] = []

    def __call__(self, msg: dict, payload=None) -> dict:
        self.ops.append(str(msg.get("op")))
        if msg.get("op") == "load":
            return {"input_size": 4}
        return {}


class _AliveProc:
    """Just enough proc shape for invalidate() to send unload ops."""

    def poll(self):
        return None


def make_model_dir(tmp_path, name: str):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.onnx").write_bytes(b"")
    (d / "tags.csv").write_text(
        "tag_id,name,category,count\n0,1girl,0,100\n1,1boy,0,90\n", encoding="utf-8"
    )
    return d


def make_manager(tmp_path, monkeypatch, max_loaded: int) -> tuple[WorkerSessionManager, _FakeChild]:
    mgr = WorkerSessionManager("cpu", 0, 0)
    fake = _FakeChild()
    monkeypatch.setattr(mgr, "_request", fake)
    mgr._proc = _AliveProc()  # unload ops flow through the fake child
    mgr.max_loaded = max_loaded
    return mgr, fake


def test_residency_cap_one_evicts_before_loading_next(tmp_path, monkeypatch):
    mgr, fake = make_manager(tmp_path, monkeypatch, max_loaded=1)
    mgr.get(make_model_dir(tmp_path, "a"), "a")
    assert mgr.loaded() == ["a"]
    mgr.get(make_model_dir(tmp_path, "b"), "b")
    assert mgr.loaded() == ["b"]  # "a" was evicted, not co-resident
    assert fake.ops == ["load", "unload", "load"]


def test_residency_cap_evicts_least_recently_used(tmp_path, monkeypatch):
    mgr, fake = make_manager(tmp_path, monkeypatch, max_loaded=2)
    mgr.get(make_model_dir(tmp_path, "a"), "a")
    mgr.get(make_model_dir(tmp_path, "b"), "b")
    mgr.get(make_model_dir(tmp_path, "c"), "c")
    assert mgr.loaded() == ["b", "c"]  # "a" is the LRU victim
    assert fake.ops.count("unload") == 1


def test_no_cap_keeps_every_model_resident(tmp_path, monkeypatch):
    mgr, fake = make_manager(tmp_path, monkeypatch, max_loaded=0)
    for name in ("a", "b", "c"):
        mgr.get(make_model_dir(tmp_path, name), name)
    assert sorted(mgr.loaded()) == ["a", "b", "c"]
    assert "unload" not in fake.ops


# ------------------------------------------------------- model-affinity scheduling


class _FakeEngine:
    """Residency follows the last model that ran (a cap of 1)."""

    provider = "CPUExecutionProvider"

    def __init__(self):
        self.order: list[str] = []
        self._loaded: list[str] = []

    def loaded_models(self):
        return [{"name": n, "idle_s": 0} for n in self._loaded]

    def tag_bytes(self, data: bytes, model: str | None = None) -> TagResult:
        name = model or ""
        self.order.append(name)
        self._loaded = [name]
        return TagResult(model=name, provider="cpu", width=1, height=1, elapsed_ms=1, tags=[], rating=None)


class _FakeStore:
    def set_running(self, job_id, provider):
        pass

    def finish_done(self, **kwargs):
        pass

    def finish_error(self, job_id, error):
        pass

    def set_status(self, job_id, status):
        pass


def make_job(idx: int, model: str):
    return Job(
        id=f"job-{idx}",
        filename="x.jpg",
        model=model,
        source="test",
        data=b"x",
        sha256=f"sha-{idx}",
        md5=f"md5-{idx}",
    )


def test_queue_runs_one_models_slice_before_swapping(tmp_path):
    async def scenario() -> list[str]:
        engine = _FakeEngine()
        runner = Runner(engine, _FakeStore(), 0)
        runner.start()
        jobs = [make_job(i, m) for i, m in enumerate(("a", "b", "a", "b", "a"))]
        for job in jobs:
            await runner.submit(job)
        for _ in range(500):
            if all(job.status in store._TERMINAL for job in jobs):
                break
            await asyncio.sleep(0.01)
        await runner.stop()
        return engine.order

    # submitted image-major (a,b,a,b,a); affinity completes a's slice first
    assert asyncio.run(scenario()) == ["a", "a", "a", "b", "b"]


def test_drain_hook_fires_only_when_queue_runs_dry():
    async def scenario() -> list[int]:
        engine = _FakeEngine()
        runner = Runner(engine, _FakeStore(), 0)
        drained_at: list[int] = []

        async def on_drain():
            drained_at.append(len(engine.order))

        runner.on_drain = on_drain
        runner.start()
        jobs = [make_job(i, m) for i, m in enumerate(("a", "a", "b"))]
        for job in jobs:
            await runner.submit(job)
        for _ in range(500):
            if all(job.status in store._TERMINAL for job in jobs):
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)  # let the post-drain hook settle
        await runner.stop()
        return drained_at

    # no drain while b's job is still queued; exactly one drain at the end
    assert asyncio.run(scenario()) == [3]


def test_queue_memory_caps():
    """keep_recent caps preview bytes, max_job_objects caps Job objects."""
    runner = Runner(_FakeEngine(), _FakeStore(), 0, keep_recent=1, max_job_objects=2)
    jobs = [make_job(i, "a") for i in range(4)]
    for job in jobs:
        job.status = store.DONE
        runner._jobs[job.id] = job
        runner._order.append(job.id)
    runner._evict_bytes()
    # only the newest done job keeps its preview bytes
    assert [job.data == b"" for job in jobs] == [True, True, True, False]
    # and only max_job_objects finished Job objects survive
    assert set(runner._jobs) == {"job-2", "job-3"}
