"""The in-memory job queue and its single inference worker.

Image bytes live only inside Job objects here; the SQLite mirror in
store.py keeps metadata. One worker thread owns all inference - GPU and
CPU alike gain nothing from concurrency here - and wait=true requests
share the same inference lock so nothing ever overlaps a session.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from . import logx, store
from .engine.backend import Engine, EngineError, TagResult

log = logx.get("queue")


class QueueFull(Exception):
    pass


class UnknownJob(Exception):
    pass


@dataclass
class Job:
    id: str
    filename: str
    model: str
    source: str
    data: bytes
    sha256: str
    md5: str
    created_at: float = field(default_factory=time.time)
    status: str = store.QUEUED
    result: TagResult | None = None
    error: str = ""
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def public(self, *, with_data: bool = False) -> dict:
        body: dict = {
            "id": self.id,
            "filename": self.filename,
            "model": self.model,
            "source": self.source,
            "status": self.status,
            "bytes": len(self.data),
            "created_at": self.created_at,
        }
        if self.status == store.DONE and self.result is not None:
            r = self.result
            body.update(
                {
                    "width": r.width,
                    "height": r.height,
                    "elapsed_ms": r.elapsed_ms,
                    "provider": r.provider,
                    "rating": r.rating,
                    "tags": [t.__dict__ for t in r.tags],
                }
            )
        if self.error:
            body["error"] = self.error
        return body


class Runner:
    """Owns pending jobs, the worker task, and the SSE broadcast set."""

    # Bytes of the most recent done jobs kept around for previews; older
    # rows keep only their metadata.
    KEEP_RECENT_BYTES = 8
    # Finished Job objects (result tags included) kept in RAM; older ones
    # drop to their SQLite history row. Big photo batches must not pile up
    # Python objects forever.
    MAX_JOB_OBJECTS = 500

    def __init__(self, engine: Engine, store_: store.Store, max_pending: int):
        self.engine = engine
        self.store = store_
        # 0 means unbounded: a monbooru relay push of unknown size never
        # bounces for capacity.
        self.max_pending = max_pending
        # Optional post-done hook (monbooru auto-push); set by the app layer.
        self.on_done = None
        # Optional drain hook: awaited when the queue runs completely dry
        # (no pending, no intake, no inline inference) - the app layer wires
        # the unload-on-drain switch here.
        self.on_drain = None
        self._queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=max_pending)
        # Submitted jobs wait here for their turn, in arrival order; the
        # worker picks from this list by model affinity, not FIFO head.
        self._pending: list[Job] = []
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []  # FIFO over _jobs for byte eviction
        self._subscribers: set[asyncio.Queue] = set()
        self._infer_lock = asyncio.Lock()
        self._worker: asyncio.Task | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._worker = asyncio.get_running_loop().create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass

    async def recover(self) -> None:
        """Boot-time sweep: unfinished rows from a previous process lost
        their bytes with it."""
        self.store.mark_unfinished_lost()

    # -- submission ----------------------------------------------------------

    async def submit(self, job: Job) -> None:
        if self.max_pending > 0:
            pending = sum(1 for j in self._jobs.values() if j.status == store.QUEUED)
            if pending >= self.max_pending:
                raise QueueFull(f"queue holds {self.max_pending} pending jobs; try again shortly")
        self._jobs[job.id] = job
        self._order.append(job.id)
        await self._queue.put(job)
        self._broadcast(job)

    def cancel(self, job_id: str) -> bool:
        """Drop a still-queued job; running inference is not interruptible
        (one shared GPU session) and finished jobs are simply history."""
        return self.drop_if_queued(job_id)

    def queued_ids(self) -> list[str]:
        return [jid for jid in self._order if (j := self._jobs.get(jid)) and j.status == store.QUEUED]

    def get(self, job_id: str) -> Job:
        if job_id not in self._jobs:
            raise UnknownJob(job_id)
        return self._jobs[job_id]

    def drop_if_queued(self, job_id: str) -> bool:
        """Client went away before the job started: forget it entirely."""
        job = self._jobs.get(job_id)
        if job is None or job.status != store.QUEUED:
            return False
        job.status = store.CANCELED
        job.done.set()
        self._forget(job)
        self.store.set_status(job_id, store.CANCELED)
        self._broadcast(job)
        return True

    # -- worker ----------------------------------------------------------------

    async def _worker_loop(self) -> None:
        while True:
            job = await self._next_job()
            try:
                await self._process(job)
            except asyncio.CancelledError:
                raise
            except Exception as err:  # never let the worker die
                log.exception("job %s crashed", job.id)
                job.status, job.error = store.ERROR, str(err)
                job.done.set()
                self.store.finish_error(job.id, str(err))
                self._broadcast(job)
            finally:
                self._evict_bytes()
            await self._maybe_drain()

    async def _maybe_drain(self) -> None:
        """Queue dry: hand the drain hook the inference lock so an inline
        (wait=true) job cannot race the unload, and re-check under it -
        work may have arrived while we waited."""
        if self.on_drain is None or self._pending or not self._queue.empty():
            return
        if self._infer_lock.locked():
            return
        async with self._infer_lock:
            if self._pending or not self._queue.empty():
                return
            await self.on_drain()

    async def _next_job(self) -> Job:
        """Fold the intake queue into the pending list, then hand out work
        by model affinity: queued jobs for an already-resident model run
        before anything that would load (and, under the residency cap,
        evict) another one. A multi-model batch therefore completes one
        model's slice of the queue before swapping models, instead of
        reloading on every image."""
        while True:
            while True:
                try:
                    job = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if job.status == store.QUEUED:  # canceled while queued
                    self._pending.append(job)
            self._pending = [j for j in self._pending if j.status == store.QUEUED]
            if self._pending:
                loaded = {m["name"] for m in self.engine.loaded_models()}
                for job in self._pending:
                    if job.model in loaded:
                        self._pending.remove(job)
                        return job
                return self._pending.pop(0)
            job = await self._queue.get()
            if job.status == store.QUEUED:
                self._pending.append(job)

    async def _process(self, job: Job) -> None:
        job.status = store.RUNNING
        self.store.set_running(job.id, self.engine.provider)
        self._broadcast(job)
        async with self._infer_lock:
            result = await asyncio.to_thread(self.engine.tag_bytes, job.data, job.model)
        job.status, job.result = store.DONE, result
        job.done.set()
        self.store.finish_done(
            job.id,
            provider=result.provider,
            width=result.width,
            height=result.height,
            elapsed_ms=result.elapsed_ms,
            tags=[t.__dict__ for t in result.tags],
            rating=result.rating or "",
        )
        self._broadcast(job)
        if self.on_done is not None:
            try:
                await self.on_done(job)
            except Exception as err:
                log.warning("post-done hook failed for %s: %s", job.id, err)

    async def run_inline(self, job: Job) -> Job:
        """wait=true path: same lock, same persistence, no queue hop."""
        self._jobs[job.id] = job
        self._order.append(job.id)
        try:
            await self._process(job)
        except EngineError as err:
            job.status, job.error = store.ERROR, str(err)
            self.store.finish_error(job.id, str(err))
        except Exception as err:
            job.status, job.error = store.ERROR, str(err)
            self.store.finish_error(job.id, str(err))
        finally:
            self._evict_bytes()
        return job

    # -- fan-out ----------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        sub: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.add(sub)
        return sub

    def unsubscribe(self, sub: asyncio.Queue) -> None:
        self._subscribers.discard(sub)

    def _broadcast(self, job: Job) -> None:
        import json

        payload = json.dumps({"event": "job", "job": job.public()}, separators=(",", ":"))
        for sub in list(self._subscribers):
            try:
                sub.put_nowait(payload)
            except asyncio.QueueFull:
                self._subscribers.discard(sub)

    # -- memory -----------------------------------------------------------------

    def _evict_bytes(self) -> None:
        """Keep bytes only for live jobs and the newest few done ones, then
        cap the finished-Job object count (results stay in SQLite)."""
        done_with_bytes = [jid for jid in self._order if (j := self._jobs.get(jid)) and j.status == store.DONE]
        for jid in done_with_bytes[: -self.KEEP_RECENT_BYTES]:
            if job := self._jobs.get(jid):
                job.data = b""
        finished = [jid for jid in self._order if (j := self._jobs.get(jid)) and j.status in store._TERMINAL]
        for jid in finished[: -self.MAX_JOB_OBJECTS]:
            self._forget(jid)

    def trim_memory(self) -> dict:
        """Manual release: forget finished Job objects entirely and drop
        preview bytes from finished jobs still held."""
        dropped = 0
        for jid in list(self._order):
            job = self._jobs.get(jid)
            if job is None:
                continue
            if job.status in store._TERMINAL:
                self._forget(job)
                dropped += 1
            elif job.data:
                job.data = b""
        return {"jobs_forgotten": dropped, "jobs_kept": len(self._jobs)}

    def _forget(self, job: "Job | str") -> None:
        jid = job.id if isinstance(job, Job) else job
        self._jobs.pop(jid, None)
        if jid in self._order:
            self._order.remove(jid)
