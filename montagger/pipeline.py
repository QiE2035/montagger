"""The tagging pipeline: a bounded prefetch window in front of inference.

Tasks (image ids) arrive in an unbounded deque; the memory bound lives in
the inflight window. Prefetch threads acquire a window slot, download the
image (bytes stay in memory, decoded via Pillow), preprocess it and hand it
to a ready queue, where the inference workers pick it up. A slot is only
released when inference (including writing the tags back) finishes - so
every completed image immediately makes room for the next download:
fetch, preprocess and inference overlap, and the network never makes the
model sit idle.

Dynamic tuning: pause/resume, workers and prefetch threads can grow or
shrink at runtime, and the window size is read on every slot acquisition.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable

from montagger.store import PENDING, PROCESSING, Store

log = logging.getLogger(__name__)

_STOP = object()


class Pipeline:
    def __init__(
        self,
        runtime: Any,
        store: Store,
        client_for: Any,
        backend: Any,
        via: str,
        publish: Callable[[str, dict[str, Any]], None],
    ) -> None:
        self.runtime = runtime
        self.store = store
        self.client_for = client_for
        self.backend = backend
        self.via = via
        self.publish = publish

        self._lock = threading.Lock()
        self._cond = threading.Condition(threading.RLock())
        self._tasks: deque[tuple[str, int]] = deque()
        self._seen: set[tuple[str, int]] = set()
        self._inflight: set[tuple[str, int]] = set()
        self._paused = False
        self._stop = False

        self._ready: deque[tuple[str, int, Any]] = deque()  # prepared work, window-bound
        self._prefetch_threads: set[threading.Thread] = set()
        self._infer_threads: set[threading.Thread] = set()
        self._prefetch_by_id: dict[int, threading.Thread] = {}
        self._infer_by_id: dict[int, threading.Thread] = {}
        self._thread_id = 0

        self._rate: deque[float] = deque()
        self._rate_lock = threading.Lock()
        self._started = time.monotonic()

    # ---- control --------------------------------------------------------

    def start(self) -> None:
        self._ensure_threads()
        log.info(
            "pipeline started (window=%d prefetch=%d workers=%d)",
            self.runtime.window, self.runtime.prefetch_threads, self.runtime.effective_workers(),
        )

    def stop(self, drain_timeout: float = 30.0) -> None:
        with self._cond:
            self._stop = True
            self._paused = False
            for _ in self._infer_threads:
                self._ready.append(_STOP)
            self._cond.notify_all()
        deadline = time.monotonic() + drain_timeout
        threads = list(self._prefetch_threads) + list(self._infer_threads)
        for thread in threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(remaining)

    def submit(self, source: str, image_ids: list[int]) -> tuple[int, int]:
        """Enqueue ids, deduplicating against everything seen so far.
        Returns (new, already_known)."""
        new: list[int] = []
        with self._cond:
            for image_id in image_ids:
                if (source, image_id) not in self._seen:
                    self._seen.add((source, image_id))
                    self._tasks.append((source, image_id))
                    new.append(image_id)
            self._cond.notify_all()
        inserted, known = self.store.submit(source, new) if new else (0, len(image_ids))
        return inserted, len(image_ids) - inserted

    def submit_pairs(self, pairs: list[tuple[str, int]]) -> int:
        """Enqueue (source, id) pairs (retried/resumed tasks) grouped by
        source. The tasks are already persisted (they failed or were left
        pending), so the store-level dedup must be bypassed - the queue is
        authoritative here. Returns the number enqueued."""
        by_source: dict[str, list[int]] = {}
        for source, image_id in pairs:
            by_source.setdefault(source, []).append(image_id)
        enqueued = 0
        with self._cond:
            for src, ids in by_source.items():
                for image_id in ids:
                    self._seen.discard((src, image_id))  # allow a retry
                    self._seen.add((src, image_id))
                    self._tasks.append((src, image_id))
                    enqueued += 1
            self._cond.notify_all()
        return enqueued

    def pause(self) -> None:
        with self._cond:
            self._paused = True
            self._cond.notify_all()
        log.info("pipeline paused")

    def resume(self) -> None:
        with self._cond:
            self._paused = False
            self._cond.notify_all()
        log.info("pipeline resumed")

    def paused(self) -> bool:
        with self._cond:
            return self._paused

    def retry_failed(self) -> int:
        count = self.submit_pairs(self.store.retry_failed())
        log.info("retrying %d failed task(s)", count)
        return count

    def clear_results(self) -> int:
        return self.store.clear_results()

    def clear_tasks(self) -> int:
        with self._cond:
            self._tasks.clear()
            self._seen.clear()
        return self.store.clear_tasks()

    def reconfigure(self) -> None:
        """Apply runtime changes (worker/thread counts) live."""
        self._ensure_threads()

    def set_backend(self, backend: Any) -> None:
        """Hot-swap the model backend. In-flight inferences hold their own
        reference, so closing the old one here is safe."""
        old = self.backend
        with self._cond:
            self.backend = backend
        if old is not backend:
            old.close()
        log.info("pipeline backend switched to %s", getattr(backend, "name", type(backend).__name__))

    def set_via(self, via: str) -> None:
        """Hot-update the source string written into monbooru tags."""
        self.via = via

    # ---- stats ----------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._cond:
            pending = len(self._tasks)
            processing = len(self._inflight)
            paused = self._paused
            stopped = self._stop
        counts = self.store.stats()  # exact, survives retries/clears
        done, failed = counts["done"], counts["failed"]
        with self._rate_lock:
            now = time.monotonic()
            while self._rate and now - self._rate[0] > 60:
                self._rate.popleft()
            recent = len(self._rate)
            uptime = min(now - self._started, 60) or 1
            throughput = recent / uptime
            eta = (pending / throughput) if throughput > 0 else None
        total = done + failed + pending + processing
        return {
            "pending": pending,
            "processing": processing,
            "done": done,
            "failed": failed,
            "total": total,
            "paused": paused,
            "stopped": stopped,
            "window": self.runtime.window,
            "throughput": round(throughput, 2),
            "eta": int(eta) if eta is not None else None,
        }

    def _bump_done(self, ok: bool) -> None:
        del ok
        with self._rate_lock:
            self._rate.append(time.monotonic())

    # ---- worker threads -------------------------------------------------

    def _ensure_threads(self) -> None:
        want_prefetch = self.runtime.prefetch_threads
        want_infer = self.runtime.effective_workers()
        with self._cond:
            # Retire extras first: the loops check _is_retired on every
            # iteration and exit themselves.
            for tid in list(self._infer_by_id)[want_infer:]:
                self._infer_by_id.pop(tid, None)
            for tid in list(self._prefetch_by_id)[want_prefetch:]:
                self._prefetch_by_id.pop(tid, None)
            for _ in range(want_prefetch - len(self._prefetch_threads)):
                self._spawn_prefetch_locked()
            for _ in range(want_infer - len(self._infer_threads)):
                self._spawn_infer_locked()

    def _spawn_prefetch_locked(self) -> None:
        self._thread_id += 1
        tid = self._thread_id

        def run() -> None:
            try:
                self._prefetch_loop(tid)
            finally:
                with self._cond:
                    self._prefetch_threads.discard(self._prefetch_by_id.pop(tid, None))

        thread = threading.Thread(
            target=run, name=f"prefetch-{tid}", daemon=True
        )
        self._prefetch_threads.add(thread)
        self._prefetch_by_id[tid] = thread
        thread.start()

    def _spawn_infer_locked(self) -> None:
        self._thread_id += 1
        tid = self._thread_id

        def run() -> None:
            try:
                self._infer_loop(tid)
            finally:
                with self._cond:
                    self._infer_threads.discard(self._infer_by_id.pop(tid, None))

        thread = threading.Thread(
            target=run, name=f"infer-{tid}", daemon=True
        )
        self._infer_threads.add(thread)
        self._infer_by_id[tid] = thread
        thread.start()

    def _is_retired(self, kind: str, thread_id: int) -> bool:
        with self._cond:
            registry = self._prefetch_by_id if kind == "prefetch" else self._infer_by_id
            return thread_id not in registry

    def _prefetch_loop(self, thread_id: int) -> None:
        while True:
            with self._cond:
                if self._stop or self._is_retired("prefetch", thread_id):
                    return
                while (
                    not self._tasks
                    or len(self._inflight) >= self.runtime.window
                    or self._paused
                ):
                    if self._stop or self._is_retired("prefetch", thread_id):
                        return
                    self._cond.wait(timeout=0.5)
                source, image_id = self._tasks.popleft()
                self._inflight.add((source, image_id))
            client = self.client_for(source)
            self.store.mark_processing(source, image_id)
            if self._skip_requested(client, source, image_id):
                with self._cond:
                    self._inflight.discard((source, image_id))
                    self._cond.notify_all()
                self.publish("result", {"image_id": image_id, "status": "done", "tags": [], "error": ""})
                self._bump_done(True)
                continue
            try:
                data = client.fetch_image(image_id)
                from io import BytesIO

                from PIL import Image
                image = Image.open(BytesIO(data))
                prepared = self.backend.preprocess(image)
                del image, data
                with self._cond:
                    self._ready.append((source, image_id, prepared))
                    self._cond.notify_all()
            except Exception as exc:
                log.warning("prefetch %d failed: %s", image_id, exc)
                self._finish(source, image_id, ok=False, error=str(exc), tags=[])

    def _skip_requested(self, client: Any, source: str, image_id: int) -> bool:
        if not self.runtime.skip_tagged:
            return False
        try:
            status = client.image_status(image_id)
            if status.get("auto_tagged_at"):
                log.info("skip %d: already tagged by monbooru", image_id)
                self.store.mark_done(source, image_id, [])
                return True
        except Exception:
            pass  # treat as not skipped; a failed check must not drop the image
        return False

    def _infer_loop(self, thread_id: int) -> None:
        while True:
            if self._is_retired("infer", thread_id):
                return
            item = None
            with self._cond:
                if self._ready:
                    item = self._ready.popleft()
                else:
                    self._cond.wait(timeout=0.5)
            if item is None:
                continue
            if item is _STOP:
                return
            source, image_id, prepared = item
            try:
                self.store.mark_processing(source, image_id)
                tags = self.backend.tag(prepared, self.runtime)
                self.client_for(source).add_tags(image_id, tags, self.via)
                self._finish(source, image_id, ok=True, error="", tags=tags)
            except Exception as exc:
                log.warning("infer %d failed: %s", image_id, exc)
                self._finish(source, image_id, ok=False, error=str(exc), tags=[])
            del prepared

    def _finish(self, source: str, image_id: int, ok: bool, error: str, tags: list[str]) -> None:
        if ok:
            self.store.mark_done(source, image_id, tags)
        else:
            self.store.mark_failed(source, image_id, error, 0)
        with self._cond:
            self._inflight.discard((source, image_id))
            self._cond.notify_all()
        self.publish("result", {"image_id": image_id, "status": "done" if ok else "failed", "tags": tags, "error": error})
        self._bump_done(ok)