"""Thread-safe event bus bridging the pipeline threads into asyncio SSE.

publish() may be called from any thread; delivery happens on the event loop
via call_soon_threadsafe. Each SSE subscriber gets its own asyncio.Queue.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subs: set[asyncio.Queue[tuple[str, dict[str, Any]]]] = set()
        self._lock = threading.Lock()

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[tuple[str, dict[str, Any]]]:
        import asyncio

        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(maxsize=200)
        with self._lock:
            self._subs.add(queue)
        return queue

    def unsubscribe(self, queue: Any) -> None:
        with self._lock:
            self._subs.discard(queue)

    def publish(self, kind: str, data: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._deliver, kind, data)
        except RuntimeError:  # loop closing
            pass

    def _deliver(self, kind: str, data: dict[str, Any]) -> None:
        for queue in list(self._subs):
            try:
                queue.put_nowait((kind, data))
            except (asyncio.QueueFull, RuntimeError):
                pass