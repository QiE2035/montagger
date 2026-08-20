"""SQLite persistence for tasks and results.

Single table `tasks` doubles as the durable task queue and the result log.
The primary key is (source, image_id): source is the monbooru instance a
task came from (its url), so identical ids across paired instances never
collide and each task knows where to fetch from and write back to. Repeated
pushes of the same source+id deduplicate for free. The pipeline owns the
single write connection (guarded by a lock); the WebUI uses short-lived
read connections per request. Schema migrations ride on PRAGMA user_version.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

PENDING = "pending"
PROCESSING = "processing"
DONE = "done"
FAILED = "failed"

STATUSES = (PENDING, PROCESSING, DONE, FAILED)

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS tasks (
    image_id   INTEGER PRIMARY KEY,
    status     TEXT    NOT NULL DEFAULT 'pending',
    tags       TEXT    NOT NULL DEFAULT '[]',
    error      TEXT    NOT NULL DEFAULT '',
    retries    INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    done_at    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tasks_status_done ON tasks (status, done_at);
"""

_SCHEMA_V2 = """
CREATE TABLE tasks (
    source     TEXT    NOT NULL DEFAULT '',
    image_id   INTEGER NOT NULL,
    status     TEXT    NOT NULL DEFAULT 'pending',
    tags       TEXT    NOT NULL DEFAULT '[]',
    error      TEXT    NOT NULL DEFAULT '',
    retries    INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    done_at    INTEGER,
    PRIMARY KEY (source, image_id)
);
CREATE INDEX IF NOT EXISTS idx_tasks_status_done ON tasks (status, done_at);
"""


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._wlock = threading.Lock()
        self._w = sqlite3.connect(self.db_path, check_same_thread=False)
        self._w.execute("PRAGMA journal_mode=WAL")
        self._w.execute("PRAGMA busy_timeout=5000")
        self._migrate()

    # ---- schema ---------------------------------------------------------

    def _migrate(self) -> None:
        version = self._w.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            self._w.executescript(_SCHEMA_V1)
            self._w.execute("PRAGMA user_version = 1")
            self._w.commit()
            version = 1
        if version < 2:
            # Multi-instance pairing: tasks become (source, image_id); a
            # fresh table is cheaper than an ALTER on a compound key.
            self._w.execute("DROP TABLE IF EXISTS tasks_old")
            self._w.execute(
                "CREATE TABLE tasks_old ("
                "image_id   INTEGER PRIMARY KEY,"
                "status     TEXT    NOT NULL DEFAULT 'pending',"
                "tags       TEXT    NOT NULL DEFAULT '[]',"
                "error      TEXT    NOT NULL DEFAULT '',"
                "retries    INTEGER NOT NULL DEFAULT 0,"
                "created_at INTEGER NOT NULL,"
                "done_at    INTEGER)"
            )
            self._w.execute(
                "INSERT INTO tasks_old (image_id, status, tags, error, retries, created_at, done_at) "
                "SELECT image_id, status, tags, error, retries, created_at, done_at FROM tasks"
            )
            self._w.execute("DROP TABLE tasks")
            self._w.executescript(_SCHEMA_V2)
            self._w.execute(
                "INSERT INTO tasks (source, image_id, status, tags, error, retries, created_at, done_at) "
                "SELECT '', image_id, status, tags, error, retries, created_at, done_at FROM tasks_old"
            )
            self._w.execute("DROP TABLE tasks_old")
            self._w.execute("PRAGMA user_version = 2")
            self._w.commit()

    # ---- writes (single connection, locked) -----------------------------

    def submit(self, source: str, image_ids: list[int]) -> tuple[int, int]:
        """Insert new tasks. Returns (inserted, already_known)."""
        if not image_ids:
            return 0, 0
        now = int(time.time())
        with self._wlock:
            cur = self._w.executemany(
                "INSERT OR IGNORE INTO tasks (source, image_id, status, created_at) VALUES (?, ?, ?, ?)",
                [(source, i, PENDING, now) for i in image_ids],
            )
            self._w.commit()
            inserted = cur.rowcount if cur.rowcount >= 0 else len(image_ids)
            return inserted, max(0, len(image_ids) - inserted)

    def mark_processing(self, source: str, image_id: int) -> None:
        with self._wlock:
            self._w.execute(
                "UPDATE tasks SET status = ? WHERE source = ? AND image_id = ?",
                (PROCESSING, source, image_id),
            )
            self._w.commit()

    def mark_done(self, source: str, image_id: int, tags: list[str]) -> None:
        import json

        with self._wlock:
            self._w.execute(
                "UPDATE tasks SET status = ?, tags = ?, error = '', done_at = ? WHERE source = ? AND image_id = ?",
                (DONE, json.dumps(tags, ensure_ascii=False), int(time.time()), source, image_id),
            )
            self._w.commit()

    def mark_failed(self, source: str, image_id: int, error: str, retries: int) -> None:
        with self._wlock:
            self._w.execute(
                "UPDATE tasks SET status = ?, error = ?, retries = ?, done_at = ? WHERE source = ? AND image_id = ?",
                (FAILED, error[:2000], retries, int(time.time()), source, image_id),
            )
            self._w.commit()

    def retry_failed(self) -> list[tuple[str, int]]:
        """Flip failed tasks back to pending; returns (source, id) pairs."""
        with self._wlock:
            rows = self._w.execute(
                "SELECT source, image_id FROM tasks WHERE status = ?", (FAILED,)
            ).fetchall()
            pairs = [(r[0], r[1]) for r in rows]
            if pairs:
                self._w.executemany(
                    "UPDATE tasks SET status = ?, error = '', done_at = NULL WHERE source = ? AND image_id = ?",
                    [(PENDING, source, image_id) for source, image_id in pairs],
                )
                self._w.commit()
            return pairs

    def resume_ids(self) -> list[tuple[str, int]]:
        """(source, id) pairs to re-enqueue at startup: pending (never
        finished), processing (left over from a crash) and failed
        (retryable)."""
        with self._wlock:
            rows = self._w.execute(
                "SELECT source, image_id FROM tasks WHERE status IN (?, ?, ?) ORDER BY created_at",
                (PENDING, PROCESSING, FAILED),
            ).fetchall()
            pairs = [(r[0], r[1]) for r in rows]
            if pairs:
                self._w.executemany(
                    "UPDATE tasks SET status = ?, done_at = NULL WHERE source = ? AND image_id = ?",
                    [(PENDING, source, image_id) for source, image_id in pairs],
                )
            self._w.commit()
            return pairs

    def clear_results(self) -> int:
        with self._wlock:
            cur = self._w.execute("DELETE FROM tasks WHERE status IN (?, ?)", (DONE, FAILED))
            self._w.commit()
            return cur.rowcount

    def clear_tasks(self) -> int:
        with self._wlock:
            cur = self._w.execute("DELETE FROM tasks")
            self._w.commit()
            return cur.rowcount

    # ---- reads (short-lived connections) --------------------------------

    def stats(self) -> dict[str, int]:
        with self._read() as db:
            rows = db.execute(
                "SELECT status, COUNT(*) FROM tasks GROUP BY status"
            ).fetchall()
        counts = {status: 0 for status in STATUSES}
        for status, count in rows:
            counts[status] = count
        return counts

    def results(
        self, page: int, page_size: int, status: str | None = None
    ) -> tuple[list[dict[str, Any]], int]:
        page = max(page, 1)
        size = max(1, min(page_size, 200))
        offset = (page - 1) * size
        where = "WHERE status = ?" if status in STATUSES else ""
        params: tuple[Any, ...] = (status,) if status in STATUSES else ()
        with self._read() as db:
            total = db.execute(
                f"SELECT COUNT(*) FROM tasks {where}", params
            ).fetchone()[0]
            rows = db.execute(
                f"SELECT source, image_id, status, tags, error, retries, created_at, done_at "
                f"FROM tasks {where} ORDER BY done_at DESC, image_id DESC LIMIT ? OFFSET ?",
                params + (size, offset),
            ).fetchall()
        keys = ("source", "image_id", "status", "tags", "error", "retries", "created_at", "done_at")
        return [dict(zip(keys, r)) for r in rows], total

    # ---- lifecycle ------------------------------------------------------

    def close(self) -> None:
        with self._wlock:
            self._w.close()

    def _read(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        return db