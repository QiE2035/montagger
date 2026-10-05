"""SQLite persistence for job metadata and API tokens.

Images never land here: a row is the durable mirror of a job's lifecycle
and its tag result, the same shape monloader's queue mirror keeps. WAL
mode with normal synchronous fsyncs; the traffic is one small row per
transition.
"""

from __future__ import annotations

import json
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    desc,
    event,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from . import logx

log = logx.get("store")

# Job lifecycle states.
QUEUED, RUNNING, DONE, ERROR, LOST, CANCELED = "queued", "running", "done", "error", "lost", "canceled"
_TERMINAL = frozenset((DONE, ERROR, LOST, CANCELED))


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(16), default="web")
    model: Mapped[str] = mapped_column(Text, default="")
    provider: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default=QUEUED)
    error: Mapped[str] = mapped_column(Text, default="")
    sha256: Mapped[str] = mapped_column(String(64), default="")
    md5: Mapped[str] = mapped_column(String(32), default="")
    bytes_len: Mapped[int] = mapped_column(Integer, default=0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    elapsed_ms: Mapped[int] = mapped_column(Integer, default=0)
    tags_json: Mapped[str] = mapped_column(Text, default="[]")
    rating: Mapped[str] = mapped_column(Text, default="")
    monbooru_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    pushed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


Index("jobs_status_idx", JobRow.status)
Index("jobs_created_idx", JobRow.created_at)
Index("jobs_sha_model_idx", JobRow.sha256, JobRow.model)


class TokenRow(Base):
    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text)
    token: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


def _open_engine(path: Path) -> Engine:
    engine = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=10000")
        cursor.close()

    return engine


class Store:
    """Repo functions over one engine; callers never touch a Session."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = _open_engine(path)
        Base.metadata.create_all(self._engine)
        self._sessions = sessionmaker(self._engine, expire_on_commit=False)

    def _run(self, fn):
        with self._sessions() as session:
            result = fn(session)
            session.commit()
            return result

    # -- jobs ----------------------------------------------------------------

    def create_job(self, *, job_id: str, filename: str, source: str, model: str, sha256: str, md5: str, bytes_len: int) -> None:
        def fn(session: Session):
            session.add(
                JobRow(
                    id=job_id, filename=filename, source=source, model=model,
                    sha256=sha256, md5=md5, bytes_len=bytes_len, status=QUEUED,
                )
            )
        self._run(fn)

    def set_running(self, job_id: str, provider: str) -> None:
        def fn(session: Session):
            session.execute(
                update(JobRow).where(JobRow.id == job_id).values(
                    status=RUNNING, provider=provider, started_at=utcnow()
                )
            )
        self._run(fn)

    def finish_done(self, job_id: str, *, provider: str, width: int, height: int, elapsed_ms: int, tags: list[dict], rating: str) -> None:
        def fn(session: Session):
            session.execute(
                update(JobRow).where(JobRow.id == job_id).values(
                    status=DONE, provider=provider, width=width, height=height,
                    elapsed_ms=elapsed_ms, tags_json=json.dumps(tags), rating=rating or "",
                    finished_at=utcnow(),
                )
            )
        self._run(fn)

    def finish_error(self, job_id: str, error: str) -> None:
        def fn(session: Session):
            session.execute(
                update(JobRow).where(JobRow.id == job_id).values(
                    status=ERROR, error=error[:2000], finished_at=utcnow()
                )
            )
        self._run(fn)

    def set_status(self, job_id: str, status: str) -> None:
        def fn(session: Session):
            session.execute(update(JobRow).where(JobRow.id == job_id).values(status=status))
        self._run(fn)

    def get_job(self, job_id: str) -> JobRow | None:
        def fn(session: Session):
            return session.get(JobRow, job_id)
        return self._run(fn)

    def find_by_hash(self, sha256: str, model: str) -> JobRow | None:
        """Most recent done row for a hash+model, for dedup reuse."""
        def fn(session: Session):
            return session.execute(
                select(JobRow)
                .where(JobRow.sha256 == sha256, JobRow.model == model, JobRow.status == DONE)
                .order_by(JobRow.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
        return self._run(fn)

    def find_pushed_by_sha(self, sha256: str, exclude_job_id: str = "") -> JobRow | None:
        """Any other row of this file already pushed to monbooru: later
        models of the same upload enrich that image instead of creating a
        second one."""
        def fn(session: Session):
            query = (
                select(JobRow)
                .where(JobRow.sha256 == sha256, JobRow.monbooru_id.is_not(None))
                .order_by(JobRow.created_at.desc())
                .limit(1)
            )
            if exclude_job_id:
                query = query.where(JobRow.id != exclude_job_id)
            return session.execute(query).scalar_one_or_none()
        return self._run(fn)

    def list_jobs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        search: str | None = None,
    ) -> tuple[list[JobRow], int]:
        """(rows, total): newest first, optional status filter and a
        filename/model/tag text search, total = matching rows regardless of
        the page window."""
        def fn(session: Session):
            query = select(JobRow)
            if status:
                query = query.where(JobRow.status == status)
            if search:
                needle = f"%{search.strip()}%"
                query = query.where(
                    JobRow.filename.ilike(needle)
                    | JobRow.model.ilike(needle)
                    | JobRow.tags_json.ilike(needle)
                )
            total = session.execute(
                select(func.count()).select_from(query.subquery())
            ).scalar_one()
            rows = list(
                session.scalars(
                    query.order_by(JobRow.created_at.desc()).limit(limit).offset(offset)
                )
            )
            return rows, int(total)
        return self._run(fn)

    def list_job_groups(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
    ) -> tuple[list[JobRow], int]:
        """(rows, total_images): the page is a window over IMAGES (sha256
        groups), newest image first, and rows carry EVERY job of those
        images - so a multi-model image's results never straddle a page
        boundary. Rows are ordered image by image, upload order within an
        image. Under search a group keeps only its matching jobs."""
        def fn(session: Session):
            needle = f"%{search.strip()}%" if search else None
            grouped = (
                select(JobRow.sha256, func.max(JobRow.created_at).label("last"))
                .group_by(JobRow.sha256)
            )
            if needle is not None:
                grouped = grouped.where(
                    JobRow.filename.ilike(needle)
                    | JobRow.model.ilike(needle)
                    | JobRow.tags_json.ilike(needle)
                )
            total = session.execute(
                select(func.count()).select_from(grouped.subquery())
            ).scalar_one()
            page = (
                grouped.order_by(desc("last"), JobRow.sha256)
                .limit(limit)
                .offset(offset)
                .subquery()
            )
            shas = [row[0] for row in session.execute(select(page.c.sha256))]
            if not shas:
                return [], int(total)
            condition = JobRow.sha256.in_(shas)
            if needle is not None:
                condition = condition & (
                    JobRow.filename.ilike(needle)
                    | JobRow.model.ilike(needle)
                    | JobRow.tags_json.ilike(needle)
                )
            rows = list(session.scalars(select(JobRow).where(condition)))
            position = {sha: i for i, sha in enumerate(shas)}
            rows.sort(key=lambda r: (position.get(r.sha256, 0), r.created_at))
            return rows, int(total)
        return self._run(fn)

    def count_status(self, *statuses: str) -> int:
        def fn(session: Session):
            return int(
                session.execute(
                    select(func.count()).select_from(JobRow).where(JobRow.status.in_(statuses))
                ).scalar_one()
            )
        return self._run(fn)

    def delete_job(self, job_id: str) -> bool:
        def fn(session: Session):
            deleted = session.execute(delete(JobRow).where(JobRow.id == job_id)).rowcount
            return bool(deleted)
        return self._run(fn)

    def list_job_ids(self, *, status: str | None = None) -> list[str]:
        def fn(session: Session):
            query = select(JobRow.id)
            if status:
                query = query.where(JobRow.status == status)
            query = query.order_by(JobRow.created_at.desc(), JobRow.id.desc())
            return list(session.scalars(query))
        return self._run(fn)

    def delete_jobs(self, job_ids: list[str]) -> int:
        if not job_ids:
            return 0
        def fn(session: Session):
            return int(
                session.execute(delete(JobRow).where(JobRow.id.in_(job_ids))).rowcount or 0
            )
        return self._run(fn)

    def clear_finished(self) -> int:
        def fn(session: Session):
            deleted = session.execute(
                delete(JobRow).where(JobRow.status.in_(_TERMINAL))
            ).rowcount
            return int(deleted or 0)
        return self._run(fn)

    def mark_unfinished_lost(self) -> int:
        """Boot-time sweep: pending bytes died with the old process."""
        def fn(session: Session):
            updated = session.execute(
                update(JobRow)
                .where(JobRow.status.in_((QUEUED, RUNNING)))
                .values(status=LOST, error="service restarted before this job ran; push the file again")
            ).rowcount
            return int(updated or 0)
        n = self._run(fn)
        if n:
            log.warning("marked %d unfinished job(s) lost", n)
        return n

    def purge_history(self, days: int) -> int:
        if days <= 0:
            return 0
        cutoff = utcnow() - timedelta(days=days)
        def fn(session: Session):
            deleted = session.execute(
                delete(JobRow).where(JobRow.status.in_(_TERMINAL), JobRow.created_at < cutoff)
            ).rowcount
            return int(deleted or 0)
        return self._run(fn)

    def set_pushed(self, job_id: str, monbooru_id: int) -> None:
        def fn(session: Session):
            session.execute(
                update(JobRow).where(JobRow.id == job_id).values(
                    monbooru_id=monbooru_id, pushed_at=utcnow()
                )
            )
        self._run(fn)

    # -- tokens ----------------------------------------------------------------

    def add_token(self, name: str) -> str:
        token = "mtag_" + secrets.token_hex(24)

        def fn(session: Session):
            session.add(TokenRow(name=name or "unnamed", token=token))
        self._run(fn)
        return token

    def list_tokens(self) -> list[TokenRow]:
        def fn(session: Session):
            return list(session.scalars(select(TokenRow).order_by(TokenRow.id)))
        return self._run(fn)

    def delete_token(self, token_id: int) -> bool:
        def fn(session: Session):
            deleted = session.execute(delete(TokenRow).where(TokenRow.id == token_id)).rowcount
            return bool(deleted)
        return self._run(fn)

    def token_known(self, token: str) -> bool:
        if not token:
            return False
        def fn(session: Session):
            row = session.execute(select(TokenRow.id).where(TokenRow.token == token)).first()
            return row is not None
        return self._run(fn)

    def has_tokens(self) -> bool:
        def fn(session: Session):
            return session.execute(select(TokenRow.id).limit(1)).first() is not None
        return self._run(fn)


def new_job_id() -> str:
    return uuid.uuid4().hex
