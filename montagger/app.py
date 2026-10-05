"""The FastAPI application: API routes, SSE, auth, static SPA hosting.

Uploads are raw byte streams read chunk by chunk - the multipart machinery
of the framework is never engaged, which is what keeps large pushes out of
spooled temporary files.
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import hashlib
import hmac
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from . import __version__, auth, logx, store
from .config import Provider, normalize_thresholds
from .engine.backend import Engine
from .engine.downloader import DownloadError
from .engine.session import available_providers
from .queue import Job, QueueFull, Runner, UnknownJob

log = logx.get("api")

SESSION_COOKIE = auth._SESSION_COOKIE


# -- request bodies -------------------------------------------------------------
# Declared as pydantic models so malformed input answers with the framework's
# standard 422 instead of a late 500. relay/tag stays hand-parsed: monbooru
# expects every reply to be HTTP 200 with an ok flag.

class MemoryReleaseBody(BaseModel):
    models: list[str] | None = None  # named sessions to drop; absent = all


class LoginBody(BaseModel):
    password: str = ""


class JobsBatchBody(BaseModel):
    action: Literal["delete", "cancel"]
    ids: list[str] = Field(default_factory=list)
    all_done: bool = False
    all_queued: bool = False


class TokenCreateBody(BaseModel):
    name: str = ""


class PairBody(BaseModel):
    api_url: str = ""


# Settings posts are partial: only the keys the UI touched are present, so
# every field is optional and only the explicitly provided ones reach the
# config. password / token follow one rule: an empty string clears, an
# explicitly sent null or an absent key means untouched.
class ServerUpdate(BaseModel):
    bind_address: str | None = None
    base_url: str | None = None


class AuthUpdate(BaseModel):
    password: str | None = None
    session_days: int | None = None


class ModelsUpdate(BaseModel):
    path: str | None = None
    default: str | None = None
    default_models: list[str] | None = None
    execution_provider: str | None = None
    device_id: int | None = None
    intra_op_threads: int | None = None
    max_upload_mb: int | None = None
    idle_unload_min: int | None = None
    isolated: bool | None = None
    disabled_categories: list[str] | None = None


class QueueUpdate(BaseModel):
    max_pending: int | None = None
    history_days: int | None = None


class HfUpdate(BaseModel):
    token: str | None = None
    endpoint: str | None = None


class MonbooruUpdate(BaseModel):
    api_url: str | None = None
    web_url: str | None = None
    token: str | None = None
    push_tags: bool | None = None
    push_images: bool | None = None
    gallery: str | None = None


class LogUpdate(BaseModel):
    debug: bool | None = None


class SettingsBody(BaseModel):
    server: ServerUpdate | None = None
    auth: AuthUpdate | None = None
    models: ModelsUpdate | None = None
    queue: QueueUpdate | None = None
    hf: HfUpdate | None = None
    monbooru: MonbooruUpdate | None = None
    log: LogUpdate | None = None
    thresholds: dict | None = None


def _rss_mb() -> float:
    """Current resident set in MiB; ru_maxrss (a peak) as the fallback."""
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            resident_pages = int(fh.read().split()[1])
        return round(resident_pages * os.sysconf("SC_PAGE_SIZE") / 1048576, 1)
    except Exception:
        try:
            import resource

            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        except Exception:
            return 0.0


def _malloc_trim() -> bool:
    """glibc only: hand freed heap back to the OS right now."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        return True
    except Exception:
        return False


class AppContext:
    def __init__(
        self,
        cfg: Provider,
        engine: Engine,
        store_: store.Store,
        runner: Runner,
        integration=None,
    ):
        self.cfg = cfg
        self.engine = engine
        self.store = store_
        self.runner = runner
        self.integration = integration
        self.sessions = auth.SessionStore()


def create_app(ctx: AppContext) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Boot sweep: unfinished rows from a previous process lost their
        # bytes with it; then the worker starts and old history expires.
        await ctx.runner.recover()
        ctx.runner.start()
        purged = ctx.store.purge_history(ctx.cfg.current().queue.history_days)
        if purged:
            log.info("purged %d expired history rows", purged)
        sweep = asyncio.create_task(_idle_sweep())
        try:
            yield
        finally:
            sweep.cancel()
            try:
                await sweep
            except (asyncio.CancelledError, Exception):
                pass
            await ctx.runner.stop()
            if ctx.integration is not None:
                ctx.integration.stop()

    async def _idle_sweep():
        """models.idle_unload_min: drop sessions that sat unused, giving
        their RAM/VRAM back without anyone opening the settings page."""
        while True:
            await asyncio.sleep(60)
            try:
                minutes = ctx.cfg.current().models.idle_unload_min
                if minutes <= 0:
                    continue
                for info in ctx.engine.loaded_models():
                    idle = info.get("idle_s")
                    if idle is not None and idle >= minutes * 60:
                        ctx.engine.unload(info["name"])
                        log.info(
                            "idle sweep: %s unused for %dm, unloaded", info["name"], minutes
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("idle sweep failed")

    app = FastAPI(
        title="montagger", version=__version__, docs_url="/docs", lifespan=lifespan
    )
    app.state.ctx = ctx

    # -- auth helpers ------------------------------------------------------

    def auth_state(request: Request) -> tuple[bool, bool]:
        """(guard_active, request_authenticated).

        The guard is the UI password alone: tokens exist so phones and
        scripts can pass the same guard without a browser session, but
        creating one never locks the web UI on its own.
        """
        cfg = ctx.cfg.current()
        guard = bool(cfg.auth.password)
        if not guard:
            return False, True
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer ") and ctx.store.token_known(
            authorization[7:].strip()
        ):
            return True, True
        if ctx.sessions.check(request.cookies.get(SESSION_COOKIE)):
            return True, True
        return True, False

    def require_auth(request: Request) -> None:
        guard, ok = auth_state(request)
        if guard and not ok:
            raise HTTPException(status_code=401, detail="authentication required")

    # -- health / memory --------------------------------------------------------

    @app.get("/health")
    @app.get("/api/v1/health")
    def health():
        return {
            "version": __version__,
            "rss_mb": round(_rss_mb() + ctx.engine.child_rss_mb(), 1),
            "models_loaded": ctx.engine.loaded_models(),
        }

    @app.post("/api/v1/memory/release")
    async def memory_release(request: Request, body: MemoryReleaseBody | None = None):
        """Manual, proactive release: unload model sessions (all, or the
        named ones), forget finished Job objects, collect, and hand freed
        heap back to the OS where the platform allows. Safe to call
        anytime; the next tagging request just reloads what it needs."""
        require_auth(request)
        names = body.models if body is not None else None
        unloaded: list[str] = []
        if names:
            for name in names:
                if ctx.engine.unload(str(name)):
                    unloaded.append(str(name))
        else:
            unloaded = [info["name"] for info in ctx.engine.loaded_models()]
            ctx.engine.unload_all()
        jobs = ctx.runner.trim_memory()
        gc.collect()
        trimmed = _malloc_trim()
        return {
            "ok": True,
            "unloaded": unloaded,
            "jobs": jobs,
            "rss_mb": round(_rss_mb() + ctx.engine.child_rss_mb(), 1),
            "malloc_trim": trimmed,
        }

    @app.post("/api/v1/history/purge")
    def history_purge(request: Request):
        """Manually run what history_days does at boot."""
        require_auth(request)
        days = ctx.cfg.current().queue.history_days
        return {"deleted": ctx.store.purge_history(days)}

    # -- auth -----------------------------------------------------------------

    @app.get("/api/v1/auth/state")
    def auth_state_endpoint(request: Request):
        guard, ok = auth_state(request)
        return {"guard_active": guard, "authenticated": ok, "open_lan": not guard}

    @app.post("/api/v1/auth/login")
    async def login(request: Request, body: LoginBody):
        cfg = ctx.cfg.current()
        if not cfg.auth.password:
            raise HTTPException(status_code=400, detail="no password configured")
        if not hmac.compare_digest(body.password.encode(), cfg.auth.password.encode()):
            raise HTTPException(status_code=401, detail="wrong password")
        sid = ctx.sessions.new(cfg.auth.session_days)
        response = JSONResponse({"ok": True})
        response.set_cookie(
            SESSION_COOKIE,
            sid,
            max_age=cfg.auth.session_days * 86400,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return response

    @app.post("/api/v1/auth/logout")
    async def logout(request: Request):
        ctx.sessions.delete(request.cookies.get(SESSION_COOKIE))
        response = JSONResponse({"ok": True})
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    # -- tagging ---------------------------------------------------------------

    @app.post("/api/v1/tag")
    async def tag_image(
        request: Request,
        model: str | None = Query(default=None),
        wait: bool = Query(default=False),
        filename: str = Query(default=""),
    ):
        require_auth(request)
        cfg = ctx.cfg.current()
        cap = cfg.models.max_upload_mb * 1024 * 1024
        sha256 = hashlib.sha256()
        md5 = hashlib.md5()
        buf = bytearray()
        async for chunk in request.stream():
            if not chunk:
                continue
            if len(buf) + len(chunk) > cap:
                raise HTTPException(
                    status_code=413,
                    detail=f"upload exceeds {cfg.models.max_upload_mb} MiB",
                )
            buf += chunk
            sha256.update(chunk)
            md5.update(chunk)
        if not buf:
            raise HTTPException(
                status_code=400, detail="empty body: send raw image bytes"
            )

        # One job per model; the decoded bytes object is shared, never
        # copied per model.
        names = ctx.engine.resolve_model_list(model)
        if not names:
            raise HTTPException(status_code=400, detail="no model configured")
        # Multi-model mode returns an array even for one model; only the
        # legacy single-default path keeps the bare-object shape.
        multi = len(names) > 1 or bool(ctx.cfg.current().models.default_models)
        data = bytes(buf)
        del buf
        base = Job(
            id="",
            filename=(filename or request.headers.get("x-filename") or "image").strip()[:200],
            model="",
            source="api",
            data=data,
            sha256=sha256.hexdigest(),
            md5=md5.hexdigest(),
        )

        jobs: list[Job] = []
        dedup_ids: set[str] = set()
        for name in names:
            job = Job(
                id=store.new_job_id(),
                filename=base.filename,
                model=name,
                source=base.source,
                data=base.data,
                sha256=base.sha256,
                md5=base.md5,
            )
            # Dedup per model: a byte-identical file tagged by the same
            # model reuses its result without touching the accelerator.
            prior = ctx.store.find_by_hash(job.sha256, name)
            if prior is not None and prior.status == store.DONE:
                ctx.store.create_job(
                    job_id=job.id,
                    filename=job.filename,
                    source=job.source,
                    model=name,
                    sha256=job.sha256,
                    md5=job.md5,
                    bytes_len=len(job.data),
                )
                ctx.store.finish_done(
                    job.id,
                    provider=prior.provider,
                    width=prior.width,
                    height=prior.height,
                    elapsed_ms=prior.elapsed_ms,
                    tags=json.loads(prior.tags_json or "[]"),
                    rating=prior.rating,
                )
                job.status = store.DONE  # never re-queued, never re-run
                dedup_ids.add(job.id)
                jobs.append(job)
                continue
            ctx.store.create_job(
                job_id=job.id,
                filename=job.filename,
                source=job.source,
                model=name,
                sha256=job.sha256,
                md5=job.md5,
                bytes_len=len(job.data),
            )
            jobs.append(job)

        fresh = [j for j in jobs if j.status == store.QUEUED]
        if wait:
            for job in fresh:
                try:
                    await ctx.runner.run_inline(job)
                except asyncio.CancelledError:
                    ctx.runner.drop_if_queued(job.id)
                    raise
            results = []
            ok_all = True
            for job in jobs:
                row = ctx.store.get_job(job.id)
                if row is None:
                    ok_all = ok_all and job.status == store.DONE
                    results.append(job.public())
                    continue
                d = _row_dict(row)
                d["dedup"] = job.id in dedup_ids
                ok_all = ok_all and row.status == store.DONE
                results.append(d)
            body = results if multi else results[0]
            return JSONResponse(body, status_code=200 if ok_all else 502)
        try:
            for job in fresh:
                await ctx.runner.submit(job)
        except QueueFull as err:
            for job in fresh:
                ctx.store.set_status(job.id, store.CANCELED)
            raise HTTPException(
                status_code=503, detail=str(err), headers={"Retry-After": "5"}
            ) from err
        payloads = []
        for job in jobs:
            if job.status == store.QUEUED:
                payloads.append(job.public())
            else:
                row = ctx.store.get_job(job.id)
                payloads.append(_row_dict(row) if row is not None else job.public())
        body = payloads if multi else payloads[0]
        return JSONResponse(body, status_code=202)

    @app.get("/api/v1/jobs")
    def jobs(
        request: Request,
        limit: int = Query(default=50, le=500),
        offset: int = Query(default=0, ge=0),
        status: str | None = None,
        q: str | None = None,
        grouped: bool = Query(default=False),
    ):
        require_auth(request)
        # grouped: the page is a window over images (sha256), and each page
        # carries every job of those images, so the UI's per-image grouping
        # never splits a multi-model image across pages.
        if grouped:
            rows, total = ctx.store.list_job_groups(limit=limit, offset=offset, search=q)
        else:
            rows, total = ctx.store.list_jobs(limit=limit, offset=offset, status=status, search=q)
        return {
            "jobs": [_row_dict(row, live=_live_overlay(row.id)) for row in rows],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def job_cancel(request: Request, job_id: str):
        require_auth(request)
        if not ctx.runner.cancel(job_id):
            raise HTTPException(status_code=409, detail="job is not queued anymore")
        return {"ok": True}

    @app.post("/api/v1/jobs/batch")
    async def jobs_batch(request: Request, body: JobsBatchBody):
        """{action: delete|cancel, ids?: [...], all_done?: true,
        all_queued?: true} - the UI's select-all helpers."""
        require_auth(request)
        affected: list[str] = list(body.ids)
        if body.all_done:
            affected.extend(ctx.store.list_job_ids(status=store.DONE))
        if body.all_queued:
            affected.extend(ctx.runner.queued_ids())
        affected = list(dict.fromkeys(affected))
        if not affected:
            return {"ok": True, "affected": 0}
        if body.action == "cancel":
            n = sum(1 for jid in affected if ctx.runner.cancel(jid))
            return {"ok": True, "affected": n}
        n = ctx.store.delete_jobs(affected)
        for jid in affected:
            try:
                ctx.runner.drop_if_queued(jid)
            except Exception:
                pass
        return {"ok": True, "affected": n}

    @app.get("/api/v1/jobs/{job_id}")
    def job_detail(request: Request, job_id: str):
        require_auth(request)
        row = ctx.store.get_job(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="no such job")
        return _row_dict(row, live=_live_overlay(job_id))

    @app.get("/api/v1/jobs/{job_id}/preview")
    def job_preview(request: Request, job_id: str):
        require_auth(request)
        try:
            job = ctx.runner.get(job_id)
        except UnknownJob:
            raise HTTPException(
                status_code=404,
                detail="image bytes are gone (kept for recent jobs only)",
            ) from None
        if not job.data:
            raise HTTPException(
                status_code=404,
                detail="image bytes are gone (kept for recent jobs only)",
            )
        return Response(content=job.data, media_type="application/octet-stream")

    @app.delete("/api/v1/jobs/{job_id}")
    def job_delete(request: Request, job_id: str):
        require_auth(request)
        if not ctx.store.delete_job(job_id):
            raise HTTPException(status_code=404, detail="no such job")
        return {"ok": True}

    @app.post("/api/v1/jobs/clear")
    def jobs_clear(request: Request):
        require_auth(request)
        return {"deleted": ctx.store.clear_finished()}

    # -- SSE ---------------------------------------------------------------------

    @app.get("/api/v1/events")
    async def events(request: Request):
        require_auth(request)
        sub = ctx.runner.subscribe()

        async def stream():
            try:
                yield ": connected\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        payload = await asyncio.wait_for(sub.get(), timeout=15.0)
                    except TimeoutError:
                        yield ": ping\n\n"
                        continue
                    yield f"data: {payload}\n\n"
            finally:
                ctx.runner.unsubscribe(sub)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- models --------------------------------------------------------------------

    @app.get("/api/v1/models")
    def models(request: Request):
        require_auth(request)
        cfg = ctx.cfg.current()
        out = []
        for status in ctx.engine.statuses():
            entry = dict(status.__dict__)
            g, cats, top_k, disabled = ctx.engine.thresholds(status.name)
            entry["effective"] = {
                "global": g,
                "categories": cats,
                "top_k": top_k,
                "disabled": disabled,
            }
            # What this model CAN emit (GPU-free): drives the threshold
            # dialog's rows. Catalog entry if present, else derive from the
            # installed labels.
            emitted = list(status.emitted_categories) or ctx.engine.emitted_categories(status.name)
            if not emitted and entry.get("default_thresholds"):
                emitted = sorted(status.default_thresholds)
            entry["emitted_categories"] = emitted
            out.append(entry)
        return {
            "default": cfg.models.default,
            "default_models": cfg.models.default_models,
            "provider": ctx.engine.provider,
            "provider_short": ctx.engine.provider_short,
            "available_providers": available_providers(),
            "models": out,
        }

    @app.post("/api/v1/models/{name}/install")
    async def model_install(request: Request, name: str):
        require_auth(request)
        try:
            model_dir = await asyncio.to_thread(ctx.engine.install, name)
        except DownloadError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        except Exception as err:
            raise HTTPException(status_code=502, detail=str(err)) from err
        return {"ok": True, "path": str(model_dir)}

    @app.delete("/api/v1/models/{name}")
    async def model_remove(request: Request, name: str):
        require_auth(request)
        try:
            await asyncio.to_thread(ctx.engine.remove, name)
        except Exception as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        return {"ok": True}

    # -- tokens ---------------------------------------------------------------------

    @app.get("/api/v1/tokens")
    def tokens_list(request: Request):
        require_auth(request)
        return [
            {
                "id": t.id,
                "name": t.name,
                "token": t.token,
                "created_at": _iso(t.created_at),
            }
            for t in ctx.store.list_tokens()
        ]

    @app.post("/api/v1/tokens")
    async def tokens_create(request: Request, body: TokenCreateBody):
        require_auth(request)
        token = ctx.store.add_token(body.name)
        return {"token": token}

    @app.delete("/api/v1/tokens/{token_id}")
    async def tokens_delete(request: Request, token_id: int):
        require_auth(request)
        if not ctx.store.delete_token(token_id):
            raise HTTPException(status_code=404, detail="no such token")
        return {"ok": True}

    # -- settings ----------------------------------------------------------------------

    @app.get("/api/v1/settings")
    def settings_get(request: Request):
        require_auth(request)
        cfg = ctx.cfg.current()
        return {
            "server": {
                "bind_address": cfg.server.bind_address,
                "base_url": cfg.server.base_url,
            },
            "auth": {
                "has_password": bool(cfg.auth.password),
                "session_days": cfg.auth.session_days,
            },
            "models": {
                "path": cfg.models.path,
                "default": cfg.models.default,
                "default_models": cfg.models.default_models,
                "execution_provider": cfg.models.execution_provider,
                "device_id": cfg.models.device_id,
                "intra_op_threads": cfg.models.intra_op_threads,
                "max_upload_mb": cfg.models.max_upload_mb,
                "idle_unload_min": cfg.models.idle_unload_min,
                "max_loaded": cfg.models.max_loaded,
                "isolated": cfg.models.isolated,
                "disabled_categories": cfg.models.disabled_categories,
                "available_providers": available_providers(),
                "running_provider": ctx.engine.provider,
                "model_root": str(ctx.engine.model_root),
                "loaded": ctx.engine.loaded_models(),
            },
            "queue": {
                "max_pending": cfg.queue.max_pending,
                "history_days": cfg.queue.history_days,
            },
            "hf": {"endpoint": cfg.hf.endpoint, "has_token": bool(cfg.hf_token())},
            "monbooru": {
                "api_url": cfg.monbooru.api_url,
                "web_url": cfg.monbooru.web_url,
                "paired": bool(ctx.integration and ctx.integration.paired),
                "waiting": bool(ctx.integration and ctx.integration.waiting),
                "push_tags": cfg.monbooru.push_tags,
                "push_images": cfg.monbooru.push_images,
                "gallery": cfg.monbooru.gallery,
            },
            "log": {"debug": cfg.log.debug},
        }

    @app.post("/api/v1/settings")
    async def settings_post(request: Request, body: SettingsBody):
        require_auth(request)
        before = ctx.cfg.current()
        ep_keys = ("execution_provider", "device_id", "intra_op_threads")
        path_before = before.models.path
        ep_before = tuple(getattr(before.models, k) for k in ep_keys)

        def mutate(config):
            if body.thresholds is not None:
                config.thresholds = normalize_thresholds(dict(body.thresholds))
            for section_name in ("server", "auth", "models", "queue", "hf", "monbooru", "log"):
                update = getattr(body, section_name)
                if update is None:
                    continue
                # An explicitly sent null means untouched, an empty string clears.
                values = {
                    key: value
                    for key, value in update.model_dump(exclude_unset=True).items()
                    if value is not None
                }
                section = getattr(config, section_name)
                secret = "password" if section_name == "auth" else "token"
                if secret in values:
                    setattr(section, secret, values.pop(secret))
                for key, value in values.items():
                    setattr(section, key, value)

        try:
            ctx.cfg.update(mutate)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

        after = ctx.cfg.current()
        if before.auth.password != after.auth.password:
            ctx.sessions.clear()
        if before.log.debug != after.log.debug:
            logx.set_debug(after.log.debug)
        if (
            tuple(getattr(after.models, k) for k in ep_keys) != ep_before
            or after.models.path != path_before
            or after.models.isolated != before.models.isolated
        ):
            ctx.engine.reload_sessions()
            log.info("engine reloaded (provider settings changed)")
        if (
            ctx.integration is not None
            and before.monbooru.api_url != after.monbooru.api_url
        ):
            ctx.integration.kick()
        return {"ok": True}

    # -- monbooru integration ------------------------------------------------------

    @app.post("/api/v1/relay/tag")
    async def relay_tag(request: Request):
        """The paired buttons' landing zone. monbooru gives a relay 10s, so
        this only enqueues and answers; results reach it via enrich."""
        integration = ctx.integration
        if integration is None or not integration.request_from_monbooru(request):
            raise HTTPException(status_code=401, detail="unauthorized")
        try:
            body = await request.json()
        except Exception:
            return {"ok": False, "message": "could not read the request"}
        ids = [
            int(i) for i in (body.get("image_ids") or []) if isinstance(i, (int, float))
        ]
        if not ids:
            return {"ok": False, "message": "no images in scope"}
        if not integration.paired:
            return {"ok": False, "message": "not paired yet"}
        count = integration.schedule_retag(ids)
        return {
            "ok": True,
            "message": f"montagger: {count} image(s) queued for AI tagging",
            "refresh": False,
        }

    @app.post("/api/v1/pair/remove")
    async def pair_remove(request: Request):
        """monbooru's teardown: it revoked the token, so drop our copy and
        offer again straight away."""
        integration = ctx.integration
        if integration is None or not integration.request_from_monbooru(request):
            raise HTTPException(status_code=401, detail="unauthorized")
        integration.forget()
        integration.kick()
        return {"status": "removed"}

    @app.post("/api/v1/monbooru/pair")
    async def monbooru_pair(request: Request, body: PairBody):
        require_auth(request)
        if ctx.integration is None:
            raise HTTPException(
                status_code=400, detail="monbooru integration unavailable"
            )
        api_url = body.api_url.strip().rstrip("/")
        if not api_url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=422, detail="set the monbooru API url first"
            )
        ctx.cfg.update(lambda c: setattr(c.monbooru, "api_url", api_url))
        ctx.integration.kick()
        return {"ok": True}

    @app.get("/api/v1/monbooru/status")
    async def monbooru_status(request: Request):
        require_auth(request)
        integration = ctx.integration
        cfg = ctx.cfg.current()
        return {
            "api_url": cfg.monbooru.api_url,
            "paired": bool(integration and integration.paired),
            "waiting": bool(integration and integration.waiting),
            "push_tags": cfg.monbooru.push_tags,
            "push_images": cfg.monbooru.push_images,
            "gallery": cfg.monbooru.gallery,
        }

    @app.post("/api/v1/monbooru/unpair")
    async def monbooru_unpair(request: Request):
        require_auth(request)
        integration = ctx.integration
        if integration is None:
            raise HTTPException(
                status_code=400, detail="monbooru integration unavailable"
            )
        if integration.paired:
            from .pair import MonbooruClient

            await asyncio.to_thread(MonbooruClient(ctx.cfg).pair_remove)
        integration.forget()
        # Clear api_url too: the offer loop re-pairs on sight of a configured
        # api_url, so leaving it set would undo the unpair within seconds.
        ctx.cfg.update(lambda c: setattr(c.monbooru, "api_url", ""))
        integration.kick()
        return {"ok": True}

    @app.get("/api/v1/monbooru/galleries")
    async def monbooru_galleries(request: Request):
        require_auth(request)
        if ctx.integration is None or not ctx.integration.paired:
            return {"galleries": []}
        from .pair import MonbooruClient

        try:
            galleries = await asyncio.to_thread(MonbooruClient(ctx.cfg).galleries)
        except Exception:
            galleries = []
        return {"galleries": galleries}

    # Wire the auto-push hook once, here, where the integration is known.
    if ctx.integration is not None:

        async def _auto_push(job):
            cfg = ctx.cfg.current()
            if job.source == "monbooru" or job.result is None or not job.data:
                return
            if not (cfg.monbooru.push_tags or cfg.monbooru.push_images):
                return
            if not ctx.integration.paired:
                return
            from .pair import MonbooruClient, wire_tags

            client = MonbooruClient(ctx.cfg)
            # A post for this file may already exist (an earlier model of
            # the same upload pushed it): later models enrich that post
            # under their own source instead of creating duplicates.
            prior = ctx.store.find_pushed_by_sha(job.sha256, job.id)
            if prior is not None and prior.monbooru_id:
                if not cfg.monbooru.push_tags:
                    return
                image_id = prior.monbooru_id
                await asyncio.to_thread(
                    client.enrich,
                    image_id,
                    wire_tags(job.result.tags),
                    job.result.model,
                )
                ctx.store.set_pushed(job.id, image_id)
                log.info("enriched monbooru #%s from %s", image_id, job.result.model)
                return
            if not cfg.monbooru.push_images:
                log.info(
                    "kept tags for %s local: 推送图片 is off and the image is not on monbooru",
                    job.filename,
                )
                return
            image_id = await asyncio.to_thread(
                client.push_image,
                job.data,
                job.filename,
                wire_tags(job.result.tags) if cfg.monbooru.push_tags else [],
                cfg.monbooru.gallery,
                job.result.model,
            )
            ctx.store.set_pushed(job.id, image_id)
            log.info(
                "pushed %s to monbooru as #%s (%s)",
                job.filename, image_id, job.result.model,
            )

        ctx.runner.on_done = _auto_push

    # -- SPA hosting -----------------------------------------------------------------

    dist = Path(__file__).resolve().parent.parent / "web" / "dist"

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="no such endpoint")
        candidate = (dist / full_path).resolve()
        if (
            dist.exists()
            and candidate.is_file()
            and str(candidate).startswith(str(dist.resolve()))
        ):
            return FileResponse(candidate)
        index = dist / "index.html"
        if index.exists():
            return FileResponse(index)
        return HTMLResponse(
            "<html><body style='font-family:sans-serif;background:#16161c;color:#e2e2e8;padding:4rem'>"
            "<h1>montagger API is up</h1>"
            "<p>The web UI is not built yet. Run <code>make ui-install ui</code> in the repo, then reload.</p>"
            "<p>API docs: <a style='color:#7986cb' href='/docs'>/docs</a></p></body></html>",
            status_code=200,
        )

    # -- helpers ----------------------------------------------------------------------

    def _live_overlay(job_id: str) -> dict | None:
        try:
            return ctx.runner.get(job_id).public()
        except UnknownJob:
            return None

    def _row_dict(row: store.JobRow, live: dict | None = None) -> dict:
        if live is not None:
            base = _row_dict(row)
            base.update({k: v for k, v in live.items() if k not in ("created_at",)})
            return base
        return {
            "id": row.id,
            "filename": row.filename,
            "source": row.source,
            "model": row.model,
            "provider": row.provider,
            "status": row.status,
            "error": row.error,
            "sha256": row.sha256,
            "md5": row.md5,
            "bytes": row.bytes_len,
            "width": row.width,
            "height": row.height,
            "elapsed_ms": row.elapsed_ms,
            "tags": json.loads(row.tags_json or "[]"),
            "rating": row.rating,
            "monbooru_id": row.monbooru_id,
            "created_at": _iso(row.created_at),
            "finished_at": _iso(row.finished_at),
            "pushed_at": _iso(row.pushed_at),
        }

    def _row_response(row: store.JobRow, dedup: bool = False) -> JSONResponse:
        body = _row_dict(row)
        body["dedup"] = dedup
        return JSONResponse(body, status_code=200)

    return app


def _iso(value) -> str | None:
    return value.isoformat(timespec="seconds") if value else None
