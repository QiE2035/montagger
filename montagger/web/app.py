"""The FastAPI application: plugin contract endpoints, WebUI pages, SSE.

Lifespan assembles store -> client -> backend -> pipeline -> pairing and
tears them down gracefully (drain the pipeline, then close everything).
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from montagger import __version__
from montagger.backends import get_backend
from montagger.client import MonbooruClient
from montagger.pairing import Pairing
from montagger.pipeline import Pipeline
from montagger.settings import BUTTONS, EP_ALIASES, RuntimeState, Settings
from montagger.store import STATUSES, Store
from montagger.tagging import RelayPayload, relay_answer
from montagger.web.auth import require_auth, require_peer, require_stream
from montagger.web.events import EventBus

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATE_DIR = WEB_DIR / "templates"

RESULT_PAGE_SIZE = 50
LIVE_MAX = 24  # recently-finished rows pushed over SSE before the table


class SettingsPatch(BaseModel):
    ep: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    character_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    window: int | None = Field(default=None, ge=1, le=1024)
    prefetch_threads: int | None = Field(default=None, ge=1, le=64)
    workers: int | None = Field(default=None, ge=1, le=64)
    skip_tagged: bool | None = None
    general_topk: int | None = None


def _render(templates: Jinja2Templates, name: str, **ctx: Any) -> str:
    return templates.get_template(name).render(**ctx)


def _plain_sse(kind: str, body: str) -> str:
    return f"event: {kind}\ndata: {body}\n\n"


def create_app(settings: Settings) -> FastAPI:
    runtime = RuntimeState.from_settings(settings)
    bus = EventBus()
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    def _progress_pct(stats: dict[str, int]) -> int:
        finished = stats["done"] + stats["failed"]
        total = stats["total"]
        return int(finished * 100 / total) if total else 0

    templates.env.filters["progress_pct"] = _progress_pct

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        bus.attach(asyncio.get_running_loop())
        state = app.state
        state.runtime = runtime
        state.bus = bus

        # Composed services, torn down in reverse order.
        store = Store(settings.state / "montagger.db")
        state.store = store

        pairing = Pairing(
            monbooru_url=settings.monbooru,
            self_url=settings.self_url,
            state_dir=settings.state,
            on_change=lambda ok: bus.publish("pair", {"paired": ok}),
        )
        state.pairing = pairing
        client = MonbooruClient(
            settings.monbooru,
            get_token=pairing.token,
            on_unauthorized=pairing.challenged,
        )
        state.client = client

        state.backend = get_backend(
            settings.backend,
            runtime,
            {"client": client, "model_dir": settings.model_dir, "activation": settings.activation},
        )

        pipeline = Pipeline(
            runtime,
            store,
            client,
            state.backend,
            via=settings.via,
            publish=bus.publish,
        )
        state.pipeline = pipeline
        if settings.resume:
            resumed = store.resume_ids()
            if resumed:
                pipeline.submit(resumed)
                log.info("resumed %d task(s) from the database", len(resumed))
        pipeline.start()
        pairing.start()

        yield

        log.info("shutting down: draining the pipeline")
        pipeline.stop(drain_timeout=30.0)
        pairing.stop()
        state.backend.close()
        client.close()
        store.close()

    app = FastAPI(title="montagger", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.templates = templates
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ---- plugin contract (monbooru <-> montagger) -----------------------

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {
            "version": __version__,
            "paired": request.app.state.pairing.paired(),
            "backend": settings.backend,
            "ep": runtime.ep,
            "pipeline_paused": request.app.state.pipeline.paused(),
        }

    @app.post("/api/v1/pair/remove", dependencies=[Depends(require_peer)])
    def pair_remove(request: Request) -> JSONResponse:
        request.app.state.pairing.unpair()
        return JSONResponse({"status": "removed"})

    @app.post("/relay/tag", dependencies=[Depends(require_peer)])
    async def relay_tag(request: Request) -> JSONResponse:
        raw = await request.body()
        try:
            payload = RelayPayload.model_validate_json(raw)
        except Exception:
            return JSONResponse({"ok": False, "message": "could not read the request"})
        answer = relay_answer(request.app.state.pipeline, payload)
        return JSONResponse(answer)

    # ---- WebUI pages ----------------------------------------------------

    @app.get("/", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
    def dashboard(request: Request) -> HTMLResponse:
        ctx = _page_ctx(request)
        ctx["sse_token"] = settings.webui_token or request.app.state.pairing.peer()
        rows, _ = request.app.state.store.results(1, LIVE_MAX, None)
        ctx["live"] = _rows_view(rows)
        return HTMLResponse(_render(templates, "index.html", **ctx))

    @app.get("/settings", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
    def settings_page(request: Request) -> HTMLResponse:
        return HTMLResponse(_render(templates, "settings.html", **_page_ctx(request)))

    def _page_ctx(request: Request) -> dict[str, Any]:
        p = request.app.state.pipeline
        return {
            "version": __version__,
            "pairing": request.app.state.pairing,
            "runtime": runtime,
            "settings": settings,
            "pipeline": p,
            "stats": p.stats(),
            "providers": _ep_options(),
            "auth": settings.webui_token or request.app.state.pairing.peer(),
        }

    def _ep_options() -> list[str]:
        try:
            import onnxruntime as ort

            available = set(ort.get_available_providers())
        except Exception:
            available = set()
        return [name for name, provider in EP_ALIASES.items() if provider in available] or ["cpu"]

    # ---- fragments & actions (htmx) -------------------------------------

    @app.get("/api/pair-light", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
    def pair_light(request: Request) -> HTMLResponse:
        return HTMLResponse(_render(templates, "partials/pair_light.html", pairing=request.app.state.pairing))

    @app.get("/api/results", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
    def results(request: Request, page: int = 1, filter: str = "all") -> HTMLResponse:
        status = filter if filter in STATUSES else None
        rows, total = request.app.state.store.results(page, RESULT_PAGE_SIZE, status)
        pages = max(1, -(-total // RESULT_PAGE_SIZE))
        return HTMLResponse(
            _render(
                templates,
                "partials/results_rows.html",
                rows=_rows_view(rows),
                total=total,
                pages=pages,
                page=page,
                filter=filter,
            )
        )

    @app.post("/api/pause", dependencies=[Depends(require_auth)])
    def pause(request: Request) -> Response:
        request.app.state.pipeline.pause()
        return Response(status_code=204)

    @app.post("/api/resume", dependencies=[Depends(require_auth)])
    def resume(request: Request) -> Response:
        request.app.state.pipeline.resume()
        return Response(status_code=204)

    @app.post("/api/retry", dependencies=[Depends(require_auth)])
    def retry(request: Request) -> Response:
        state = request.app.state
        count = state.pipeline.retry_failed()
        state.bus.publish("notice", {"text": f"retrying {count} failed"})
        return Response(status_code=204)

    @app.post("/api/clear-results", dependencies=[Depends(require_auth)])
    def clear_results(request: Request) -> Response:
        count = request.app.state.pipeline.clear_results()
        request.app.state.bus.publish("notice", {"text": f"cleared {count} results"})
        return Response(status_code=204)

    @app.post("/api/clear-tasks", dependencies=[Depends(require_auth)])
    def clear_tasks(request: Request) -> Response:
        count = request.app.state.pipeline.clear_tasks()
        request.app.state.bus.publish("notice", {"text": f"cleared {count} tasks"})
        return Response(status_code=204)

    @app.post("/api/settings", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
    async def patch_settings(request: Request) -> HTMLResponse:
        form = await request.form()
        patch = SettingsPatch(
            **{k: v for k, v in form.items() if k in SettingsPatch.model_fields}
        )
        changed = patch.model_dump(exclude_unset=True, exclude_none=True)
        if not changed:
            return HTMLResponse(_render(templates, "settings.html", **_page_ctx(request)))
        settings.write_back(changed)
        runtime.update(**changed)
        settings.ep = runtime.ep  # keep the mirror accurate for /health
        request.app.state.backend.reload(runtime)
        request.app.state.pipeline.reconfigure()
        request.app.state.bus.publish("notice", {"text": "settings saved"})
        return HTMLResponse(_render(templates, "settings.html", **_page_ctx(request)))

    # ---- SSE ------------------------------------------------------------

    @app.get("/api/stream", dependencies=[Depends(require_stream)])
    def stream(request: Request) -> StreamingResponse:
        p = request.app.state.pipeline
        bus = request.app.state.bus

        def live_rows() -> list[dict[str, Any]]:
            rows, _ = request.app.state.store.results(1, LIVE_MAX, None)
            return _rows_view(rows)

        async def generator() -> AsyncIterator[str]:
            queue = bus.subscribe()
            try:
                while True:
                    status_body = _render(
                        templates,
                        "partials/status_cards.html",
                        stats=p.stats(),
                        runtime=runtime,
                        pairing=request.app.state.pairing,
                        version=__version__,
                    )
                    yield _plain_sse("status", status_body)
                    deadline = asyncio.get_running_loop().time() + 1.0
                    while asyncio.get_running_loop().time() < deadline:
                        try:
                            kind, data = await asyncio.wait_for(queue.get(), timeout=0.1)
                        except (asyncio.TimeoutError, RuntimeError):
                            break
                        if kind == "result":
                            rows = _rows_view([data])
                            fragment = _render(templates, "partials/result_row.html", row=rows[0]) if rows else ""
                            if fragment:
                                yield _plain_sse("result", fragment)
                        elif kind == "pair":
                            yield _plain_sse(
                                "pair",
                                _render(templates, "partials/pair_light.html", pairing=request.app.state.pairing),
                            )
                        elif kind == "notice":
                            yield _plain_sse("notice", json.dumps(data))
            finally:
                bus.unsubscribe(queue)

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _rows_view(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    view: list[dict[str, Any]] = []
    for row in rows:
        try:
            tags = json.loads(row.get("tags") or "[]")
        except json.JSONDecodeError:
            tags = []
        view.append(
            {
                "image_id": row.get("image_id"),
                "status": row.get("status", ""),
                "tags": tags,
                "error": row.get("error", ""),
            }
        )
    return view