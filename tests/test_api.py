"""API surface over a stub engine: upload, dedup, auth, settings, relay guard."""

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from montagger.app import AppContext, create_app
from montagger.config import Provider
from montagger.engine.backend import TagResult
from montagger.engine.scoring import ScoredTag
from montagger.engine.session import EngineError
from montagger.queue import Runner
from montagger.store import Store, new_job_id


class StubEngine:
    """The Engine surface create_app touches, minus onnxruntime."""

    def __init__(self, cfg: Provider, model_root):
        self._cfg = cfg
        self.model_root = model_root
        self.provider = "CPUExecutionProvider"
        self.provider_short = "cpu"

    def reload_sessions(self):
        pass

    def resolve_model_name(self, model=None):
        return model or self._cfg.current().models.default

    def resolve_model_list(self, model_param=None):
        cfg = self._cfg.current()
        if model_param and model_param.strip():
            names = []
            for part in model_param.split(","):
                name = part.strip()
                if name and name not in names:
                    names.append(name)
            if names:
                return names
        names = [m.strip() for m in cfg.models.default_models if m.strip()]
        if names:
            return names
        default = cfg.models.default.strip()
        return [default] if default else []

    def is_available(self, name):
        return "" if name == "stub" else "not installed"

    def loaded_models(self):
        return []

    def child_rss_mb(self):
        return 0.0

    def unload(self, name):
        return False

    def unload_all(self):
        return 0

    def emitted_categories(self, name):
        return ["general", "character", "rating"]

    def statuses(self):
        from montagger.engine.backend import ModelStatus

        return [
            ModelStatus(
                name="stub", description="stub model", discovered=True, available=True,
                default_threshold=0.35, default_thresholds={"character": 0.5},
                default_top_k={"character": 8},
                emitted_categories=["general", "character", "rating"],
            )
        ]

    def thresholds(self, name):
        # Mirror Engine.thresholds' merge (stub has no catalog entry).
        ov = self._cfg.current().threshold_overrides(name)
        g = ov["global"] if ov["global"] is not None else 0.35
        cats = {"character": 0.5}
        cats.update(ov["categories"])
        return g, cats, dict(ov["top_k"]), list(ov["disabled"])

    def tag_bytes(self, data, model=None):
        name = model or "stub"
        reason = self.is_available(name)
        if reason:
            raise EngineError(f"model {name}: {reason}")
        if not data.startswith(b"\x89PNG"):
            raise EngineError("cannot identify image file")
        gate = getattr(self, "gate", None)
        if gate is not None:
            if not gate.wait(timeout=5):
                raise EngineError("test gate timed out")
        return TagResult(
            model=name, provider="cpu", width=2, height=2, elapsed_ms=1,
            tags=[ScoredTag(name="1girl", category="general", confidence=0.9)],
            rating="general",
        )


def png_bytes(size=(2, 2)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    cfg = Provider(tmp_path / "montagger.toml")
    cfg.update(lambda c: setattr(c.models, "default", "stub"))
    engine = StubEngine(cfg, tmp_path / "models")
    store_ = Store(tmp_path / "test.sqlite")
    runner = Runner(engine, store_, 8)
    app = create_app(AppContext(cfg, engine, store_, runner, integration=None))
    with TestClient(app) as c:
        c.ctx = AppContext(cfg, engine, store_, runner)
        yield c


def test_health(client):
    assert "version" in client.get("/health").json()


def test_tag_wait_true(client):
    resp = client.post("/api/v1/tag?wait=true&filename=a.png", content=png_bytes())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done" and body["rating"] == "general"
    assert body["tags"][0]["name"] == "1girl"


def test_tag_garbage_is_error(client):
    resp = client.post("/api/v1/tag?wait=true", content=b"garbage")
    assert resp.status_code == 502
    assert resp.json()["status"] == "error"


def test_tag_empty_body_400(client):
    assert client.post("/api/v1/tag", content=b"").status_code == 400


def test_tag_queued_then_done(client):
    resp = client.post("/api/v1/tag?filename=b.png", content=png_bytes())
    assert resp.status_code == 202
    job_id = resp.json()["id"]
    deadline = time.time() + 5
    while time.time() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] == "done":
            assert body.get("dedup") in (None, False)
            return
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_dedup_reuses_result(client):
    data = png_bytes()
    first = client.post("/api/v1/tag?wait=true", content=data).json()
    second = client.post("/api/v1/tag?wait=true", content=data).json()
    assert second["dedup"] is True
    assert second["tags"] == first["tags"]


def test_models_endpoint(client):
    body = client.get("/api/v1/models").json()
    assert body["provider_short"] == "cpu"
    assert body["models"][0]["name"] == "stub"
    assert body["models"][0]["effective"]["categories"] == {"character": 0.5}
    assert body["models"][0]["effective"]["top_k"] == {}
    assert body["models"][0]["emitted_categories"] == ["general", "character", "rating"]


def test_multi_model_upload(client):
    cfg = client.ctx.cfg
    cfg.update(lambda c: setattr(c.models, "default_models", ["stub"]))
    resp = client.post("/api/v1/tag?wait=true&model=stub,stub2&filename=m.png", content=png_bytes())
    # stub2 is not installed: its job errors, stub's succeeds
    assert resp.status_code == 502
    body = resp.json()
    assert isinstance(body, list) and len(body) == 2
    assert body[0]["model"] == "stub" and body[0]["status"] == "done"
    assert body[1]["model"] == "stub2" and body[1]["status"] == "error"
    # default_models drives the no-param upload shape
    resp = client.post("/api/v1/tag?wait=true&filename=m.png", content=png_bytes())
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list) and [j["model"] for j in body] == ["stub"]


def test_jobs_list_shape_and_pagination(client):
    for i in range(3):
        client.post("/api/v1/tag?wait=true", content=png_bytes())
    body = client.get("/api/v1/jobs?limit=2").json()
    assert set(body) >= {"jobs", "total", "offset", "limit"}
    assert body["total"] == 3 and body["limit"] == 2 and len(body["jobs"]) == 2
    page2 = client.get("/api/v1/jobs?limit=2&offset=2").json()
    assert len(page2["jobs"]) == 1
    found = client.get("/api/v1/jobs?q=1girl").json()
    assert found["total"] == 3  # tags_json matches the stored tag name
    assert client.get("/api/v1/jobs?q=nonexistent-string").json()["total"] == 0


def test_job_cancel_and_batch(client):
    import threading

    engine = client.ctx.engine
    gate = threading.Event()
    engine.gate = gate
    try:
        # First job blocks the single worker inside tag_bytes...
        blocked = client.post("/api/v1/tag?filename=blocked.png", content=png_bytes()).json()
        # ...so a second upload stays queued long enough to cancel.
        resp = client.post("/api/v1/tag?filename=c.png", content=png_bytes())
        job_id = resp.json()["id"]
        r = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert r.status_code == 200
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "canceled"
        assert client.post(f"/api/v1/jobs/{job_id}/cancel").status_code == 409
        gate.set()
        assert client.get(f"/api/v1/jobs/{blocked['id']}").json()["status"] == "done"
    finally:
        gate.set()
        engine.gate = None
    # batch delete of all done rows
    client.post("/api/v1/tag?wait=true", content=png_bytes())
    r = client.post("/api/v1/jobs/batch", json={"action": "delete", "all_done": True})
    assert r.json()["affected"] >= 1
    # batch cancel on a blocked worker leaves fresh jobs canceled
    gate2 = threading.Event()
    engine.gate = gate2
    try:
        held = client.post("/api/v1/tag?filename=held.png", content=png_bytes()).json()
        fresh = client.post("/api/v1/tag?filename=fresh.png", content=png_bytes()).json()
        r = client.post("/api/v1/jobs/batch", json={"action": "cancel", "all_queued": True})
        assert r.json()["affected"] == 1
        assert client.get(f"/api/v1/jobs/{fresh['id']}").json()["status"] == "canceled"
        gate2.set()
        assert client.get(f"/api/v1/jobs/{held['id']}").json()["status"] == "done"
    finally:
        gate2.set()
        engine.gate = None


def test_memory_release_and_history_purge(client):
    r = client.post("/api/v1/memory/release", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and "rss_mb" in body and "jobs" in body
    r = client.post("/api/v1/history/purge")
    assert r.status_code == 200 and "deleted" in r.json()


def test_password_guard(client):
    # open LAN by default
    assert client.get("/api/v1/jobs").status_code == 200
    # set a password: everything locks
    client.post("/api/v1/settings", json={"auth": {"password": "pw"}})
    assert client.get("/api/v1/jobs").status_code == 401
    state = client.get("/api/v1/auth/state").json()
    assert state["guard_active"] and not state["authenticated"]
    # wrong password refused, right one sets the cookie
    assert client.post("/api/v1/auth/login", json={"password": "nope"}).status_code == 401
    ok = client.post("/api/v1/auth/login", json={"password": "pw"})
    assert ok.status_code == 200
    assert client.get("/api/v1/jobs").status_code == 200
    # bearer token also passes the guard
    token = client.post("/api/v1/tokens", json={"name": "t"}).json()["token"]
    fresh = TestClient(client.app)
    fresh.headers.update({"Authorization": f"Bearer {token}"})
    assert fresh.get("/api/v1/jobs").status_code == 200


def test_settings_validation(client):
    resp = client.post("/api/v1/settings", json={"models": {"max_upload_mb": 0}})
    assert resp.status_code == 422
    # nested threshold form accepted and reflected in /api/v1/models
    ok = client.post(
        "/api/v1/settings",
        json={"thresholds": {"stub": {"global": 0.4, "categories": {"character": 0.6},
                                       "top_k": {"character": 5}, "disabled": ["rating"]}}},
    )
    assert ok.status_code == 200
    eff = client.get("/api/v1/models").json()["models"][0]["effective"]
    assert eff["global"] == 0.4 and eff["categories"] == {"character": 0.6}
    assert eff["top_k"] == {"character": 5} and eff["disabled"] == ["rating"]
    assert client.post(
        "/api/v1/settings", json={"thresholds": {"stub": {"global": 5}}}
    ).status_code == 422
    # default_models + push switch validation
    assert client.post(
        "/api/v1/settings", json={"models": {"default_models": ["stub"]}}
    ).status_code == 200
    assert client.post(
        "/api/v1/settings", json={"models": {"default_models": [""]}}
    ).status_code == 422
    body = client.get("/api/v1/settings").json()
    assert body["monbooru"]["push_tags"] is True  # default: tags yes
    assert body["monbooru"]["push_images"] is False  # default: images no


def test_relay_requires_peer_secret(client):
    resp = client.post("/api/v1/relay/tag", json={"image_ids": [1]})
    assert resp.status_code == 401


def test_unknown_model_rejected(client):
    resp = client.post("/api/v1/tag?wait=true&model=joytag", content=png_bytes())
    assert resp.status_code == 502  # is_available reason becomes the error
    body = resp.json()
    assert body["status"] == "error" and "not installed" in body["error"]
