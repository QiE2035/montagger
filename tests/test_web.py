"""Web layer: auth, pages, result paging, settings write-back, SSE smoke."""

from __future__ import annotations

import threading
import time
import sys
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from montagger import __version__
from montagger.settings import Settings
from montagger.store import Store
from montagger.web.app import create_app


@pytest.fixture
def app(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    config = tmp_path / "montagger.toml"
    config.write_text("webui_token = \"tok\"\nwindow = 16\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "montagger",
            "--config", str(config),
            "--state", str(tmp_path),
        ],
    )
    settings = Settings()
    with TestClient(create_app(settings)) as client:
        yield client


def test_health_public(app: TestClient) -> None:
    resp = app.get("/health")
    assert resp.status_code == 200
    assert resp.json()["version"] == __version__
    assert resp.json()["paired"] is False


def test_dashboard_requires_auth(app: TestClient) -> None:
    resp = app.get("/")
    assert resp.status_code == 401
    assert "sign in" in resp.text  # the login page, not a bare 401
    resp = app.get("/", headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 200
    assert "montagger" in resp.text


def test_login_flow(app: TestClient) -> None:
    # wrong token stays on the login page
    resp = app.post("/api/login", data={"token": "wrong"})
    assert resp.status_code == 401
    assert "wrong token" in resp.text
    assert app.get("/").status_code == 401

    # right token issues a session cookie and unlocks the pages
    resp = app.post("/api/login", data={"token": "tok"}, follow_redirects=False)
    assert resp.status_code == 303
    assert "montagger_session" in resp.headers["set-cookie"]
    assert app.get("/").status_code == 200
    assert app.get("/api/results?page=1&filter=all").status_code == 200


def test_relay_requires_peer(app: TestClient) -> None:
    resp = app.post("/relay/tag", json={"image_ids": [1, 2]}, headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 401  # webui token is not the pairing peer secret
    resp = app.post("/relay/tag", json={"image_ids": [1, 2]})
    assert resp.status_code == 401


def test_results_paging(app: TestClient) -> None:
    store: Store = app.app.state.store
    store.submit("http://127.0.0.1:9", list(range(1, 61)))
    for i in range(1, 61):
        store.mark_done("http://127.0.0.1:9", i, [f"tag{i}"])

    resp = app.get("/api/results?page=1&filter=all", headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 200
    assert "60 result(s)" in resp.text
    assert "page 1 / 2" in resp.text

    resp = app.get("/api/results?page=2&filter=all", headers={"Authorization": "Bearer tok"})
    assert "page 2 / 2" in resp.text


def test_settings_write_back(app: TestClient, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    resp = app.post(
        "/api/settings",
        data={"window": "9", "threshold": "0.5"},
        headers={"Authorization": "Bearer tok", "X-Montagger": app.app.state.csrf},
    )
    assert resp.status_code == 200
    time.sleep(0.1)
    raw = (tmp_path / "montagger.toml").read_text(encoding="utf-8")
    assert "window = 9" in raw
    assert "threshold = 0.5" in raw


def test_settings_hot_apply(app: TestClient, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Everything hot-editable takes effect immediately, not on restart."""
    monkeypatch.chdir(tmp_path)
    headers = {"Authorization": "Bearer tok", "X-Montagger": app.app.state.csrf}
    resp = app.post(
        "/api/settings",
        data={
            "_section": "tagging",
            "via": "webui",
        },
        headers=headers,
    )
    assert resp.status_code == 200
    assert app.app.state.pipeline.via == "webui"

    resp = app.post("/api/settings", data={"webui_token": "newtok"}, headers=headers)
    assert resp.status_code == 200
    assert app.get("/", headers={"Authorization": "Bearer newtok"}).status_code == 200
    assert app.get("/", headers={"Authorization": "Bearer tok"}).status_code == 401


def test_settings_page_shows_sources(app: TestClient) -> None:
    resp = app.get("/settings", headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 200
    assert "execution provider" in resp.text
    assert "restart required" in resp.text


def test_pairing_is_manual(app: TestClient) -> None:
    # the settings page offers the connect button while unpaired
    resp = app.get("/settings", headers={"Authorization": "Bearer tok"})
    assert "To pair with a monbooru" in resp.text
    assert 'name="url"' in resp.text

    # connect requires the csrf header like every state change
    resp = app.post("/api/pair/connect", headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 403
    headers = {"Authorization": "Bearer tok", "X-Montagger": app.app.state.csrf}
    resp = app.post("/api/pair/connect", headers=headers)
    assert resp.status_code == 200
    assert "enter the monbooru address" in resp.text


class _FakeMonbooru(BaseHTTPRequestHandler):
    """Approves every pairing instantly; serves images and tag writes."""

    protocol_version = "HTTP/1.1"
    peer: str = ""

    def log_message(self, *args):  # silence
        pass

    def _reply(self, code: int, body=None) -> None:
        data = json.dumps(body).encode() if body is not None else b""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        if self.path == "/api/v1/pair/request":
            self.server.peer_offers += 1
            self._reply(200, {"request_id": f"req-{self.server.peer_offers}"})
        elif self.path == "/api/v1/pair/remove":
            self._reply(200, {"status": "removed"})
        elif self.path == "/api/v1/images/1/tags":
            self._reply(200, {"status": "ok"})
        else:
            self._reply(404, {"error": "no"})

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/v1/pair/status"):
            self._reply(200, {"status": "approved", "token": "issued-tok"})
        else:
            self._reply(404, {"error": "no"})


@pytest.fixture
def paired_app(app: TestClient, tmp_path: pytest.TempPathFactory):
    """An app with two live pairings (url + issued token stored in TOML)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeMonbooru)
    server.peer_offers = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    headers = {"Authorization": "Bearer tok", "X-Montagger": app.app.state.csrf}
    resp = app.post("/api/pair/connect", data={"url": url}, headers=headers)
    assert resp.status_code == 200
    # pairing completes in a poll thread
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and not app.app.state.pairing.entry_for_url(url):
        time.sleep(0.05)
    entry = app.app.state.pairing.entry_for_url(url)
    assert entry is not None
    assert _wait_for(lambda: entry.complete)
    yield app, url, entry
    server.shutdown()
    server.server_close()


def _wait_for(pred, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_relay_routes_by_peer(paired_app: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relay click from a paired monbooru is tagged back against that same
    instance - the peer secret recovers the source url."""
    app, url, entry = paired_app
    # pause so the worker cannot race past the assertions
    app.app.state.pipeline.pause()
    resp = app.post(
        "/relay/tag",
        json={"payload": 1, "monbooru": "v1", "image_ids": [1]},
        headers={"Authorization": f"Bearer {entry.peer}"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    rows, _ = app.app.state.store.results(1, 50, "pending")
    assert rows[0]["source"] == url
    assert rows[0]["image_id"] == 1

    # an unknown secret is refused
    resp = app.post(
        "/relay/tag",
        json={"image_ids": [2]},
        headers={"Authorization": "Bearer unknown"},
    )
    assert resp.status_code == 401


def test_sse_requires_token(app: TestClient) -> None:
    # Real SSE streaming (frames every second) is verified manually against a
    # live uvicorn; TestClient does not play well with streaming responses.
    assert app.get("/api/stream").status_code == 401
    assert app.get("/api/stream?token=wrong").status_code == 401


def test_actions_accept_htmx(app: TestClient) -> None:
    headers = {"Authorization": "Bearer tok", "X-Montagger": app.app.state.csrf}
    resp = app.post("/api/retry", headers=headers)
    assert resp.status_code == 204
    resp = app.post("/api/pause", headers=headers)
    assert resp.status_code == 204
    resp = app.post("/api/resume", headers=headers)
    assert resp.status_code == 204