"""Pairing against a fake monbooru (stdlib http.server)."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from montagger.pairing import Pairing

PAIRED_TOKEN = "issued-token-123"


class FakeMonbooru(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[tuple[str, str]] = []  # (method, path)
    approve_after = 1
    _poll = 0
    _lock = threading.Lock()

    def log_message(self, *args):  # silence
        pass

    def _reply(self, code: int, body: dict | None = None) -> None:
        data = json.dumps(body).encode() if body is not None else b""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw or b"{}")
        with self._lock:
            FakeMonbooru.requests.append(("POST", self.path))
            if self.path == "/api/v1/pair/request":
                request_id = body.get("peer_token", "x") + "-req"
                self._reply(200, {"request_id": request_id})
            else:
                self._reply(404, {"error": "not found"})

    def do_GET(self):  # noqa: N802
        with self._lock:
            FakeMonbooru.requests.append(("GET", self.path))
            if self.path == "/api/v1/galleries":
                self._auth_ok()
            elif self.path.startswith("/api/v1/pair/status"):
                self._poll += 1
                if self._poll >= FakeMonbooru.approve_after * 2:
                    self._reply(200, {"status": "approved", "token": PAIRED_TOKEN})
                else:
                    self._reply(200, {"status": "pending"})
            else:
                self._reply(404, {"error": "not found"})

    def _auth_ok(self) -> None:
        auth = self.headers.get("Authorization", "")
        self._reply(200, [] if auth else {"error": "unauthorized"})


@pytest.fixture
def monbooru_port() -> int:
    FakeMonbooru.requests = []
    FakeMonbooru._poll = 0
    FakeMonbooru.approve_after = 1
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeMonbooru)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def test_pair_approval_and_stored_credentials(monbooru_port: int, tmp_path: pytest.TempPathFactory) -> None:
    pairing = Pairing(monbooru_url=_url(monbooru_port), self_url="http://127.0.0.1:9999", state_dir=tmp_path)
    pairing.start()
    deadline = time.monotonic() + 8
    while not pairing.paired() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pairing.paired()
    assert pairing.token() == PAIRED_TOKEN
    assert (tmp_path / "credentials.json").exists()

    # a fresh instance restores the stored credentials
    restored = Pairing(monbooru_url=_url(monbooru_port), self_url="http://127.0.0.1:9999", state_dir=tmp_path)
    assert restored.token() == PAIRED_TOKEN


def test_is_authentic(monbooru_port: int, tmp_path: pytest.TempPathFactory) -> None:
    pairing = Pairing(monbooru_url=_url(monbooru_port), self_url="http://127.0.0.1:9999", state_dir=tmp_path)
    pairing.start()
    deadline = time.monotonic() + 8
    while not pairing.paired() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pairing.is_authentic(pairing.peer())
    assert not pairing.is_authentic("wrong")


def test_401_challenge_reoffers(monbooru_port: int, tmp_path: pytest.TempPathFactory) -> None:
    pairing = Pairing(monbooru_url=_url(monbooru_port), self_url="http://127.0.0.1:9999", state_dir=tmp_path)
    pairing.start()
    deadline = time.monotonic() + 8
    while not pairing.paired() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pairing.paired()

    pairing.forget()
    assert not pairing.paired()
    pairing.challenged()
    deadline = time.monotonic() + 8
    while not pairing.paired() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pairing.paired()