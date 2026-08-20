"""Pairing against a fake monbooru (stdlib http.server).

Pairing is manual (like monloader): start() never offers, the operator
calls connect(), and a poll thread waits for approval.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from montagger.pairing import Pairing

PAIRED_TOKEN = "issued-token-123"


def _wait_for(pred, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


class FakeMonbooru(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[tuple[str, str]] = []  # (method, path)
    approve_after = 1
    deny = False
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
            elif self.path == "/api/v1/pair/remove":
                self._reply(200, {"status": "removed"})
            else:
                self._reply(404, {"error": "not found"})

    def do_GET(self):  # noqa: N802
        with self._lock:
            FakeMonbooru.requests.append(("GET", self.path))
            if self.path.startswith("/api/v1/pair/status"):
                if FakeMonbooru.deny:
                    self._reply(200, {"status": "denied"})
                    return
                self._poll += 1
                if self._poll >= FakeMonbooru.approve_after * 2:
                    self._reply(200, {"status": "approved", "token": PAIRED_TOKEN})
                else:
                    self._reply(200, {"status": "pending"})
            else:
                self._reply(404, {"error": "not found"})


@pytest.fixture
def monbooru_port() -> int:
    FakeMonbooru.requests = []
    FakeMonbooru._poll = 0
    FakeMonbooru.approve_after = 1
    FakeMonbooru.deny = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeMonbooru)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _make(port: int, tmp_path: pytest.TempPathFactory) -> Pairing:
    return Pairing(monbooru_url=_url(port), self_url="http://127.0.0.1:9999", state_dir=tmp_path)


def test_start_never_offers(monbooru_port: int, tmp_path: pytest.TempPathFactory) -> None:
    pairing = _make(monbooru_port, tmp_path)
    pairing.start()
    time.sleep(0.3)
    assert FakeMonbooru.requests == []  # no reach-out on its own
    assert not pairing.paired()
    assert not pairing.waiting()


def test_manual_connect_and_stored_credentials(monbooru_port: int, tmp_path: pytest.TempPathFactory) -> None:
    pairing = _make(monbooru_port, tmp_path)
    pairing.start()
    assert pairing.waiting() is False
    pairing.connect()
    assert pairing.waiting()  # in flight until approved
    # Steady state: credentials are stored before the attempt is cleared
    # (so the panel never flashes back to unpaired), so assert the pair.
    assert _wait_for(lambda: pairing.paired() and not pairing.waiting())
    assert pairing.token() == PAIRED_TOKEN
    assert not pairing.waiting()
    assert (tmp_path / "credentials.json").exists()

    # a fresh instance restores the stored credentials without offering
    restored = _make(monbooru_port, tmp_path)
    assert restored.paired()
    assert restored.token() == PAIRED_TOKEN
    offers = [m for m, p in FakeMonbooru.requests if p == "/api/v1/pair/request"]
    assert len(offers) == 1


def test_connect_while_unreachable_sets_message(tmp_path: pytest.TempPathFactory) -> None:
    pairing = Pairing(monbooru_url="http://127.0.0.1:9", self_url="http://x", state_dir=tmp_path)
    pairing.start()
    pairing.connect()
    assert not pairing.waiting()
    assert "could not reach monbooru" in pairing.state_message()


def test_denied_clears_attempt(monbooru_port: int, tmp_path: pytest.TempPathFactory) -> None:
    FakeMonbooru.deny = True
    pairing = _make(monbooru_port, tmp_path)
    pairing.start()
    pairing.connect()
    assert _wait_for(lambda: not pairing.waiting())
    assert not pairing.paired()
    assert "denied" in pairing.state_message()


def test_cancel_clears_attempt(monbooru_port: int, tmp_path: pytest.TempPathFactory) -> None:
    pairing = _make(monbooru_port, tmp_path)
    pairing.start()
    pairing.connect()
    assert pairing.waiting()
    pairing.cancel()
    assert not pairing.waiting()
    time.sleep(0.3)
    assert not pairing.paired()  # the poll thread must not commit afterwards


def test_challenge_forgets_without_reoffering(monbooru_port: int, tmp_path: pytest.TempPathFactory) -> None:
    pairing = _make(monbooru_port, tmp_path)
    pairing.start()
    pairing.connect()
    assert _wait_for(lambda: pairing.paired())
    offers_before = len([m for m, p in FakeMonbooru.requests if p == "/api/v1/pair/request"])

    pairing.challenged()
    assert not pairing.paired()
    time.sleep(0.3)  # no silent re-offer
    offers_after = len([m for m, p in FakeMonbooru.requests if p == "/api/v1/pair/request"])
    assert offers_after == offers_before


def test_remove_tears_down_both_ends(monbooru_port: int, tmp_path: pytest.TempPathFactory) -> None:
    pairing = _make(monbooru_port, tmp_path)
    pairing.start()
    pairing.connect()
    assert _wait_for(lambda: pairing.paired())
    pairing.remove()
    assert not pairing.paired()
    assert ("POST", "/api/v1/pair/remove") in FakeMonbooru.requests  # notified monbooru
    assert not (tmp_path / "credentials.json").exists()


def test_is_authentic(monbooru_port: int, tmp_path: pytest.TempPathFactory) -> None:
    pairing = _make(monbooru_port, tmp_path)
    pairing.start()
    pairing.connect()
    assert _wait_for(lambda: pairing.paired())
    assert pairing.is_authentic(pairing.peer())
    assert not pairing.is_authentic("wrong")