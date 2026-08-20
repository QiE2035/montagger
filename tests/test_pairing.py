"""Pairing against fake monbooru instances (stdlib http.server).

Pairing is manual and one-to-many (like monloader): start() never offers,
the operator calls connect(url) per instance, a poll thread waits for
approval, and credentials are persisted in montagger.toml's [[pairing]]
array instead of a sidecar file.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from montagger.pairing import Pairing

POLL = 2.0


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
    shared_lock = threading.Lock()
    token_seq = 0

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
        with FakeMonbooru.shared_lock:
            FakeMonbooru.requests.append(("POST", self.path))
        if self.path == "/api/v1/pair/request":
            self._reply(200, {"request_id": "req-" + str(self.server.server_address[1])})
        elif self.path == "/api/v1/pair/remove":
            self._reply(200, {"status": "removed"})
        else:
            self._reply(404, {"error": "not found"})

    def do_GET(self):  # noqa: N802
        with FakeMonbooru.shared_lock:
            FakeMonbooru.requests.append(("GET", self.path))
        if self.path.startswith("/api/v1/pair/status"):
            FakeMonbooru.token_seq += 1
            self._reply(200, {"status": "approved", "token": f"tok-{self.server.server_address[1]}-{FakeMonbooru.token_seq}"})
        else:
            self._reply(404, {"error": "not found"})


@pytest.fixture
def monbooru() -> list[int]:
    FakeMonbooru.requests = []
    FakeMonbooru.token_seq = 0
    servers: list[ThreadingHTTPServer] = []
    ports: list[int] = []
    for _ in range(2):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeMonbooru)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        ports.append(server.server_address[1])
    yield ports
    for server in servers:
        server.shutdown()
        server.server_close()


def _url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _make(tmp_path: Path) -> Pairing:
    return Pairing(
        self_url="http://127.0.0.1:9999",
        config_path=tmp_path / "montagger.toml",
    )


def test_start_never_offers(monbooru: list[int], tmp_path: pytest.TempPathFactory) -> None:
    pairing = _make(tmp_path)
    pairing.start()
    time.sleep(0.3)
    assert FakeMonbooru.requests == []  # no reach-out on its own
    assert not pairing.paired()
    assert not pairing.waiting()


def test_connect_appends_and_persists(monbooru: list[int], tmp_path: pytest.TempPathFactory) -> None:
    a, b = _url(monbooru[0]), _url(monbooru[1])
    pairing = _make(tmp_path)
    pairing.start()
    pairing.connect(a)
    assert pairing.waiting() and pairing.waiting_url() == a
    # Steady state: credentials stored before the attempt is cleared.
    assert _wait_for(lambda: pairing.paired() and not pairing.waiting())
    assert pairing.entry_for_url(a).token.startswith("tok-")

    # A second instance pairs independently.
    pairing.connect(b)
    assert _wait_for(lambda: len([e for e in pairing.entries() if e.complete]) == 2)
    entry_b = pairing.entry_for_url(b)
    assert entry_b.complete
    assert pairing.entry_for_url(a).token != entry_b.token

    raw = (tmp_path / "montagger.toml").read_text(encoding="utf-8")
    assert "[[pairing]]" in raw
    assert a in raw and b in raw
    assert "token" in raw

    # TOML survives a restart: a fresh instance restores both pairings.
    restored = _make(tmp_path)
    restored.start()
    assert restored.entry_for_url(a).complete
    assert restored.entry_for_url(b).complete
    offers = [p for m, p in FakeMonbooru.requests if p == "/api/v1/pair/request"]
    assert len(offers) == 2


def test_peer_recovery_and_auth(monbooru: list[int], tmp_path: pytest.TempPathFactory) -> None:
    a, b = _url(monbooru[0]), _url(monbooru[1])
    pairing = _make(tmp_path)
    pairing.start()
    pairing.connect(a)
    assert _wait_for(lambda: pairing.entry_for_url(a) is not None, timeout=10)
    entry_a = pairing.entry_for_url(a)
    assert _wait_for(lambda: entry_a.complete)
    pairing.connect(b)
    assert _wait_for(lambda: pairing.entry_for_url(b) is not None and pairing.entry_for_url(b).complete)

    assert pairing.is_authentic(entry_a.peer)
    assert pairing.is_authentic(pairing.entry_for_url(b).peer)
    assert not pairing.is_authentic("nope")
    # each peer recovers its own instance
    assert pairing.entry_for_peer(entry_a.peer).url == a
    assert pairing.entry_for_peer(pairing.entry_for_url(b).peer).url == b
    assert pairing.token_for(a) == entry_a.token


def test_connect_while_unreachable_sets_message(tmp_path: pytest.TempPathFactory) -> None:
    pairing = _make(tmp_path)
    pairing.start()
    pairing.connect("http://127.0.0.1:9")
    assert not pairing.waiting()
    assert "could not reach monbooru" in pairing.state_message()


def test_challenged_drops_only_that_instance(monbooru: list[int], tmp_path: pytest.TempPathFactory) -> None:
    a, b = _url(monbooru[0]), _url(monbooru[1])
    pairing = _make(tmp_path)
    pairing.start()
    pairing.connect(a)
    assert _wait_for(lambda: pairing.entry_for_url(a) is not None and pairing.entry_for_url(a).complete)
    pairing.connect(b)
    assert _wait_for(lambda: pairing.entry_for_url(b) is not None and pairing.entry_for_url(b).complete)

    pairing.challenged(a)
    assert pairing.entry_for_url(a) is None
    assert pairing.entry_for_url(b).complete
    assert "rejected" in pairing.state_message()


def test_remove_and_unpair(monbooru: list[int], tmp_path: pytest.TempPathFactory) -> None:
    a, b = _url(monbooru[0]), _url(monbooru[1])
    pairing = _make(tmp_path)
    pairing.start()
    pairing.connect(a)
    assert _wait_for(lambda: pairing.entry_for_url(a) is not None and pairing.entry_for_url(a).complete)

    # operator-side remove asks the instance to drop its side too
    pairing.remove(a)
    assert pairing.entry_for_url(a) is None
    raw = (tmp_path / "montagger.toml").read_text(encoding="utf-8")
    assert a not in raw

    # monbooru-side removal arrives with the peer secret
    pairing.connect(b)
    assert _wait_for(lambda: pairing.entry_for_url(b) is not None and pairing.entry_for_url(b).complete)
    pairing.unpair(pairing.entry_for_url(b).peer)
    assert pairing.entry_for_url(b) is None