"""The isolated inference worker: framing, child lifecycle, error paths."""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

from montagger.engine.session import EngineError, WorkerSessionManager
from montagger.engine.worker import encode_frame, read_frame


def test_frame_roundtrip():
    msg = {"id": 7, "op": "infer", "data_b64": "aGk="}
    buf = io.BytesIO(encode_frame(msg) + encode_frame({"id": 8, "op": "ping"}))
    assert read_frame(buf) == msg
    assert read_frame(buf) == {"id": 8, "op": "ping"}
    assert read_frame(buf) is None  # clean EOF


def test_frame_rejects_huge_header():
    import struct

    buf = io.BytesIO(struct.pack(">I", 1024 * 1024 * 1024) + b"junk")
    with pytest.raises(EngineError, match="refused"):
        read_frame(buf)


def test_worker_server_rejects_bad_requests(tmp_path):
    from montagger.engine.worker import WorkerServer

    server = WorkerServer("cpu", 0, 0)
    with pytest.raises(EngineError, match="missing"):
        server.handle({"op": "load", "model": "x", "dir": str(tmp_path / "nope")})
    with pytest.raises(EngineError, match="not loaded"):
        server.handle({"op": "infer", "model": "x"})
    with pytest.raises(EngineError, match="unknown op"):
        server.handle({"op": "dance"})
    assert server.handle({"op": "ping"}) == {}


def test_child_subprocess_lifecycle():
    """Spawn the real worker child: ping answers, bad load errors, quit exits."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "montagger.engine.worker", "cpu", "0", "0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        proc.stdin.write(encode_frame({"id": 1, "op": "ping"}))
        proc.stdin.flush()
        assert read_frame(proc.stdout) == {"id": 1, "ok": True}

        proc.stdin.write(
            encode_frame({"id": 2, "op": "load", "model": "x", "dir": "/nonexistent"})
        )
        proc.stdin.flush()
        reply = read_frame(proc.stdout)
        assert reply["ok"] is False and "missing" in reply["error"]

        proc.stdin.write(encode_frame({"id": 3, "op": "quit"}))
        proc.stdin.flush()
        assert proc.wait(timeout=15) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_manager_error_path_and_teardown(tmp_path):
    mgr = WorkerSessionManager("cpu", 0, 0)
    try:
        assert mgr.child_rss_mb() == 0.0  # nothing spawned yet
        with pytest.raises(EngineError, match="missing"):
            mgr.get(tmp_path / "nope", "x")
        assert mgr.loaded() == []
        mgr._request({"id": 0, "op": "ping"})  # force the child up
        assert mgr.child_rss_mb() > 0  # child is up and resident
        mgr.invalidate(None)  # kills the child
        assert mgr.child_rss_mb() == 0.0
    finally:
        mgr._kill_child()
