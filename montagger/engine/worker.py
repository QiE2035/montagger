"""Isolated inference worker.

ONNX Runtime never truly returns accelerator memory to the OS while the
process lives: the CUDA context, cuDNN handles, and arena bookkeeping stay
resident even after every session is dropped. montagger therefore runs all
ORT sessions in a dedicated child process. Releasing memory = terminating
the child, which hands every byte back in one shot.

Protocol (4-byte big-endian length + UTF-8 JSON on stdin/stdout):

    -> {"id": n, "op": "load",  "model": name, "dir": path}
    <- {"id": n, "ok": true, "input_size": px}
    -> {"id": n, "op": "infer", "model": name, "data_b64": ...}
    <- {"id": n, "ok": true, "scores_b64": ..., "width": w, "height": h}
    -> {"id": n, "op": "unload", "model": name}   -> {"id": n, "ok": true}
    -> {"id": n, "op": "ping"}                    -> {"id": n, "ok": true}
    -> {"id": n, "op": "quit"}                    (child exits)

Errors come back as {"id": n, "ok": false, "error": "..."} without killing
the loop; the child only dies on quit or a broken pipe.
"""

from __future__ import annotations

import base64
import json
import struct
import sys
from pathlib import Path

from .session import EngineError, ModelRuntime

MAX_FRAME = 256 * 1024 * 1024  # refuses absurd frames; images are a few MB


def encode_frame(obj: dict) -> bytes:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(body)) + body


def read_frame(stream) -> dict | None:
    """One framed message; None on clean EOF before any byte."""
    head = stream.read(4)
    if not head:
        return None
    if len(head) < 4:
        raise EngineError("worker protocol: truncated frame header")
    (size,) = struct.unpack(">I", head)
    if size > MAX_FRAME:
        raise EngineError(f"worker protocol: frame of {size} bytes refused")
    body = b""
    while len(body) < size:
        chunk = stream.read(size - len(body))
        if not chunk:
            raise EngineError("worker protocol: truncated frame body")
        body += chunk
    return json.loads(body.decode("utf-8"))


class WorkerServer:
    """The child-side loop: real ModelRuntimes, one reply per request."""

    def __init__(self, provider: str, device_id: int, intra_op_threads: int):
        from .session import resolve_provider

        self.provider = resolve_provider(provider)
        self.device_id = device_id
        self.intra_op_threads = intra_op_threads
        self.runtimes: dict[str, ModelRuntime] = {}

    def handle(self, msg: dict) -> dict:
        op = msg.get("op")
        name = str(msg.get("model", ""))
        if op == "load":
            if name in self.runtimes:
                runtime = self.runtimes[name]
            else:
                runtime = ModelRuntime(
                    name, Path(str(msg["dir"])), self.provider,
                    self.device_id, self.intra_op_threads,
                )
                self.runtimes[name] = runtime
            return {"input_size": runtime.input_size}
        if op == "infer":
            runtime = self.runtimes.get(name)
            if runtime is None:
                raise EngineError(f"model {name} is not loaded in the worker")
            data = base64.b64decode(msg["data_b64"])
            scores, width, height = runtime.infer(data)
            return {
                "scores_b64": base64.b64encode(scores.tobytes()).decode("ascii"),
                "width": width,
                "height": height,
            }
        if op == "unload":
            self.runtimes.pop(name, None)
            return {}
        if op == "ping":
            return {}
        raise EngineError(f"unknown op {op!r}")


def serve(provider: str, device_id: int, intra_op_threads: int) -> None:
    """Read requests from stdin until quit; replies go to stdout."""
    server = WorkerServer(provider, device_id, intra_op_threads)
    out = sys.stdout.buffer
    while True:
        try:
            msg = read_frame(sys.stdin.buffer)
        except Exception as err:  # a broken stream is fatal for the child
            print(f"montagger worker: fatal: {err}", file=sys.stderr, flush=True)
            return
        if msg is None or msg.get("op") == "quit":
            return
        reply = {"id": msg.get("id")}
        try:
            reply.update(server.handle(msg))
            reply["ok"] = True
        except Exception as err:
            reply.clear()
            reply["id"] = msg.get("id")
            reply["ok"] = False
            reply["error"] = str(err)
        try:
            out.write(encode_frame(reply))
            out.flush()
        except BrokenPipeError:
            return


if __name__ == "__main__":
    serve(
        sys.argv[1] if len(sys.argv) > 1 else "cpu",
        int(sys.argv[2]) if len(sys.argv) > 2 else 0,
        int(sys.argv[3]) if len(sys.argv) > 3 else 0,
    )
