"""The thin HF downloader: token header, endpoint rewrite, atomic .part."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from montagger.config import Config
from montagger.engine.catalog import CatalogEntry, CatalogFile
from montagger.engine import downloader


class StubHF(BaseHTTPRequestHandler):
    payload = b"onnx-bytes" * 1000
    seen = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        StubHF.seen["auth"] = self.headers.get("Authorization")
        StubHF.seen["path"] = self.path
        if self.path.startswith("/gated/"):
            self.send_response(401)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), StubHF)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def entry(name="m", files=None, gated=False):
    return CatalogEntry(
        name=name,
        description="",
        files=files or [CatalogFile(url="https://huggingface.co/repo/resolve/main/model.onnx", filename="model.onnx")],
        gated=gated,
    )


def test_download_rewrites_endpoint_and_writes_atomically(server, tmp_path, provider):
    provider.update(lambda c: setattr(c.hf, "endpoint", server))
    model_dir = downloader.install(provider.current(), tmp_path, entry())
    assert (model_dir / "model.onnx").read_bytes() == StubHF.payload
    assert not list(model_dir.glob("*.part"))  # atomic rename left nothing behind
    assert StubHF.seen["path"].endswith("/repo/resolve/main/model.onnx")


def test_token_header_when_configured(server, tmp_path, provider):
    provider.update(lambda c: setattr(c.hf, "token", "hf_secret"))
    provider.update(lambda c: setattr(c.hf, "endpoint", server))
    downloader.install(provider.current(), tmp_path, entry(name="m2"))
    assert StubHF.seen["auth"] == "Bearer hf_secret"


def test_gated_without_token_refused(server, tmp_path, provider):
    provider.update(lambda c: setattr(c.hf, "token", ""))
    with pytest.raises(downloader.DownloadError, match="gated"):
        downloader.install(provider.current(), tmp_path, entry(name="g", gated=True))


def test_http_error_cleans_part(server, tmp_path, provider):
    provider.update(lambda c: setattr(c.hf, "endpoint", server))
    gated = entry(name="gg", files=[CatalogFile(url=f"{server}/gated/x", filename="model.onnx")], gated=False)
    with pytest.raises(downloader.DownloadError):
        downloader.install(provider.current(), tmp_path, gated)
    assert not (tmp_path / "gg" / "model.onnx.part").exists()
