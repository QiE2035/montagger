"""What montagger actually sends to monbooru: category prefixes on tags,
model provenance in source (enrich) and via (push)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import ValidationError

from montagger.config import Provider
from montagger.engine.scoring import ScoredTag
from montagger.pair import MonbooruClient, wire_tags


def test_wire_tags_prefix_every_category():
    tags = [
        ScoredTag(name="1girl", category="general", confidence=0.9),
        ScoredTag(name="hatsune_miku", category="character", confidence=0.8),
        ScoredTag(name="explicit", category="rating", confidence=0.7),
    ]
    assert wire_tags(tags) == [
        "general:1girl",
        "character:hatsune_miku",
        "rating:explicit",
    ]


def test_wire_tags_colon_in_name_survives():
    # Only the first colon splits: "general::3" → category general, name ":3".
    tags = [ScoredTag(name=":3", category="general", confidence=0.9)]
    assert wire_tags(tags) == ["general::3"]


class Capture(BaseHTTPRequestHandler):
    seen = {}

    def log_message(self, *a):
        pass

    def _ok(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        Capture.seen[self.path] = {"auth": self.headers.get("Authorization"), "raw": raw}
        if "/enrich" in self.path:
            self._ok({"merge": {}})
        else:
            self._ok({"id": 77})


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Capture)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


@pytest.fixture
def client(tmp_path, server):
    cfg = Provider(tmp_path / "c.toml")
    cfg.update(lambda c: setattr(c.monbooru, "api_url", server))
    cfg.update(lambda c: setattr(c.monbooru, "token", "tok"))
    return MonbooruClient(cfg), server


def test_enrich_carries_categories_and_model(client):
    mc, server = client
    tags = [ScoredTag(name="miku", category="character", confidence=0.9)]
    mc.enrich(5, wire_tags(tags), model="wd-swinv2")
    sent = json.loads(Capture.seen["/api/v1/images/5/enrich"]["raw"])
    assert sent["tags"] == ["character:miku"]
    assert sent["source"] == "montagger/wd-swinv2"


def test_push_carries_via_and_categories(client):
    mc, server = client
    tags = [ScoredTag(name="1girl", category="general", confidence=0.9)]
    mc.push_image(b"jpeg-bytes", "x.jpg", wire_tags(tags), "g1", model="wd-swinv2")
    sent = Capture.seen["/api/v1/images?gallery=g1"]["raw"]
    # multipart: fields arrive urlencoded in the body; parse loosely.
    body = sent.decode("utf-8", "replace")
    assert '"general:1girl"' in body
    assert "montagger/wd-swinv2" in body
    assert "local AI tag" not in body  # the fake source field is gone
    assert "autotag" not in body  # no builtin-tagger passthrough anymore


def test_wire_tags_from_rows_and_resolve_models(tmp_path):
    from montagger.config import Config
    from montagger.pair import resolve_models, wire_tags_from_rows

    rows = [
        {"name": "1girl", "category": "general", "confidence": 0.9},
        {"name": ":3", "category": "general", "confidence": 0.8},
    ]
    assert wire_tags_from_rows(rows) == ["general:1girl", "general::3"]
    cfg = Config()
    assert resolve_models(cfg) == ["wd-swinv2"]  # default fallback
    cfg.models.default_models = ["wd-swinv2", "joytag"]
    assert resolve_models(cfg) == ["wd-swinv2", "joytag"]
    # blanks and duplicates are rejected at the model boundary now
    with pytest.raises(ValidationError):
        cfg.models.default_models = [" ", "a", "a"]
