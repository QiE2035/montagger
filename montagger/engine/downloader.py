"""The model downloader: plain streaming GET against the catalog's direct
HF resolve URLs.

Deliberately not huggingface_hub: the catalog links are plain downloads,
and hub-grade machinery (its dependency chain, cache layout and
.incomplete staging) buys nothing here. What HF actually gates behind
login is covered with an explicit bearer token, and the endpoint is
configurable for mirrors. Downloads stream to <name>/<file>.part inside
the model directory and rename atomically, so a crashed download never
damages an existing model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from .. import logx

if TYPE_CHECKING:
    from ..config import Config

log = logx.get("downloader")

# Growth per read; a 1.3 GB model downloads in ~4 MB chunks.
_CHUNK = 4 * 1024 * 1024


class DownloadError(Exception):
    pass


def hf_headers(cfg: "Config") -> dict:
    headers = {"User-Agent": "montagger"}
    token = cfg.hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def hf_url(cfg: "Config", url: str) -> str:
    """Rewrite the HF origin to the configured endpoint (mirror support)."""
    endpoint = cfg.hf_endpoint()
    if endpoint and url.startswith("https://huggingface.co/"):
        return endpoint + url[len("https://huggingface.co") :]
    return url


def install(cfg: Config, model_root: Path, entry, progress=None) -> Path:
    """Download every catalog file of one entry into model_root/<name>/.
    Returns the model directory. progress(received, total) is optional."""
    model_dir = model_root / entry.name
    model_dir.mkdir(parents=True, exist_ok=True)
    headers = hf_headers(cfg)
    if entry.gated and not cfg.hf_token():
        raise DownloadError(
            f"{entry.name} is a gated Hugging Face repo: set hf.token in montagger.toml "
            "(or the HF_TOKEN environment variable) and install again"
        )

    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=600.0)) as client:
        for f in entry.files:
            target = model_dir / f.filename
            part = model_dir / (f.filename + ".part")
            done = target.exists()
            if done:
                log.info("model %s: %s already present", entry.name, f.filename)
                continue
            url = hf_url(cfg, f.url)
            log.info("model %s: downloading %s", entry.name, f.filename)
            try:
                with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code in (401, 403):
                        hint = " (gated repo: configure hf.token)" if entry.gated else ""
                        raise DownloadError(f"{f.filename}: HTTP {resp.status_code}{hint}")
                    if resp.status_code != 200:
                        raise DownloadError(f"{f.filename}: HTTP {resp.status_code}")
                    total = int(resp.headers.get("content-length") or 0)
                    received = 0
                    with open(part, "wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=_CHUNK):
                            fh.write(chunk)
                            received += len(chunk)
                            if progress is not None and total:
                                progress(received, total)
                os.replace(part, target)
            except DownloadError:
                _cleanup(part)
                raise
            except (httpx.HTTPError, OSError) as err:
                _cleanup(part)
                raise DownloadError(f"{f.filename}: {err}") from err
    return model_dir


def _cleanup(part: Path) -> None:
    try:
        part.unlink(missing_ok=True)
    except OSError:
        pass
