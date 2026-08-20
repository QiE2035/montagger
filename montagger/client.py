"""REST client for the monbooru API.

Everything stays in memory: fetch_image returns raw bytes that the pipeline
decodes with Pillow straight from a BytesIO. One shared httpx client keeps
connections alive across the many per-image calls (there is no bulk tag
endpoint, so 50k images mean 50k calls - connection reuse matters).

A 401/403 raises MonbooruAuthError after calling the registered handler so
the pairing machinery can re-offer; transient failures (network, 429, 5xx)
are retried a couple of times with backoff.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

import httpx

log = logging.getLogger(__name__)


class MonbooruError(Exception):
    """Expected API refusal (4xx/5xx)."""


class MonbooruAuthError(MonbooruError):
    """Credentials were rejected; pairing must re-offer."""


class MonbooruStatusError(MonbooruError):
    """A 4xx/5xx reply that is not an auth failure."""

    def __init__(self, status: int, detail: str = "") -> None:
        self.status = status
        super().__init__(f"{status}: {detail}")


class MonbooruClient:
    def __init__(
        self,
        base_url: str,
        get_token: Callable[[], str],
        on_unauthorized: Callable[[], None] | None = None,
    ) -> None:
        self.base = base_url.rstrip("/")
        self.get_token = get_token
        self.on_unauthorized = on_unauthorized
        self.http = httpx.Client(
            timeout=httpx.Timeout(connect=10, read=120, write=60, pool=10),
            follow_redirects=True,
        )

    def set_base_url(self, url: str) -> None:
        """Hot-update the monbooru address (settings save). The shared
        httpx client is host-agnostic, so only the path base changes."""
        self.base = url.rstrip("/")

    # ---- core calls -----------------------------------------------------

    def fetch_image(self, image_id: int) -> bytes:
        resp = self.http.get(f"{self.base}/api/v1/images/{image_id}/file")
        self._check(resp, image_id=image_id)
        return resp.content

    def add_tags(self, image_id: int, tags: list[str], via: str, retries: int = 2) -> None:
        body = {"tags": tags, "via": via}
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = self.http.post(f"{self.base}/api/v1/images/{image_id}/tags", json=body)
                self._check(resp, image_id=image_id)
                return
            except MonbooruAuthError:
                raise
            except MonbooruStatusError as exc:  # only retry transient statuses
                if exc.status not in (408, 429) and exc.status < 500:
                    raise
                last = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
            except httpx.TransportError as exc:
                last = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        raise MonbooruError(f"add_tags failed after {retries + 1} tries: {last}")

    def image_status(self, image_id: int) -> dict[str, Any]:
        resp = self.http.get(f"{self.base}/api/v1/images/{image_id}")
        self._check(resp, image_id=image_id)
        return resp.json()

    def galleries(self) -> list[dict[str, Any]]:
        resp = self.http.get(f"{self.base}/api/v1/galleries")
        self._check(resp)
        return resp.json()

    def categories(self) -> dict[str, int]:
        """Map API category names to ids, e.g. {'general': 0, 'character': 1}."""
        resp = self.http.get(f"{self.base}/api/v1/categories")
        self._check(resp)
        data = resp.json()
        if isinstance(data, list):
            return {item.get("name", ""): item.get("id", i) for i, item in enumerate(data)}
        if isinstance(data, dict):
            return {k: v for k, v in data.items()}
        return {}

    # ---- helpers --------------------------------------------------------

    def _check(self, resp: httpx.Response, **ctx: Any) -> None:
        if resp.status_code < 400:
            return
        detail = resp.text[:500]
        if resp.status_code in (401, 403):
            log.warning("monbooru rejected our credentials (%s): %s", resp.status_code, detail)
            if self.on_unauthorized:
                self.on_unauthorized()
            raise MonbooruAuthError(f"{resp.status_code}: {detail}")
        raise MonbooruStatusError(resp.status_code, f"{ctx}: {detail}")

    def close(self) -> None:
        self.http.close()