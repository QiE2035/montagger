"""Authentication for the WebUI and plugin endpoints.

Two credentials are accepted for browser-facing routes: the pairing peer
secret (requests proxied through monbooru's open mode present it) and the
optional --webui-token for direct access. The relay and pair/remove
endpoints require the pairing secret specifically. SSE cannot send
Authorization headers, so /api/stream also accepts ?token=.
"""

from __future__ import annotations

import hmac
import logging
from typing import Any

from fastapi import Header, HTTPException, Query, Request

log = logging.getLogger(__name__)


def _peer_secret(request: Request) -> str:
    return getattr(request.app.state, "pairing", None).peer()  # type: ignore[union-attr]


def _webui_token(request: Request) -> str:
    return getattr(request.app.state, "settings", None).webui_token  # type: ignore[union-attr]


def _check(presented: str, request: Request) -> bool:
    if not presented:  # an empty presented secret must never match an empty peer
        return False
    peer = _peer_secret(request)
    if peer and hmac.compare_digest(presented, peer):
        return True
    token = _webui_token(request)
    return bool(token) and hmac.compare_digest(presented, token)


def require_auth(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """Browser-facing routes: Bearer peer secret or webui token."""
    presented = ""
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:]
    if not _check(presented, request):
        raise HTTPException(status_code=401, detail="unauthorized")


def require_peer(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """Plugin contract routes: the pairing secret only."""
    presented = ""
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:]
    peer = _peer_secret(request)
    if not peer or not presented or not hmac.compare_digest(presented, peer):
        raise HTTPException(status_code=401, detail="unauthorized")


def require_stream(
    request: Request,
    token: str | None = Query(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    """SSE: EventSource cannot set headers, so accept ?token= as well."""
    presented = token or ""
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:]
    if not _check(presented, request):
        raise HTTPException(status_code=401, detail="unauthorized")