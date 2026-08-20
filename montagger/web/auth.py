"""Authentication for the WebUI and plugin endpoints.

The WebUI is open by default (like monbooru/monloader): when no webui_token
is configured, pages, SSE and actions are reachable without credentials, and
state-changing endpoints are protected by a per-instance CSRF token embedded
in the page (htmx sends it on every request, cross-site forms cannot). When a
webui_token IS configured, browser-facing routes require a login session
cookie or that token (bearer or ?token= for SSE, which cannot set headers).
The relay and pair/remove endpoints always require the pairing peer secret.
"""

from __future__ import annotations

import hmac
import logging
from typing import Any

from fastapi import Header, HTTPException, Query, Request

log = logging.getLogger(__name__)

SESSION_COOKIE = "montagger_session"


def _webui_token(request: Request) -> str:
    return getattr(request.app.state, "settings", None).webui_token  # type: ignore[union-attr]


def _check(presented: str, request: Request) -> bool:
    if not presented:  # an empty presented secret must never match an empty peer
        return False
    pairing = getattr(request.app.state, "pairing", None)
    if pairing and pairing.is_authentic(presented):
        return True
    token = _webui_token(request)
    return bool(token) and hmac.compare_digest(presented, token)


def _webui_enabled(request: Request) -> bool:
    return bool(_webui_token(request))


def _session_ok(request: Request) -> bool:
    value = request.cookies.get(SESSION_COOKIE)
    if not value:
        return False
    sessions: set[str] = getattr(request.app.state, "sessions", set())
    return value in sessions


def require_auth(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """Browser-facing routes.

    Open when no webui_token is configured; otherwise accept a login session
    cookie, or the token (matches either the webui token or the pairing peer
    secret, the latter for requests proxied through monbooru's open mode).
    """
    if _session_ok(request) or not _webui_enabled(request):
        return
    presented = ""
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:]
    if not _check(presented, request):
        raise HTTPException(status_code=401, detail="unauthorized")


def require_csrf(
    request: Request,
    x_montagger: str | None = Header(default=None),
) -> None:
    """State-changing WebUI routes: require the per-instance CSRF token.

    A login session is bound to the user's cookie jar (CSRF-safe), so it
    bypasses the check; otherwise the X-Montagger header the page embeds must
    match. Cross-site requests cannot set that header without CORS preflight.
    """
    if _session_ok(request):
        return
    token = getattr(request.app.state, "csrf", "")
    if not token or not x_montagger or not hmac.compare_digest(x_montagger, token):
        raise HTTPException(status_code=403, detail="csrf rejected")


def require_peer(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """Plugin contract routes: one of the stored pairing peer secrets."""
    presented = ""
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:]
    pairing = getattr(request.app.state, "pairing", None)
    if not pairing or not presented or not pairing.is_authentic(presented):
        raise HTTPException(status_code=401, detail="unauthorized")


def require_stream(
    request: Request,
    token: str | None = Query(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    """SSE: EventSource cannot set headers, so accept ?token= as well.

    Open when no webui_token is configured; otherwise the session cookie (sent
    automatically for same-origin EventSource), the ?token= query parameter,
    or a bearer header are accepted.
    """
    if _session_ok(request) or not _webui_enabled(request):
        return
    presented = token or ""
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:]
    if not _check(presented, request):
        raise HTTPException(status_code=401, detail="unauthorized")