"""Auth: an optional UI password over cookie sessions, and API bearer
tokens stored in SQLite.

Open-LAN rules: with no password set and no tokens configured, every
surface answers unauthenticated - that is the default for a first run on
a trusted LAN, and the settings page says so in as many words. The moment
either mechanism exists, the surface it guards starts enforcing.
"""

from __future__ import annotations

import secrets
import threading
import time

from . import logx

log = logx.get("auth")

_SESSION_COOKIE = "montagger_session"


class SessionStore:
    """In-memory session set (copied from monloader)."""

    def __init__(self):
        self._mu = threading.RLock()
        self._sessions: dict[str, float] = {}  # id -> expires at (monotonic)

    def new(self, lifetime_days: int) -> str:
        session_id = secrets.token_urlsafe(32)
        with self._mu:
            self._sessions[session_id] = time.monotonic() + lifetime_days * 86400
        return session_id

    def check(self, session_id: str | None) -> bool:
        if not session_id:
            return False
        with self._mu:
            expires = self._sessions.get(session_id)
            if expires is None:
                return False
            if time.monotonic() > expires:
                del self._sessions[session_id]
                return False
            return True

    def delete(self, session_id: str | None) -> None:
        if session_id:
            with self._mu:
                self._sessions.pop(session_id, None)

    def clear(self) -> None:
        with self._mu:
            self._sessions.clear()
