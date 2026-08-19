"""Pairing with monbooru - a faithful port of the simple-edit plugin flow.

Offer the pairing, poll for approval, persist the credentials monbooru
issues, and re-offer when they stop working or the operator removes the
pairing on the monbooru side. Credentials live in
<state>/credentials.json (0600) as {"token", "peer"}: token authenticates
our API calls to monbooru, peer is the secret monbooru presents on every
request it sends to us.
"""

from __future__ import annotations

import hmac
import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from montagger import __version__
from montagger.settings import BUTTONS

log = logging.getLogger(__name__)

APP_NAME = "montagger"
SCOPES = ["read", "write"]
POLL_INTERVAL = 2.0


class Credentials:
    def __init__(self, token: str = "", peer: str = "") -> None:
        self.token = token
        self.peer = peer

    @property
    def complete(self) -> bool:
        return bool(self.token and self.peer)


class Pairing:
    def __init__(
        self,
        monbooru_url: str,
        self_url: str,
        state_dir: Path,
        on_change: Callable[[bool], None] | None = None,
    ) -> None:
        self.monbooru_url = monbooru_url.rstrip("/")
        self.self_url = self_url
        self.state_dir = Path(state_dir)
        self.on_change = on_change
        self.http = httpx.Client(timeout=httpx.Timeout(connect=10, read=30, write=30, pool=10))
        self._lock = threading.Lock()
        self._creds = Credentials()
        self._stopping = threading.Event()
        self._running = threading.Event()

    # ---- public API -----------------------------------------------------

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        threading.Thread(target=self._loop, name="pairing", daemon=True).start()

    def stop(self) -> None:
        self._stopping.set()

    def token(self) -> str:
        return self._ensure_loaded().token

    def peer(self) -> str:
        return self._ensure_loaded().peer

    def paired(self) -> bool:
        return self._ensure_loaded().complete

    def is_authentic(self, presented: str) -> bool:
        """Constant-time check that a request really came from monbooru."""
        with self._lock:
            peer = self._creds.peer
        return bool(peer) and hmac.compare_digest(presented, peer)

    def challenged(self) -> None:
        """401/403 from the API: drop the credentials and re-offer."""
        self.forget()
        if self._running.is_set():
            threading.Thread(target=self._loop, name="pairing", daemon=True).start()

    def unpair(self) -> None:
        """The operator removed the pairing on the monbooru side."""
        self.forget()
        if self._running.is_set():
            threading.Thread(target=self._loop, name="pairing", daemon=True).start()

    # ---- internals ------------------------------------------------------

    def _loop(self) -> None:
        self._ensure_loaded()
        # Restore stored credentials first, validate them, else offer.
        stored = self._load()
        if stored and "token" in stored and "peer" in stored:
            self._set(Credentials(**stored))
            if self.token_accepted():
                return
            log.info("monbooru no longer accepts the stored credentials; offering again")
            self.forget()
        while not self._stopping.is_set():
            try:
                peer = secrets.token_urlsafe(32)
                token = self._offer(peer)
            except httpx.TransportError:
                if not getattr(self, "_unreachable_logged", False):
                    log.warning("monbooru unreachable; retrying pairing every 3s", exc_info=True)
                    self._unreachable_logged = True
                else:
                    log.info("monbooru still unreachable; retrying")
                if self._stopping.wait(3.0):
                    return
                continue
            if token is None:  # the offer aged out without approval
                log.info("the pairing offer expired, offering again")
                continue
            self._set(Credentials(token=token, peer=peer))
            self._save()
            log.info("paired with %s", self.monbooru_url)
            return

    def _offer(self, peer: str) -> str | None:
        offered: dict[str, Any] = {}
        resp = self.http.post(
            f"{self.monbooru_url}/api/v1/pair/request",
            json={
                "app": APP_NAME,
                "url": self.self_url,
                "requested_scopes": SCOPES,
                "peer_token": peer,
                "version": __version__,
                "buttons": BUTTONS,
            },
        )
        if resp.status_code >= 400:
            log.warning("pairing request refused: %s %s", resp.status_code, resp.text[:300])
            time.sleep(POLL_INTERVAL)
            return None
        offered = resp.json()
        request_id = offered.get("request_id")
        log.info("waiting for approval in monbooru: Settings > Plugins")
        while not self._stopping.is_set():
            if self._stopping.is_set():
                return None
            poll = self.http.get(f"{self.monbooru_url}/api/v1/pair/status?id={request_id}")
            if poll.status_code == 404:
                return None  # the offer expired
            if poll.status_code >= 400:
                time.sleep(POLL_INTERVAL)
                continue
            answer = poll.json()
            token = answer.get("token") or ""
            if token:
                return token
            if answer.get("status") == "denied":
                log.info("the pairing was denied; offering again shortly")
                time.sleep(POLL_INTERVAL * 3)
                return None
            time.sleep(POLL_INTERVAL)
        return None

    def token_accepted(self) -> bool:
        try:
            resp = self.http.get(f"{self.monbooru_url}/api/v1/galleries", headers=self._auth())
            return resp.status_code not in (401, 403, 503)
        except httpx.TransportError:
            return False

    def _auth(self) -> dict[str, str]:
        with self._lock:
            token = self._creds.token
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _set(self, creds: Credentials) -> None:
        with self._lock:
            self._creds = creds
        if self.on_change:
            self.on_change(creds.complete)

    def _ensure_loaded(self) -> Credentials:
        with self._lock:
            if not self._creds.complete:
                stored = self._load()
                if stored:
                    self._creds = Credentials(**stored)
            return self._creds

    def _path(self) -> Path:
        return self.state_dir / "credentials.json"

    def _load(self) -> dict[str, str]:
        try:
            raw = json.loads(self._path().read_text(encoding="utf-8"))
            return {k: str(raw[k]) for k in ("token", "peer") if k in raw}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self._path()
        path.write_text(
            json.dumps({"token": self._creds.token, "peer": self._creds.peer}, indent=2),
            encoding="utf-8",
        )
        try:
            path.chmod(0o600)
        except OSError:  # Windows: chmod is a no-op for this purpose
            pass

    def forget(self) -> None:
        with self._lock:
            self._creds = Credentials()
        try:
            self._path().unlink()
        except FileNotFoundError:
            pass
        if self.on_change:
            self.on_change(False)