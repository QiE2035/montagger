"""Pairing with monbooru - manual, operator-initiated (like monloader).

montagger never reaches out on its own: start() only restores stored
credentials from disk. To pair, the operator clicks "connect to monbooru"
in the settings page, which sends one offer and starts polling for
approval; the WebUI re-renders the panel every 2s while an attempt is in
flight. Credentials live in <state>/credentials.json (0600) as
{"token", "peer"}: token authenticates our API calls to monbooru, peer is
the secret monbooru presents on every request it sends to us. A 401/403
from the API drops the credentials and leaves the operator to connect
again - never a silent re-offer.
"""

from __future__ import annotations

import hmac
import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Callable

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
        self._attempt_lock = threading.Lock()
        self._attempt: dict[str, str] | None = None  # {"request_id", "peer"} in flight
        self._attempt_stop = threading.Event()
        self._message = ""  # last operator-facing message (connect/deny/expire...)

    # ---- public API -----------------------------------------------------

    def start(self) -> None:
        """Restore stored credentials. No offer is sent until the operator
        clicks connect in the WebUI."""
        self._ensure_loaded()

    def stop(self) -> None:
        self._stopping.set()
        self.cancel()

    def token(self) -> str:
        return self._ensure_loaded().token

    def peer(self) -> str:
        return self._ensure_loaded().peer

    def paired(self) -> bool:
        return self._ensure_loaded().complete

    def waiting(self) -> bool:
        """An offer was sent and is awaiting approval in monbooru."""
        with self._attempt_lock:
            return self._attempt is not None

    def state_message(self) -> str:
        """Last operator-facing message (why the last attempt failed etc)."""
        return self._message

    def is_authentic(self, presented: str) -> bool:
        """Constant-time check that a request really came from monbooru."""
        with self._lock:
            peer = self._creds.peer
        return bool(peer) and hmac.compare_digest(presented, peer)

    def challenged(self) -> None:
        """401/403 from the API: drop the credentials, no re-offer - the
        operator must connect again in the WebUI."""
        self.forget()
        self._message = "monbooru rejected the credentials; connect again to re-pair"

    def set_base_url(self, url: str) -> None:
        """Hot-update the monbooru address (used from the settings save).
        The shared httpx client is host-agnostic; stored credentials are
        kept, they may still be valid after a port change."""
        self.monbooru_url = url.rstrip("/")

    def set_self_url(self, url: str) -> None:
        """Hot-update the address monbooru calls us back at."""
        self.self_url = url

    def unpair(self) -> None:
        """The operator removed the pairing on the monbooru side."""
        self.forget()
        self._message = "the pairing was removed on the monbooru side; connect again to re-pair"

    def connect(self) -> None:
        """Manual connect: send one pairing offer, then poll for approval in
        a background thread. Failures surface via state_message()."""
        with self._lock:
            if self._creds.complete:
                self._message = "already paired; remove the pairing first to re-pair"
                return
        with self._attempt_lock:
            if self._attempt is not None:
                self._message = "waiting"
                return
        try:
            peer = secrets.token_urlsafe(32)
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
        except httpx.TransportError as exc:
            self._message = f"could not reach monbooru: {exc}"
            return
        if resp.status_code >= 400:
            self._message = f"monbooru refused the request: {resp.status_code} {resp.text[:200]}"
            return
        request_id = (resp.json() or {}).get("request_id")
        if not request_id:
            self._message = "monbooru answered without a request_id"
            return
        with self._attempt_lock:
            self._attempt = {"request_id": request_id, "peer": peer}
        self._attempt_stop.clear()
        threading.Thread(
            target=self._poll_loop, args=(request_id, peer), name="pairing-poll", daemon=True
        ).start()
        log.info("pairing: offered pairing with %s; waiting for approval", self.monbooru_url)
        self._notify()

    def cancel(self) -> None:
        """Abort an in-flight attempt."""
        self._attempt_stop.set()
        with self._attempt_lock:
            if self._attempt is not None:
                self._attempt = None
                self._message = ""
        self._notify()

    def remove(self) -> None:
        """Operator-side removal: drop our credentials and ask monbooru to
        drop its side too (one click tears down both ends, like monloader)."""
        with self._lock:
            token = self._creds.token
        self.forget()
        self._message = ""
        if token:
            try:
                resp = self.http.post(
                    f"{self.monbooru_url}/api/v1/pair/remove",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if resp.status_code >= 400:
                    self._message = (
                        "removed locally; monbooru could not be reached - remove the pairing there by hand"
                    )
            except httpx.TransportError:
                self._message = (
                    "removed locally; monbooru could not be reached - remove the pairing there by hand"
                )
        self._notify()

    # ---- internals ------------------------------------------------------

    def _poll_loop(self, request_id: str, peer: str) -> None:
        """Poll pair/status until the offer is approved, expires or is
        denied, or the operator cancels."""
        while not self._attempt_stop.is_set():
            try:
                poll = self.http.get(f"{self.monbooru_url}/api/v1/pair/status?id={request_id}")
            except httpx.TransportError:
                self._finish_attempt(
                    request_id, "monbooru unreachable; the attempt was dropped, try again"
                )
                return
            if poll.status_code == 404:
                self._finish_attempt(request_id, "the pairing offer expired; connect again")
                return
            if poll.status_code >= 400:
                self._attempt_stop.wait(POLL_INTERVAL)
                continue
            answer = poll.json()
            token = answer.get("token") or ""
            if token:
                self._set(Credentials(token=token, peer=peer))
                self._save()
                self._finish_attempt(request_id, "")
                log.info("paired with %s", self.monbooru_url)
                return
            if answer.get("status") == "denied":
                self._finish_attempt(request_id, "the pairing was denied in monbooru")
                return
            self._attempt_stop.wait(POLL_INTERVAL)
        # cancelled: cancel() already cleared the attempt

    def _finish_attempt(self, request_id: str, message: str) -> None:
        """Clear the attempt only if it is still the one this poller owns."""
        with self._attempt_lock:
            att = self._attempt
            if att is None or att["request_id"] != request_id:
                return  # cancelled or superseded
            self._attempt = None
        self._message = message  # an empty message clears stale ones
        self._notify()

    def _notify(self) -> None:
        if self.on_change:
            self.on_change(self._creds.complete)

    def forget(self) -> None:
        with self._lock:
            self._creds = Credentials()
        try:
            self._path().unlink(missing_ok=True)
        except OSError:
            pass
        self._notify()

    def _set(self, creds: Credentials) -> None:
        with self._lock:
            self._creds = creds
        self._notify()

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