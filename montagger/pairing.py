"""Pairing with monbooru - manual, operator-initiated, one-to-many.

montagger never reaches out on its own: start() only restores stored
credentials. To pair, the operator clicks "connect to monbooru" in the
settings page (per monbooru url), which sends one offer and starts polling
for approval; the WebUI re-renders the panel every 2s while an attempt is
in flight.

Credentials live in montagger.toml as a [[pairing]] array (like
monloader's [[auth.tokens]]), one entry per monbooru instance:

    [[pairing]]
    url = "http://127.0.0.1:8080"
    token = "..."   # authenticates our API calls to that monbooru
    peer = "..."    # the secret that monbooru presents when calling us
    created_at = "..."

A 401/403 from the API drops that instance's credentials and leaves the
operator to connect again - never a silent re-offer. Inbound requests
present the peer secret, which is unique per pairing, so the source
monbooru is recovered by matching it against the entries.
"""

from __future__ import annotations

import hmac
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
import tomlkit
import tomllib

from montagger import __version__
from montagger.settings import BUTTONS, TOML_LOCK

log = logging.getLogger(__name__)

APP_NAME = "montagger"
SCOPES = ["read", "write"]
POLL_INTERVAL = 2.0


@dataclass
class PairingEntry:
    url: str
    token: str = ""
    peer: str = ""
    created_at: str = ""

    @property
    def complete(self) -> bool:
        return bool(self.token and self.peer)


def _entries_to_toml(entries: list[PairingEntry]) -> Any:
    """Build a tomlkit array-of-tables that survives round-trips."""
    aot = tomlkit.aot()
    for entry in entries:
        table = tomlkit.table()
        table["url"] = entry.url
        table["token"] = entry.token
        table["peer"] = entry.peer
        table["created_at"] = entry.created_at
        aot.append(table)
    return aot


class Pairing:
    def __init__(
        self,
        self_url: str,
        config_path: Path,
        on_change: Callable[[bool], None] | None = None,
    ) -> None:
        self.self_url = self_url
        self.config_path = Path(config_path)
        self.on_change = on_change
        self.http = httpx.Client(timeout=httpx.Timeout(connect=10, read=30, write=30, pool=10))
        self._lock = threading.RLock()
        self._entries: list[PairingEntry] = []
        self._stopping = threading.Event()
        self._attempt: dict[str, str] | None = None  # {"url", "request_id", "peer"}
        self._attempt_stop = threading.Event()
        self._attempt_lock = threading.Lock()
        self._message = ""  # last operator-facing message (connect/deny/expire...)

    # ---- public API -----------------------------------------------------

    def start(self) -> None:
        """Restore stored credentials from montagger.toml. No offer is sent
        until the operator clicks connect in the WebUI."""
        with self._lock:
            self._entries = self._load()
        complete = sum(1 for e in self._entries if e.complete)
        if complete:
            log.info("pairing: restored %d stored pairing(s)", complete)

    def stop(self) -> None:
        self._stopping.set()
        self.cancel()

    def entries(self) -> list[PairingEntry]:
        with self._lock:
            return list(self._entries)

    def paired(self) -> bool:
        with self._lock:
            return any(e.complete for e in self._entries)

    def entry_for_url(self, url: str) -> PairingEntry | None:
        url = url.rstrip("/")
        with self._lock:
            for entry in self._entries:
                if entry.url == url:
                    return entry
        return None

    def entry_for_peer(self, presented: str) -> PairingEntry | None:
        """Recover the source monbooru from the peer secret it presents."""
        with self._lock:
            for entry in self._entries:
                if entry.peer and hmac.compare_digest(presented, entry.peer):
                    return entry
        return None

    def token_for(self, url: str) -> str:
        entry = self.entry_for_url(url)
        return entry.token if entry else ""

    def is_authentic(self, presented: str) -> bool:
        return self.entry_for_peer(presented) is not None

    def waiting(self) -> bool:
        """An offer was sent and is awaiting approval in monbooru."""
        with self._attempt_lock:
            return self._attempt is not None

    def waiting_url(self) -> str:
        with self._attempt_lock:
            return (self._attempt or {}).get("url", "")

    def state_message(self) -> str:
        """Last operator-facing message (why the last attempt failed etc)."""
        return self._message

    def challenged(self, url: str) -> None:
        """401/403 from the API: drop that instance's credentials, no
        re-offer - the operator must connect again in the WebUI."""
        self._drop_entry(url)
        self._message = f"monbooru {url} rejected the credentials; connect again to re-pair"

    def unpair(self, presented: str) -> None:
        """The pairing was removed on the monbooru side (pair/remove call)."""
        entry = self.entry_for_peer(presented)
        message = "the pairing was removed on the monbooru side; connect again to re-pair"
        if entry:
            self._drop_entry(entry.url)
            message = f"the pairing with {entry.url} was removed on the monbooru side; connect again to re-pair"
        self._message = message

    def connect(self, url: str) -> None:
        """Manual connect: send one pairing offer to the given monbooru,
        then poll for approval in a background thread. Failures surface via
        state_message()."""
        url = url.strip().rstrip("/")
        if not url:
            self._message = "enter the monbooru address to pair with"
            return
        entry = self.entry_for_url(url)
        if entry and entry.complete:
            self._message = f"already paired with {url}; remove that pairing first"
            return
        with self._attempt_lock:
            if self._attempt is not None:
                self._message = "waiting"
                return
        try:
            peer = secrets.token_urlsafe(32)
            resp = self.http.post(
                f"{url}/api/v1/pair/request",
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
            self._message = f"could not reach monbooru {url}: {exc}"
            return
        if resp.status_code >= 400:
            self._message = f"monbooru {url} refused the request: {resp.status_code} {resp.text[:200]}"
            return
        request_id = (resp.json() or {}).get("request_id")
        if not request_id:
            self._message = f"monbooru {url} answered without a request_id"
            return
        with self._attempt_lock:
            self._attempt = {"url": url, "request_id": request_id, "peer": peer}
        self._attempt_stop.clear()
        threading.Thread(
            target=self._poll_loop, args=(url, request_id, peer), name="pairing-poll", daemon=True
        ).start()
        log.info("pairing: offered pairing with %s; waiting for approval", url)
        self._notify()

    def cancel(self) -> None:
        """Abort an in-flight attempt."""
        self._attempt_stop.set()
        with self._attempt_lock:
            if self._attempt is not None:
                self._attempt = None
                self._message = ""
        self._notify()

    def remove(self, url: str) -> None:
        """Operator-side removal: drop the stored credentials and ask that
        monbooru to drop its side too (one click tears down both ends, like
        monloader)."""
        url = url.rstrip("/")
        entry = self.entry_for_url(url)
        token = entry.token if entry else ""
        self._drop_entry(url)
        self._message = ""
        if token:
            try:
                resp = self.http.post(
                    f"{url}/api/v1/pair/remove",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if resp.status_code >= 400:
                    self._message = (
                        f"removed locally; monbooru {url} could not be reached - remove the pairing there by hand"
                    )
            except httpx.TransportError:
                self._message = (
                    f"removed locally; monbooru {url} could not be reached - remove the pairing there by hand"
                )
        self._notify()

    def set_self_url(self, url: str) -> None:
        """Hot-update the address monbooru calls us back at."""
        self.self_url = url

    # ---- internals ------------------------------------------------------

    def _poll_loop(self, url: str, request_id: str, peer: str) -> None:
        """Poll pair/status until the offer is approved, expires or is
        denied, or the operator cancels."""
        while not self._attempt_stop.is_set():
            try:
                poll = self.http.get(f"{url}/api/v1/pair/status?id={request_id}")
            except httpx.TransportError:
                self._finish_attempt(url, "monbooru unreachable; the attempt was dropped, try again")
                return
            if poll.status_code == 404:
                self._finish_attempt(url, "the pairing offer expired; connect again")
                return
            if poll.status_code >= 400:
                self._attempt_stop.wait(POLL_INTERVAL)
                continue
            answer = poll.json()
            token = answer.get("token") or ""
            if token:
                with self._lock:
                    self._entries = [e for e in self._entries if e.url != url]
                    self._entries.append(
                        PairingEntry(
                            url=url,
                            token=token,
                            peer=peer,
                            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        )
                    )
                    self._save(self._entries)
                self._finish_attempt(url, "")
                log.info("paired with %s", url)
                return
            if answer.get("status") == "denied":
                self._finish_attempt(url, "the pairing was denied in monbooru")
                return
            self._attempt_stop.wait(POLL_INTERVAL)
        # cancelled: cancel() already cleared the attempt

    def _finish_attempt(self, url: str, message: str) -> None:
        """Clear the attempt only if it is still the one this poller owns."""
        with self._attempt_lock:
            att = self._attempt
            if att is None or att.get("url") != url:
                return  # cancelled or superseded
            self._attempt = None
        self._message = message  # an empty message clears stale ones
        self._notify()

    def _drop_entry(self, url: str) -> None:
        url = url.rstrip("/")
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.url != url]
            if len(self._entries) != before:
                self._save(self._entries)
        self._notify()

    def _notify(self) -> None:
        if self.on_change:
            self.on_change(self.paired())

    # ---- toml persistence ----------------------------------------------

    def _load(self) -> list[PairingEntry]:
        with TOML_LOCK:
            try:
                doc = tomllib.loads(self.config_path.read_text(encoding="utf-8"))
            except OSError:
                return []
            raw = doc.get("pairing") or []
            entries: list[PairingEntry] = []
            for item in raw:
                url = str(item.get("url", "")).strip().rstrip("/")
                if not url:
                    continue
                entries.append(
                    PairingEntry(
                        url=url,
                        token=str(item.get("token", "")),
                        peer=str(item.get("peer", "")),
                        created_at=str(item.get("created_at", "")),
                    )
                )
            return entries

    def _save(self, entries: list[PairingEntry]) -> None:
        with TOML_LOCK:
            path = self.config_path
            try:
                doc = tomlkit.parse(path.read_text(encoding="utf-8"))
            except OSError:
                doc = tomlkit.document()
            doc["pairing"] = _entries_to_toml(entries)
            path.write_text(tomlkit.dumps(doc), encoding="utf-8")