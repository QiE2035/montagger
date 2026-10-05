"""monbooru pairing and the relay / push-back integration.

The handshake is the one simple-edit and monloader speak: the plugin offers
itself (POST /api/v1/pair/request with its buttons and a peer secret),
monbooru shows the approval card, and the issued API token comes back
through the status poll. monbooru then presents the peer secret on every
call it makes into the plugin, and the plugin presents its token on every
call into monbooru.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import logx
from .config import Config, Provider

log = logx.get("pair")

APP_NAME = "montagger"

# Declared at pairing; the operator approves this exact list. A relay click
# gets 10 seconds, so the handler only enqueues and answers.
BUTTONS = [
    {"slot": "detail-actions", "label": "AI 打标", "mode": "relay", "path": "/api/v1/relay/tag", "media": "image"},
    {"slot": "batch-bar", "label": "AI 打标", "mode": "relay", "path": "/api/v1/relay/tag", "media": "image"},
]


@dataclass
class Credentials:
    token: str  # monbooru issued; authenticates our API calls
    peer: str  # our secret; monbooru presents it on calls into us


class PairError(Exception):
    pass


def generate_secret() -> str:
    return secrets.token_hex(32)


class MonbooruClient:
    """REST calls into monbooru with the paired token."""

    def __init__(self, cfg: Provider):
        self.cfg = cfg

    def _base(self) -> str:
        return self.cfg.current().monbooru.api_url.rstrip("/")

    def _token(self) -> str:
        return self.cfg.current().monbooru.token

    def call(self, method: str, path: str, *, json_body=None, files=None, data=None, token: str = "") -> tuple[int, dict | bytes]:
        headers = {}
        bearer = token or self._token()
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        url = self._base() + path
        timeout = float(self.cfg.current().monbooru.timeout_s)
        try:
            resp = httpx.request(
                method, url, json=json_body, files=files, data=data,
                headers=headers, timeout=timeout,
            )
        except httpx.HTTPError as err:
            raise PairError(f"monbooru unreachable: {err}") from err
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            return resp.status_code, resp.json()
        return resp.status_code, resp.content

    # -- pairing ---------------------------------------------------------

    def pair_request(self, api_url: str, peer: str, self_url: str) -> str:
        status, body = self.call(
            "POST", "/api/v1/pair/request",
            json_body={
                "app": APP_NAME,
                "url": self_url,
                "requested_scopes": ["read", "write"],
                "peer_token": peer,
                "version": _version(),
                "buttons": BUTTONS,
            },
            token="",
        )
        if status != 200 or not isinstance(body, dict) or not body.get("request_id"):
            raise PairError(f"pairing request failed: HTTP {status}")
        return str(body["request_id"])

    def pair_status(self, api_url: str, request_id: str) -> tuple[str, str]:
        """(status, token); token non-empty means approved."""
        status, body = self.call("GET", f"/api/v1/pair/status?id={request_id}", token="")
        if status == 404:
            return "expired", ""
        if status != 200 or not isinstance(body, dict):
            return "error", ""
        token = str(body.get("token", ""))
        if token:
            return "approved", token
        return str(body.get("status", "pending")), ""

    def token_accepted(self) -> bool:
        try:
            status, _ = self.call("GET", "/api/v1/galleries")
        except PairError:
            return False
        return status not in (401, 503)

    def pair_remove(self) -> None:
        try:
            self.call("POST", "/api/v1/pair/remove", json_body={})
        except PairError as err:
            log.warning("pair/remove failed: %s", err)

    # -- images ------------------------------------------------------------

    def get_file(self, image_id: int) -> bytes:
        status, body = self.call("GET", f"/api/v1/images/{image_id}/file")
        if status != 200 or isinstance(body, dict):
            raise PairError(f"fetch image {image_id}: HTTP {status}")
        return body

    def enrich(self, image_id: int, tags: list[str], model: str = "") -> dict:
        """Apply tagged names to an existing image.

        Tags carry their `category:name` prefix (monbooru's
        resolveCategoryTag files them into real categories), and the source
        string - `montagger/<model>` - becomes the tags' origin on
        monbooru's side, so the model that produced them stays readable.
        """
        source = f"{APP_NAME}/{model}" if model else APP_NAME
        status, body = self.call(
            "POST", f"/api/v1/images/{image_id}/enrich",
            json_body={"tags": tags, "source": source, "verify": False},
        )
        if status != 200:
            raise PairError(f"enrich {image_id}: HTTP {status}")
        return body if isinstance(body, dict) else {}

    def report_fetch(self, image_id: int, state: str, message: str) -> None:
        try:
            self.call("POST", f"/api/v1/images/{image_id}/fetch-status", json_body={"state": state, "message": message[:500]})
        except PairError as err:
            log.warning("fetch-status %d: %s", image_id, err)

    def galleries(self) -> list[dict]:
        status, body = self.call("GET", "/api/v1/galleries")
        if status != 200 or not isinstance(body, dict):
            return []
        return list(body.get("galleries", []))

    def push_image(
        self,
        data: bytes,
        filename: str,
        tags: list[str],
        gallery: str,
        model: str = "",
    ) -> int:
        """Push one tagged file as a new monbooru image; returns its id.

        Tags carry the `category:name` prefix; `via=montagger/<model>` is
        what monbooru records as each tag's tagger_name provenance. No
        source field: a fake site row would misread as a booru origin.
        """
        provenance = f"{APP_NAME}/{model}" if model else APP_NAME
        fields = {
            "tags": json.dumps(tags),
            "via": provenance,
        }
        status, body = self.call(
            "POST", f"/api/v1/images?gallery={gallery}" if gallery else "/api/v1/images",
            files={"file": (filename or "upload", data)},
            data=fields,
        )
        if status not in (200, 201) or not isinstance(body, dict):
            detail = body.get("detail", "") if isinstance(body, dict) else ""
            raise PairError(f"push {filename}: HTTP {status} {detail}")
        return int(body.get("id") or 0)


def wire_tags(result_tags) -> list[str]:
    """ScoredTag list → the names monbooru receives: `category:name`, which
    its resolveCategoryTag turns into real categories. A tag name that
    itself contains a colon survives - only the first colon is the split -
    and a category monbooru does not know falls back to general, never to a
    lost tag."""
    return [f"{t.category}:{t.name}" for t in result_tags]


def wire_tags_from_rows(rows: list[dict]) -> list[str]:
    """Stored tag dicts (tags_json) → the same `category:name` wire form."""
    out = []
    for t in rows:
        name = str(t.get("name", "")).strip()
        if not name:
            continue
        category = str(t.get("category", "")).strip()
        out.append(f"{category}:{name}" if category else name)
    return out


def resolve_models(cfg: Config) -> list[str]:
    """Models a relay retag runs: configured default_models, else
    [default]. Empty means nothing is configured. Matches the API's
    resolve_model_list: blanks dropped, duplicates collapsed."""
    names: list[str] = []
    for raw in cfg.models.default_models:
        name = raw.strip()
        if name and name not in names:
            names.append(name)
    if names:
        return names
    default = cfg.models.default.strip()
    return [default] if default else []


def _version() -> str:
    from . import __version__

    return __version__


class Integration:
    """Owns the credentials file, the offer loop, and the retag tasks."""

    def __init__(self, cfg: Provider, state_path: Path, runner):
        self.cfg = cfg
        self.state_path = state_path
        self.runner = runner
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._creds: Credentials | None = None
        self._waiting = threading.Event()  # an offer is on the table
        self._tasks: set[asyncio.Task] = set()
        # Relay pipeline width (monbooru.relay_concurrency, fixed at
        # startup): one slot per in-flight image - download, queue, tag,
        # enrich - so a push of any size streams through at that width
        # instead of pulling every image's bytes into RAM at once.
        width = max(1, int(self.cfg.current().monbooru.relay_concurrency))
        self._relay_gate = asyncio.Semaphore(width)
        # Dedicated fetch threads: concurrent downloads must not occupy the
        # default executor that inference runs on.
        self._fetch_pool = ThreadPoolExecutor(
            max_workers=min(8, width), thread_name_prefix="monbooru-fetch"
        )
        self._thread: threading.Thread | None = None
        self._load()

    # -- credentials ---------------------------------------------------------

    @property
    def creds(self) -> Credentials | None:
        with self._lock:
            return self._creds

    @property
    def paired(self) -> bool:
        return self.creds is not None

    @property
    def waiting(self) -> bool:
        return self._waiting.is_set()

    def _load(self) -> None:
        try:
            doc = json.loads(self.state_path.read_text(encoding="utf-8"))
            creds = Credentials(token=str(doc.get("token", "")), peer=str(doc.get("peer", "")))
            if creds.token and creds.peer:
                with self._lock:
                    self._creds = creds
        except (OSError, ValueError):
            pass

    def _store(self, creds: Credentials) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.part")
        tmp.write_text(json.dumps({"token": creds.token, "peer": creds.peer}), encoding="utf-8")
        tmp.replace(self.state_path)
        with self._lock:
            self._creds = creds

    def forget(self) -> None:
        with self._lock:
            self._creds = None
        try:
            self.state_path.unlink()
        except OSError:
            pass

    # -- offer loop -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="pairing", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._fetch_pool.shutdown(wait=False)

    def kick(self) -> None:
        """Config changed (api_url or manual re-pair): wake the loop."""
        self._wake.set()

    def self_url(self) -> str:
        bind = self.cfg.current().server.bind_address
        _, _, port = bind.rpartition(":")
        return f"http://localhost:{port or '8457'}"

    def _loop(self) -> None:
        while not self._stop.is_set():
            api_url = self.cfg.current().monbooru.api_url.strip()
            if not api_url:
                self._waiting.clear()
                self._wake.wait(5)
                self._wake.clear()
                continue
            creds = self.creds
            if creds is not None:
                client = MonbooruClient(self.cfg)
                saved_token = self.cfg.current().monbooru.token
                if client.token_accepted() or not saved_token:
                    # Healthy pairing (or the token only lives in our file):
                    # idle until the config or a teardown wakes us.
                    self._waiting.clear()
                    self._wake.wait(30)
                    self._wake.clear()
                    continue
                log.info("monbooru no longer accepts the stored credentials; offering again")
                self.forget()
                continue
            try:
                self._offer_once(api_url)
            except PairError as err:
                log.info("pairing offer failed (%s); retrying in 5s", err)
                self._waiting.clear()
                self._wake.wait(5)
                self._wake.clear()

    def _offer_once(self, api_url: str) -> None:
        client = MonbooruClient(self.cfg)
        peer = generate_secret()
        request_id = client.pair_request(api_url, peer, self.self_url())
        self._waiting.set()
        log.info("pairing offer sent; approve it in monbooru: Settings > Plugins")
        while not self._stop.is_set():
            status, token = client.pair_status(api_url, request_id)
            if status == "approved" and token:
                self._store(Credentials(token=token, peer=peer))
                self._waiting.clear()
                # Keep monbooru's copy of the token in the config too, so
                # push-back works even though our file already holds one.
                try:
                    self.cfg.update(lambda c: setattr(c.monbooru, "token", token))
                except Exception:
                    pass
                log.info("paired with monbooru")
                return
            if status in ("denied", "expired", "error"):
                self._waiting.clear()
                if status == "denied":
                    log.info("the pairing was denied; offering again in 10s")
                    self._wake.wait(10)
                    self._wake.clear()
                return
            self._stop.wait(2)

    # -- inbound guard -----------------------------------------------------------

    def request_from_monbooru(self, request) -> bool:
        """True when the bearer matches the peer secret monbooru holds."""
        creds = self.creds
        if creds is None or not creds.peer:
            return False
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer "):
            return False
        return secrets.compare_digest(header[7:].strip(), creds.peer)

    # -- relay ---------------------------------------------------------------

    def schedule_retag(self, image_ids: list[int]) -> int:
        """Fire-and-forget retag of monbooru images; each task pulls the
        file, queues it for inference, and enriches or reports back."""
        loop = asyncio.get_running_loop()
        count = 0
        for image_id in image_ids:
            task = loop.create_task(self._retag_one(int(image_id)))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            count += 1
        return count

    async def _retag_one(self, image_id: int) -> None:
        """One relay slot covers the whole pipeline - fetch, queue, tag,
        enrich - so a push of any size streams through at the configured
        width instead of pulling every image's bytes into RAM at once."""
        async with self._relay_gate:
            await self._retag_one_locked(image_id)

    async def _retag_one_locked(self, image_id: int) -> None:
        """Fetch once, tag with EVERY configured model, and enrich each
        model's result as its own source (`montagger/<model>`) - the tag
        sets never merge on monbooru's side, so each model's read of the
        image stays inspectable on its own."""
        from .queue import Job
        from .store import DONE, new_job_id

        runner = self.runner
        client = MonbooruClient(self.cfg)
        try:
            data = await asyncio.get_running_loop().run_in_executor(
                self._fetch_pool, client.get_file, image_id
            )
            sha256 = hashlib.sha256(data).hexdigest()
            md5 = hashlib.md5(data).hexdigest()
            models = resolve_models(self.cfg.current())
            if not models:
                client.report_fetch(image_id, "error", "no model configured")
                return
            total = 0
            for model in models:
                # Same-model dedup first: a stored result enriches without
                # touching the accelerator.
                prior = runner.store.find_by_hash(sha256, model)
                if prior is not None and prior.status == DONE:
                    import json as _json

                    tags = wire_tags_from_rows(_json.loads(prior.tags_json or "[]"))
                    if tags:
                        await asyncio.to_thread(client.enrich, image_id, tags, model)
                        total += len(tags)
                    continue
                job = Job(
                    id=new_job_id(),
                    filename=f"monbooru #{image_id}",
                    model=model,
                    source="monbooru",
                    data=data,
                    sha256=sha256,
                    md5=md5,
                )
                runner.store.create_job(
                    job_id=job.id, filename=job.filename, source=job.source,
                    model=model, sha256=sha256, md5=md5, bytes_len=len(data),
                )
                await runner.submit(job)
                await job.done.wait()
                if job.status == DONE and job.result is not None:
                    tags = wire_tags(job.result.tags)
                    await asyncio.to_thread(client.enrich, image_id, tags, model)
                    runner.store.set_pushed(job.id, image_id)
                    total += len(tags)
                else:
                    client.report_fetch(image_id, "error", job.error or f"{model}: tagging failed")
                    return
            log.info(
                "monbooru #%d tagged with %d tag(s) across %d model(s)",
                image_id, total, len(models),
            )
        except Exception as err:
            log.warning("retag monbooru #%d failed: %s", image_id, err)
            client.report_fetch(image_id, "error", str(err))
