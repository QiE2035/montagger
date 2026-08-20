# montagger

Tagger plugin for monbooru, written in Python: a local ONNX tagging pipeline (FastAPI app + thread pool + SQLite), paired through the standard plugin contract. Full architecture, WebUI and durability design: [README.md](./README.md).

## Commands

- Setup: `uv sync --extra dev` (CPU). Provider extras are **mutually exclusive** (`dev` | `gpu` | `directml` | `openvino`) - install exactly one, never two provider packages.
- Run: `uv run montagger` (config file resolved from `--config/-c` > `MONTAGGER_CONFIG` > `./montagger.toml`; every key works as env var or CLI flag).
- Tests: `uv run pytest` (`testpaths = tests`).

## Architecture

- Single process. FastAPI lifespan assembles `Store` → `MonbooruClient` → backend → `Pipeline` → `Pairing` in `web/app.py` `create_app`; shutdown drains the pipeline gracefully.
- Backends register via `register(name)` in `backends/__init__.py` and implement `preprocess`/`tag`; the pipeline knows only the ABC. `heuristic` backend = no-model smoke test.
- Settings are pydantic-settings: CLI > init > env (`MONTAGGER_*`, `MONBOORU_URL` alias) > TOML > defaults. Hot-tunable fields are mirrored into the thread-safe `RuntimeState` (ep, thresholds, general_topk, backend, model_dir, activation, window, thread counts, skip_tagged); other hot fields (`monbooru`, `url`, `via`, `log_level`, `webui_token`) are applied directly to the live objects (`Pairing.set_base_url`, `Pipeline.set_via`, logger level, settings attribute). The WebUI patches `write_back` to `montagger.toml` via tomlkit (comments survive) and keeps the `settings` object in sync so templates and `/health` stay truthful. Keep `write_back` in mind when adding settings.
- Memory bound: images exist only as `BytesIO` bytes plus a short-lived PIL image. `preprocess` must not keep a reference to the PIL image - the pipeline drops it right after. `window` caps inflight work.

## External contracts (easy to get wrong)

- **Relay**: `POST /relay/tag` must answer within 10 s - enqueue and reply immediately, never tag synchronously (`tagging.py` `relay_answer`).
- **Pairing**: manual, like monloader - `start()` only restores stored credentials, the operator clicks connect in the WebUI (`pairing.py`). The offer is sent once, then a poll thread waits for approval; credentials live in `<state>/credentials.json` (0600). API calls authenticate with the token; inbound requests present the peer secret (`require_peer`). On 401/403 drop credentials and never re-offer silently (`MonbooruAuthError`) - the operator connects again. Changing the monbooru url keeps stored credentials (re-pair if it points to a different instance).
- **monbooru API**: `GET /api/v1/images/{id}/file` (fetch bytes into memory), `POST /api/v1/images/{id}/tags` with `{"tags", "via"}`; category names reconciled against `/api/v1/categories` when the backend loads. Retry only transient failures (408/429/5xx) with backoff; raise on other 4xx (`client.py`).

## Conventions & pitfalls

- Store: one locked write connection; WebUI requests use short-lived read connections per request. Schema migrations ride on `PRAGMA user_version`.
- DirectML EP is not thread-safe → `effective_workers()` forces 1 worker. Hot switches (ep, backend, model_dir, activation) rebuild the ONNX session via `reload()` (falling back to CPU when the provider is not installed); switching `backend` swaps the whole backend object through `Pipeline.set_backend`, which closes the old one.
- SSE cannot send headers: `/api/stream` accepts the token via `?token=` (EventSource).
- WebUI is open by default (no `webui_token`): pages and SSE are public, state changes are guarded by a per-instance CSRF token embedded in the page (`auth.py`); with a `webui_token` configured, browser routes additionally require the login session cookie (or the token).
- Name threads (`threading.Thread(name=...)`), start every module with a docstring, use `from __future__ import annotations`.
- Tests use a real `Store` (tmp file) with fake client/backend so the threading is exercised honestly - keep that style, don't reach for mocks.