# montagger

A tagger plugin for [monbooru](https://github.com/monbooru/monbooru), written in Python.

montagger runs as its own process, pairs with monbooru like any other plugin (Settings > Plugins), and adds two relay buttons - *tag with montagger* on the image detail page and in the gallery batch bar. Clicking one pushes the selection to montagger, which downloads the images in memory (nothing is written to disk except the task database), runs a local ONNX model and writes the tags back through the monbooru API (`via: montagger`).

A WebUI (dashboard + settings) shows the live queue, throughput, results with paging, and lets you hot-swap the execution provider, thresholds and thread counts.

```
monbooru button click ──> relay 10s  ──> pipeline (prefetch window + inference workers)
                                            │ download in memory (BytesIO)
                                            ├─ preprocess (model-specific)
                                            ├─ ONNX infer (CPU / CUDA / DirectML / OpenVINO)
                                            └─ POST /api/v1/images/{id}/tags  via=montagger
                                          SQLite: task list + results (resume-safe)
```

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- An ONNX tagger model (WD14 family: `wd-swinv2`, `camie-v2`, `animetimm-eva02`; joytag works too). You can reuse the models monbooru already downloaded: point `model_dir` at a subfolder of its `model_path`, e.g. `E:\Data\Cache\Monbooru\models\wd-swinv2` (a `.onnx` plus a `.csv` or `.txt` tag file per folder).

## Install

```sh
uv sync --extra dev            # CPU provider (default)
uv sync --extra gpu            # CUDA (onnxruntime-gpu replaces the CPU package)
uv sync --extra directml       # DirectML
uv sync --extra openvino       # OpenVINO
```

The provider packages are mutually exclusive - install only the one you want. The WebUI only offers execution providers the installed package actually has.

## Configure

Copy `montagger.toml.example` to `montagger.toml` and edit it. Every key can also be set via an environment variable (`MONTAGGER_<KEY>`, and the monbooru address accepts the ecosystem alias `MONBOORU_URL`) or a command line flag (`montagger --help` shows them, along with the matching env var names). Precedence: CLI > environment > montagger.toml > defaults. The WebUI writes hot settings back into the TOML file (comments survive).

Reasonable starting point with the monbooru models:

```toml
addr = "127.0.0.1:8301"
monbooru = "http://127.0.0.1:8080"
model_dir = "E:\\Data\\Cache\\Monbooru\\models\\wd-swinv2"
backend = "onnx"
ep = "directml"      # or cpu / cuda / openvino
threshold = 0.35
window = 16
```

## Run

```sh
uv run montagger
```

1. Open `http://127.0.0.1:8301` (or reach it through monbooru's plugin pages once paired).
2. monbooru Settings > Plugins: approve the `montagger` pairing request. The buttons appear on the image detail page and the batch bar.
3. Select images and click *tag with montagger*. The relay answers in milliseconds (the ids are queued); the WebUI follows the progress live.

For a no-model smoke test keep `backend = "heuristic"` - it derives simple tags (portrait/landscape, bright/dark, grayscale/colorful, size) from the image itself.

## WebUI

- **dashboard** `/`: live counts (pending / processing / done / failed), progress bar, throughput and ETA, an operations bar (pause, retry failed, clear results/tasks) and a paged results table (50 per page, filter all/done/failed). Live updates come over SSE (htmx-sse); the pair light in the top bar polls.
- **settings** `/settings`: hot-tunable values are applied immediately and written back to `montagger.toml` - execution provider (builds a new ONNX session on switch), thresholds, inflight window, prefetch/inference thread counts, general tag cap, skip-already-tagged. Values that need a restart are shown read-only with a badge.

Access: routes require the pairing secret (used automatically when the page is opened through monbooru's plugin proxy), or `webui_token` when you connect directly. The SSE stream takes the token as a query parameter (`EventSource` cannot send headers); keep montagger on localhost.

## How the pipeline works

Images sit in memory only:

```
tasks (deque, unbounded) -> [window slots] -> ready queue -> inference workers
```

- The **inflight window** (default 16) bounds how many images are at any moment in fetch + preprocess + ready + infer, so memory is capped, and download, preprocess and inference overlap - the model never waits on the network.
- A prefetch thread downloads the image (`GET /api/v1/images/{id}/file`), decodes it with Pillow from a `BytesIO` and preprocesses it (model dependent); the raw image is dropped right after, releasing the memory.
- An inference worker runs the ONNX session, then writes the tags (`POST /api/v1/images/{id}/tags`, transient failures retried twice with backoff). Only when the write lands is the window slot released.
- Single images failing (download, decode, model, write) are marked `failed` in the database; the batch continues. `retry failed` re-queues them.
- DirectML forces a single inference worker (the EP is not thread-safe).

### Durability

The SQLite database (`<state>/montagger.db`) keeps the task list and all results. Restarting montagger re-enqueues pending/failed/interrupted tasks (`resume = true`), so a 50k-image run survives restarts - finished images are never re-processed. Results page through SQL; all 50k stay visible.

### Pairing & credentials

montagger follows the standard plugin contract (identical to the simple-edit reference): it offers a pairing with `app = montagger` and scopes `read` + `write`, monbooru presents the peer secret on every inbound request (relay clicks, our pages), and we authenticate API calls with the token from the pairing. Credentials live in `<state>/credentials.json` (0600). If the API starts answering 401/403 the credentials are dropped and a fresh pairing is offered; when you remove the pairing in monbooru, montagger offers again on its own.

## Tests

```sh
uv run pytest
```

## Notes

- Tags written by montagger are ordinary monbooru tags with `via = montagger`; re-tagging is idempotent.
- Category names (rating/character/copyright/artist) map from the WD14 tag prefixes and are reconciled against `/api/v1/categories` when the model loads; a category monbooru does not know falls back to general.
- No image data is ever written to disk; only status records.