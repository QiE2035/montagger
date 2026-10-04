# montagger

Push the photos on your phone to this PC and tag them with local AI models.

montagger is a companion for [monbooru](https://github.com/monbooru/monbooru): it runs the same ONNX taggers (WD14 SwinV2, EVA02, JoyTag, Camie) that monbooru's built-in tagger uses, as a small standalone service. Open it from your phone's browser, pick photos, and the tags come back grouped by category with confidences - ready to copy into any booru, or to push straight into a monbooru gallery.

<table>
  <tr>
    <td>手机浏览器 → <code>http://你的PC:8457</code></td>
    <td>多选照片 → 上传 → GPU/CPU 推理</td>
    <td>标签按类别分组，一键复制，可选回推 monbooru</td>
  </tr>
</table>

---

## Features

- **Push from anywhere on the LAN.** A mobile-first web UI (Vue 3, dark-first, installable as a PWA), a raw-byte HTTP API for iOS Shortcuts / Tasker / curl, and relay buttons inside monbooru once paired.
- **The same tags as monbooru.** The preprocessing, label parsing, category rules and dispatch tables are ported line-for-line from monbooru's tagger; the same model folder produces the same tags in both.
- **Shares models with monbooru.** Point `models.path` at monbooru's `paths.model_path` and both programs use one set of downloads. Or install from the built-in catalog straight from the UI.
- **Explicit execution providers.** Pick `cpu`, `cuda`, `directml`, `openvino` (or whatever your onnxruntime build offers) in settings. The value is verified against the installed build at startup - montagger refuses to start on a mismatch rather than silently falling back.
- **Queue with live progress.** Uploads land in an in-memory queue (bounded or unbounded, your choice); one inference worker serializes them (GPU or CPU); SSE pushes status to every open screen. The queue page has server-side pagination, search, per-job cancel and batch delete/cancel.
- **Multi-model tagging.** Configure any number of default models and every upload is tagged by each of them - results stay side by side, never merged. Pushes to monbooru keep the same separation: one create per file plus one enrich per extra model, each labeled `montagger/<model>`, and identical (file, model) pairs are never pushed twice.
- **Memory you control.** Models can idle-unload after N minutes, unload per-model from the UI, or flush everything with one button (releases ORT sessions, drops cached jobs, runs `malloc_trim`). `/health` reports resident memory so you can see it work. Inference runs in an isolated child process by default (`models.isolated`): onnxruntime never returns the CUDA context to the OS within a process, so terminating the worker is the only way a "release" can actually shrink RSS - unloading the last model (or the flush button) takes the whole child down.
- **Zero temp files.** Uploads stream as raw bytes and decode from memory; nothing touches the disk except metadata in SQLite and the model files you install.
- **Optional monbooru pairing.** The same handshake monloader speaks. After pairing: "AI 打标" relay buttons on monbooru's image page and batch bar, plus two independent auto-push switches - 推送标签 (on; finished results are written back to images monbooru already has, one source per model) and 推送图片 (off; direct uploads become new monbooru posts, tags riding along when 推送标签 is on).

## Requirements

Python 3.11+ (tested on 3.14) with the packages from `requirements.txt`:

```
pip install -r requirements.txt
```

For the web UI build: Node.js with **pnpm** (`make ui-install ui` - only needed when building from source; releases ship the built UI).

### onnxruntime distributions

Only one onnxruntime flavor can be installed at a time. Pick yours, then set `models.execution_provider` in `montagger.toml` to match:

| 环境 | pip 包 | execution_provider |
|---|---|---|
| 任意 CPU | `onnxruntime` | `cpu` |
| NVIDIA CUDA 12（Linux/Windows） | `onnxruntime-gpu` | `cuda` |
| Windows 任意 DX12 GPU | `onnxruntime-directml` | `directml` |
| Intel CPU/GPU/NPU | `onnxruntime-openvino` | `openvino` |

montagger verifies the configured provider against `onnxruntime.get_available_providers()` at startup. A mismatch is a hard error that lists what your build offers - it never downgrades behind your back.

## Quick start

```
pip install -r requirements.txt
python3 -m montagger
```

Open `http://localhost:8457`. First start writes a default `montagger.toml` next to the checkout; the Models page installs a tagger for you (wd-swinv2, ~330 MB).

For a mirror-friendly download (e.g. inside China) set:

```toml
[hf]
endpoint = "https://hf-mirror.com"
# token = "hf_..."   # only for gated repos (animetimm-eva02)
```

### From your phone

- **Browser**: open `http://<pc-ip>:8457`, add to home screen for an app icon.
- **iOS 快捷指令**: one "Get contents of URL" action -
  `POST http://<pc-ip>:8457/api/v1/tag?wait=true&filename=photo.jpg`
  header `Authorization: Bearer <token>` (generate one in Settings), body = the photo. The response JSON carries `tags` - one object with the legacy single-default setup, or one object per configured default model (plus `?model=a,b` to override; multi-model responses are arrays).
- **monbooru relay**: pair once (Settings → monbooru), then use the "AI 打标" button on any monbooru image.

### As a monbooru plugin

Copy this folder into monbooru's `<configdir>/plugins/` and start it from **Settings → Plugins**, then approve the pairing card. The Python dependencies must already be on the machine - monbooru provides no runtime.

## Model folders

```
<models path>/
└── wd-swinv2/
    ├── model.onnx
    ├── tags.csv            # or tags.txt / camie metadata json
    ├── tagger.json         # optional profile sidecar (same format as monbooru)
    └── dispatch.json       # optional per-tag category overrides (same format)
```

`models.json` in the model path root can add or replace catalog entries, exactly like monbooru.

## Documented differences from monbooru's built-in tagger

- **No inferred categories.** JoyTag outputs are single_general; monbooru re-files a name by how the operator tagged it elsewhere. montagger keeps no tag database, so those labels stay general.
- **Single frame.** Animated inputs tag their first frame (monbooru aggregates frames across manga archives).
- **Resize kernels.** Go's `x/image/draw.BiLinear` and PIL's `BILINEAR` are the same filter family but not byte-identical; scores can differ in the third decimal. Alpha is premultiplied first, matching Go's RGBA semantics.
- **No per-tag confidence over the API.** monbooru's create/enrich endpoints accept tag names but carry no confidence field - only its built-in tagger writes that column. Categories travel as `category:name` prefixes (filed into real categories by monbooru's `resolveCategoryTag`), and model provenance travels as the tag origin: enrich records source `montagger/<model>`, pushes record `via=montagger/<model>` as each tag's `tagger_name`.
- **Tags pushed by montagger are always manual (`is_auto=0`).** monbooru's public API has no path that writes `is_auto=1` - that column is only set by its built-in tagger writing the DB directly. montagger deliberately keeps its own tags (per-model sources, cross-model dedupe off); if you specifically want the "autotagged" look, run monbooru's built-in tagger on the image instead.

## Warning

> **Intended for local network use.** montagger's UI is not designed to be exposed to the public internet. Set a UI password in Settings before putting it anywhere questionable; the API speaks Bearer tokens.

## Development

```
make run        # backend on :8457 (serves web/dist when built)
make ui-install # pnpm install for the frontend
make ui         # build the frontend into web/dist
make ui-dev     # vite dev server with API proxy
make test       # pytest
```

The tagging pipeline never writes temporary files; the one exception is model installation (`<file>.part` → atomic rename), which exists so a crashed download can never damage an existing model.

## Acknowledgements

See [monbooru's acknowledgements](https://github.com/monbooru/monbooru#acknowledgements), and the model authors: SmilingWolf (WD14), animetimm (EVA02), fancyfeast (JoyTag), Camais03 (Camie).

## License

[AGPL-3.0](LICENSE)
