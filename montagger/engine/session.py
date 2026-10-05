"""onnxruntime session management.

The execution provider is an explicit configuration choice: the short name
from montagger.toml is matched against what the installed onnxruntime build
actually offers, and an unavailable pick is a hard error - montagger never
falls back or downgrades on its own.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import preprocess
from .dispatch import load_dispatch
from .labels import Label, load_labels
from .profile import Profile, resolve_profile

DEFAULT_MODEL_FILE = "model.onnx"
DEFAULT_TAGS_FILE = "tags.csv"
DEFAULT_TEXT_TAGS_FILE = "tags.txt"
_SIDECARS = frozenset(("tagger.json", "dispatch.json"))


class EngineError(Exception):
    """A model or provider problem the operator must fix."""


def evict_to_limit(runtimes: dict, last_use: dict, limit: int, unload) -> int:
    """Unload least-recently-used models until fewer than `limit` remain.

    Both session managers call this before loading one more model, so the
    configured residency cap holds. limit <= 0 disables the cap. `unload`
    is the manager's own invalidate(); it pops from both dicts as well,
    which this tolerates. Returns how many models were evicted."""
    if limit <= 0:
        return 0
    evicted = 0
    while len(runtimes) >= limit:
        victim = min(runtimes, key=lambda n: last_use.get(n, 0.0))
        unload(victim)
        runtimes.pop(victim, None)
        last_use.pop(victim, None)
        evicted += 1
    return evicted


def available_providers() -> list[str]:
    return list(ort.get_available_providers())


def resolve_provider(short_name: str) -> str:
    """Short config name → the onnxruntime provider it names, verified
    against the installed build. 'cpu' is always available."""
    from ..config import PROVIDER_NAMES

    wanted = PROVIDER_NAMES.get(short_name.strip().lower())
    if wanted is None:
        raise EngineError(
            f"unknown execution_provider {short_name!r}; known: "
            + ", ".join(sorted(PROVIDER_NAMES))
        )
    available = available_providers()
    if wanted not in available:
        raise EngineError(
            f"execution_provider {short_name!r} ({wanted}) is not in the installed "
            f"onnxruntime build; this build offers: {', '.join(available)}. "
            "Install the matching onnxruntime distribution and set the provider explicitly."
        )
    return wanted


def resolve_tagger_files(model_dir: Path) -> tuple[str, str]:
    """Port of tagger.resolveTaggerFiles: pick the model file and the label
    file when the folder holds several, else fall back to the defaults."""
    if not model_dir.is_dir():
        raise EngineError(f"model directory is missing: {model_dir}")
    onnx_files: list[str] = []
    label_files: list[str] = []
    has_csv = has_txt = False
    for entry in sorted(model_dir.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        ext = entry.suffix.lower()
        if ext == ".onnx":
            onnx_files.append(name)
        elif ext == ".csv":
            label_files.append(name)
            has_csv = has_csv or name == DEFAULT_TAGS_FILE
        elif ext == ".txt":
            label_files.append(name)
            has_txt = has_txt or name == DEFAULT_TEXT_TAGS_FILE
        elif ext == ".json" and name not in _SIDECARS:
            label_files.append(name)

    if has_csv:
        tags_file = DEFAULT_TAGS_FILE
    elif has_txt:
        tags_file = DEFAULT_TEXT_TAGS_FILE
    elif len(label_files) == 1:
        tags_file = label_files[0]
    else:
        tags_file = DEFAULT_TAGS_FILE

    if DEFAULT_MODEL_FILE in onnx_files or not onnx_files:
        model_file = DEFAULT_MODEL_FILE
    elif len(onnx_files) == 1:
        model_file = onnx_files[0]
    else:
        model_file = DEFAULT_MODEL_FILE
    return model_file, tags_file


def has_tagger_files(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    for entry in model_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in (".onnx", ".csv", ".txt"):
            return True
        if entry.is_file() and entry.suffix.lower() == ".json" and entry.name not in _SIDECARS:
            return True
    return False


@dataclass
class CandidateLabel:
    name: str
    category: str
    placeholder: bool


def build_candidates(profile: Profile, labels: list[Label], dispatch) -> list[CandidateLabel]:
    """Per-channel candidate labels: the category resolution pass shared by
    a loaded runtime and the GPU-free model_insights() path."""
    from .categories import resolve_category

    out: list[CandidateLabel] = []
    for idx, label in enumerate(labels):
        if label.placeholder:
            out.append(CandidateLabel(name=label.name, category="", placeholder=True))
            continue
        res = resolve_category(profile, label, dispatch)
        if res.skip:
            out.append(CandidateLabel(name=label.name, category="", placeholder=True))
            continue
        name = label.name
        if not res.override and profile.category_scheme == "single_general":
            # montagger keeps no tag database, so monbooru's inferred
            # categories stop here: the label stays general. Documented
            # divergence; joytag output is otherwise identical.
            pass
        rule = dispatch.lookup(label.name)
        if rule is not None and rule.name:
            name = rule.name
        out.append(CandidateLabel(name=name, category=res.category, placeholder=False))
    return out


def model_insights(model_dir: Path, name: str) -> list[str]:
    """Categories a model can emit, derived from profile + labels + dispatch
    alone - no onnxruntime session, no GPU touch."""
    model_file, tags_file = resolve_tagger_files(model_dir)
    tags_path = model_dir / tags_file
    if not (model_dir / model_file).exists() or not tags_path.exists():
        raise EngineError(f"model {name}: missing {model_file} or {tags_file}")
    profile = resolve_profile(model_dir, name, tags_file)
    labels = load_labels(tags_path, profile.label_format)
    dispatch = load_dispatch(model_dir, name)
    return sorted({c.category for c in build_candidates(profile, labels, dispatch) if not c.placeholder})


class ModelRuntime:
    """A loaded model: ORT session plus its labels, profile, dispatch, and
    the pre-resolved per-channel candidate labels."""

    def __init__(self, name: str, model_dir: Path, provider: str, device_id: int, intra_op_threads: int):
        self.name = name
        model_file, tags_file = resolve_tagger_files(model_dir)
        onnx_path = model_dir / model_file
        tags_path = model_dir / tags_file
        if not onnx_path.exists():
            raise EngineError(f"model {name}: missing {model_file}")
        if not tags_path.exists():
            raise EngineError(f"model {name}: missing {tags_file}")

        self.profile: Profile = resolve_profile(model_dir, name, tags_file)
        self.labels: list[Label] = load_labels(tags_path, self.profile.label_format)
        self.dispatch = load_dispatch(model_dir, name)

        opts = ort.SessionOptions()
        if intra_op_threads > 0:
            opts.intra_op_num_threads = intra_op_threads
        providers, provider_options = self._provider_setup(provider, device_id, opts)

        try:
            self.session = ort.InferenceSession(
                str(onnx_path), sess_options=opts, providers=providers, provider_options=provider_options
            )
        except Exception as err:
            raise EngineError(f"model {name}: session creation failed: {err}") from err

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if not inputs or not outputs:
            raise EngineError(f"model {name}: no input/output")
        if len(inputs) > 1:
            raise EngineError(f"model {name}: expected a single input, got {len(inputs)}")
        out_idx = self.profile.output_index
        if out_idx >= len(outputs):
            raise EngineError(
                f"model {name}: profile output_index {out_idx} out of range (have {len(outputs)})"
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[out_idx].name

        self.input_size = self.profile.input_size
        if self.input_size == 0:
            self.input_size = preprocess.infer_input_size(inputs[0].shape, self.profile.layout)
        if self.input_size <= 0:
            raise EngineError(
                f"model {name}: cannot infer input size from {inputs[0].shape}; "
                "set input_size in the tagger.json sidecar"
            )
        self.candidates = build_candidates(self.profile, self.labels, self.dispatch)

    @staticmethod
    def _provider_setup(provider: str, device_id: int, opts: ort.SessionOptions):
        """Session provider list and options, mirroring monbooru's EP tuning.
        CPU stays last as ORT's standard graph fallback; the accelerator
        itself was validated at startup and is never swapped."""
        if provider == "CPUExecutionProvider":
            # The CPU arena is a one-way growth ratchet; on a shared box it
            # reads as a leak. Metadata allocations are tiny here, so the
            # arena buys nothing worth keeping.
            opts.enable_cpu_mem_arena = False
            return [provider], []
        if provider == "CUDAExecutionProvider":
            try:
                opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
            except Exception:
                pass  # older builds: harmless to skip
            cuda_opts = {
                "device_id": str(device_id),
                # HEURISTIC skips cuDNN's multi-second exhaustive search,
                # kSameAsRequested grows the arena by the request instead of
                # doubling, and default-stream copies avoid a cross-stream sync.
                "cudnn_conv_algo_search": "HEURISTIC",
                "arena_extend_strategy": "kSameAsRequested",
                "do_copy_in_default_stream": "1",
            }
            return [provider, "CPUExecutionProvider"], [cuda_opts, {}]
        if provider == "DmlExecutionProvider":
            # DirectML requires the memory pattern off and sequential execution.
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            return [provider, "CPUExecutionProvider"], [{"device_id": str(device_id)}, {}]
        if provider == "OpenVINOExecutionProvider":
            return [provider, "CPUExecutionProvider"], [{"device_type": "GPU"}, {}]
        return [provider, "CPUExecutionProvider"], [{}, {}]

    def infer(self, data: bytes) -> tuple[np.ndarray, int, int]:
        """Bytes → (post-activation scores, width, height). Scores are float32
        per output channel, sigmoid applied for logits models."""
        img = preprocess.decode(data)
        width, height = img.width, img.height
        tensor = preprocess.build_tensor(preprocess.pad_and_resize(img, self.input_size, self.profile), self.input_size, self.profile)
        out = self.session.run([self.output_name], {self.input_name: tensor})[0]
        scores = np.asarray(out, dtype=np.float32).reshape(-1)
        if self.profile.activation == "logits":
            # Overflow-safe sigmoid, all float32.
            pos = scores >= 0
            result = np.empty_like(scores)
            result[pos] = 1.0 / (1.0 + np.exp(-scores[pos]))
            exp_x = np.exp(scores[~pos])
            result[~pos] = exp_x / (1.0 + exp_x)
            scores = result
        return scores, width, height


class SessionManager:
    """Caches ModelRuntime per model name; invalidated explicitly (after a
    model install), by the idle-unload sweep, or when the resolved profile
    fingerprint changes."""

    def __init__(self, provider: str, device_id: int = 0, intra_op_threads: int = 0):
        self.provider = resolve_provider(provider)  # fail fast at startup
        self.provider_short = provider
        self.device_id = device_id
        self.intra_op_threads = intra_op_threads
        # Residency cap; the Engine refreshes it from config on every tag so
        # a settings change takes effect without a restart. 0 = unlimited.
        self.max_loaded = 0
        self._runtimes: dict[str, ModelRuntime] = {}
        self._last_use: dict[str, float] = {}

    def get(self, model_dir: Path, name: str) -> ModelRuntime:
        runtime = self._runtimes.get(name)
        if runtime is not None:
            self._last_use[name] = time.monotonic()
            return runtime
        runtime = ModelRuntime(name, model_dir, self.provider, self.device_id, self.intra_op_threads)
        evict_to_limit(self._runtimes, self._last_use, self.max_loaded, self.invalidate)
        self._runtimes[name] = runtime
        self._last_use[name] = time.monotonic()
        return runtime

    def invalidate(self, name: str | None = None) -> None:
        if name is None:
            self._runtimes.clear()
            self._last_use.clear()
        else:
            self._runtimes.pop(name, None)
            self._last_use.pop(name, None)

    def loaded(self) -> list[str]:
        return list(self._runtimes)

    def idle_seconds(self, name: str) -> float | None:
        """Seconds since the model last served a request; None if not loaded."""
        ts = self._last_use.get(name)
        if ts is None:
            return None
        return time.monotonic() - ts


class RemoteRuntime:
    """Parent-side handle for a model served by the worker process.

    Carries everything the tagging pipeline needs locally (profile, labels,
    dispatch, candidates - all CPU-only, a few MB) and forwards only
    `infer` across the pipe, so scores are the sole data that ever comes
    back and no accelerator state lives in this process.
    """

    def __init__(self, name: str, model_dir: Path, input_size: int):
        self.name = name
        self._send = None  # wired by WorkerSessionManager
        model_file, tags_file = resolve_tagger_files(model_dir)
        onnx_path = model_dir / model_file
        tags_path = model_dir / tags_file
        if not onnx_path.exists():
            raise EngineError(f"model {name}: missing {model_file}")
        if not tags_path.exists():
            raise EngineError(f"model {name}: missing {tags_file}")
        self.profile: Profile = resolve_profile(model_dir, name, tags_file)
        self.labels: list[Label] = load_labels(tags_path, self.profile.label_format)
        self.dispatch = load_dispatch(model_dir, name)
        self.candidates = build_candidates(self.profile, self.labels, self.dispatch)
        self.input_size = input_size

    def infer(self, data: bytes) -> tuple[np.ndarray, int, int]:
        """Bytes in, post-activation scores out - computed in the worker."""
        assert self._send is not None, "RemoteRuntime not wired to a manager"
        reply = self._send({"op": "infer", "model": self.name}, payload=data)
        scores = np.frombuffer(
            base64.b64decode(reply["scores_b64"]), dtype=np.float32
        ).reshape(-1)
        return scores, int(reply["width"]), int(reply["height"])


class WorkerSessionManager:
    """SessionManager twin that owns one child process holding every ORT
    session. API-compatible with SessionManager; additionally offers
    child_rss_mb() for /health and truly frees accelerator memory when the
    child is torn down (invalidate(None) or manager destruction)."""

    def __init__(self, provider: str, device_id: int = 0, intra_op_threads: int = 0):
        self.provider = resolve_provider(provider)  # fail fast at startup
        self.provider_short = provider
        self.device_id = device_id
        self.intra_op_threads = intra_op_threads
        # Residency cap; see SessionManager. 0 = unlimited.
        self.max_loaded = 0
        self._runtimes: dict[str, RemoteRuntime] = {}
        self._last_use: dict[str, float] = {}
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # -- child plumbing ----------------------------------------------------------

    def _ensure_child(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        if self._proc is not None:
            self._runtimes.clear()  # the child took its models with it
        self._proc = subprocess.Popen(
            [
                sys.executable, "-m", "montagger.engine.worker",
                self.provider_short, str(self.device_id), str(self.intra_op_threads),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # worker logs join our stream
        )
        return self._proc

    def _request(self, msg: dict, payload: bytes | None = None) -> dict:
        from . import worker  # lazy: worker.py imports this module back

        proc = self._ensure_child()
        wire = dict(msg)
        if payload is not None:
            wire["data_b64"] = base64.b64encode(payload).decode("ascii")
        with self._lock:
            try:
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(worker.encode_frame(wire))
                proc.stdin.flush()
                reply = worker.read_frame(proc.stdout)
            except (BrokenPipeError, EOFError, OSError) as err:
                self._proc = None
                raise EngineError(f"inference worker died: {err}") from err
        if reply is None:
            self._proc = None
            raise EngineError("inference worker exited unexpectedly")
        if not reply.get("ok"):
            raise EngineError(str(reply.get("error", "worker request failed")))
        return reply

    # -- SessionManager API --------------------------------------------------------

    def get(self, model_dir: Path, name: str) -> RemoteRuntime:
        runtime = self._runtimes.get(name)
        if runtime is None:
            # Mirror in-process timing: file problems raise here, before any
            # queue work starts.
            probe = RemoteRuntime(name, model_dir, input_size=0)
            evict_to_limit(self._runtimes, self._last_use, self.max_loaded, self.invalidate)
            reply = self._request({"op": "load", "model": name, "dir": str(model_dir)})
            probe.input_size = int(reply.get("input_size") or probe.profile.input_size)
            if probe.input_size <= 0:
                raise EngineError(f"model {name}: cannot resolve input size")
            probe._send = self._request
            runtime = probe
            self._runtimes[name] = runtime
        self._last_use[name] = time.monotonic()
        return runtime

    def invalidate(self, name: str | None = None) -> None:
        if name is None:
            self._runtimes.clear()
            self._last_use.clear()
            self._kill_child()
            return
        self._runtimes.pop(name, None)
        self._last_use.pop(name, None)
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._request({"op": "unload", "model": name})
            except EngineError:
                pass  # a dead worker is as good as an unloaded one

    def _kill_child(self) -> None:
        from . import worker

        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(worker.encode_frame({"id": 0, "op": "quit"}))
                proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def loaded(self) -> list[str]:
        return list(self._runtimes)

    def idle_seconds(self, name: str) -> float | None:
        ts = self._last_use.get(name)
        if ts is None:
            return None
        return time.monotonic() - ts

    def child_rss_mb(self) -> float:
        """Resident memory of the worker process (0 if it is not running)."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return 0.0
        try:
            with open(f"/proc/{proc.pid}/statm", "rb") as fh:
                resident = int(fh.read().split()[1])
            return resident * os.sysconf("SC_PAGE_SIZE") / (1000 * 1000)
        except (OSError, IndexError, ValueError):
            return 0.0

    def __del__(self):
        try:
            self._kill_child()
        except Exception:
            pass
