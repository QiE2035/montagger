"""Model profile: how a model's input must be built and its output read.

Field semantics match monbooru's tagger.Profile one for one; a sidecar
profile.json next to the model overrides any subset of the embedded
default, so the previous resolution layer fills the rest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Profile:
    input_size: int = 0  # 0 reads the size from the model's input
    layout: str = ""  # nhwc | nchw
    channels: str = ""  # rgb | bgr
    normalize: str = ""  # none | div255 | imagenet | clip
    pad: str = ""  # white_square | mean_color_aspect
    fill_color: tuple = (0, 0, 0)  # mean_color_aspect fill; camie default when zero
    activation: str = ""  # sigmoid_in_model | logits
    label_format: str = ""  # wd14_csv | joytag_txt | camie_json
    category_scheme: str = ""  # wd14_numeric | single_general | name_string
    output_index: int = 0  # camie's refined head and eva02's head are not output 0

    def fingerprint(self) -> str:
        return repr(  # cheap identity for session cache keys
            (
                self.input_size,
                self.layout,
                self.channels,
                self.normalize,
                self.pad,
                self.fill_color,
                self.activation,
                self.label_format,
                self.category_scheme,
                self.output_index,
            )
        )

    def validate(self) -> None:
        _check(self.layout, ("nhwc", "nchw"), "layout")
        _check(self.channels, ("rgb", "bgr"), "channels")
        _check(self.normalize, ("none", "div255", "imagenet", "clip"), "normalize")
        _check(self.pad, ("white_square", "mean_color_aspect"), "pad")
        _check(self.activation, ("sigmoid_in_model", "logits"), "activation")
        _check(self.label_format, ("wd14_csv", "joytag_txt", "camie_json"), "label_format")
        _check(self.category_scheme, ("wd14_numeric", "single_general", "name_string"), "category_scheme")
        if self.input_size < 0:
            raise ValueError(f"bad input_size {self.input_size}")
        if self.output_index < 0:
            raise ValueError(f"bad output_index {self.output_index}")


def _check(value: str, allowed: tuple, axis: str) -> None:
    if value not in allowed:
        raise ValueError(f"bad profile {axis} {value!r} (want one of {', '.join(allowed)})")


_DATA = Path(__file__).parent / "data"

# Heuristic profiles keyed by the tags file extension, ported from
# profile.go: the extension guesses the family before the embedded default
# and any sidecar refine it.
_WD14_PROFILE = dict(
    layout="nhwc", channels="bgr", normalize="none", pad="white_square",
    activation="sigmoid_in_model", label_format="wd14_csv", category_scheme="wd14_numeric",
)
_JOYTAG_PROFILE = dict(
    layout="nchw", channels="rgb", normalize="clip", pad="white_square",
    activation="logits", label_format="joytag_txt", category_scheme="single_general",
)


def _heuristic_profile(tags_file: str) -> dict:
    ext = Path(tags_file).suffix.lower()
    if ext == ".csv":
        return dict(_WD14_PROFILE)
    if ext == ".txt":
        return dict(_JOYTAG_PROFILE)
    return {}


def _embedded_profile(name: str) -> dict:
    path = _DATA / "profiles" / f"{name}.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"embedded profile {name}: unsupported version {doc.get('version')}")
    return doc.get("profile", {})


def _sidecar_profile(model_dir: Path) -> dict:
    # The sidecar lives in the model folder as tagger.json (monbooru's name).
    path = model_dir / "tagger.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported profile version {doc.get('version')}")
    return doc.get("profile", {})


def resolve_profile(model_dir: Path, name: str, tags_file: str) -> Profile:
    """Heuristic (tags extension) → embedded default → model-folder sidecar;
    a later layer overwrites any axis it sets, blanks inherit. The sidecar
    is tagger.json next to the model, matching monbooru."""
    merged = {}
    for layer in (_heuristic_profile(tags_file), _embedded_profile(name), _sidecar_profile(model_dir)):
        merged.update(layer)
    fill = merged.get("fill_color")
    profile = Profile(
        input_size=int(merged.get("input_size") or 0),
        layout=merged.get("layout", ""),
        channels=merged.get("channels", ""),
        normalize=merged.get("normalize", ""),
        pad=merged.get("pad", ""),
        fill_color=tuple(int(c) for c in fill) if fill else (0, 0, 0),
        activation=merged.get("activation", ""),
        label_format=merged.get("label_format", ""),
        category_scheme=merged.get("category_scheme", ""),
        output_index=int(merged.get("output_index") or 0),
    )
    profile.validate()
    return profile


__all__ = ["Profile", "resolve_profile", "replace"]
