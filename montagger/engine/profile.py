"""Model profile: how a model's input must be built and its output read.

Field semantics match monbooru's tagger.Profile one for one; a sidecar
profile.json next to the model overrides any subset of the embedded
default, so the previous resolution layer fills the rest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROFILE_SCHEMA_VERSION = 1

Layout = Literal["nhwc", "nchw"]
Channels = Literal["rgb", "bgr"]
Normalize = Literal["none", "div255", "imagenet", "clip"]
Pad = Literal["white_square", "mean_color_aspect"]
Activation = Literal["sigmoid_in_model", "logits"]
LabelFormat = Literal["wd14_csv", "joytag_txt", "camie_json"]
CategoryScheme = Literal["wd14_numeric", "single_general", "name_string"]


class Profile(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_size: int = Field(default=0, ge=0)  # 0 reads the size from the model's input
    layout: Layout
    channels: Channels
    normalize: Normalize
    pad: Pad
    # mean_color_aspect fill; camie default when zero
    fill_color: tuple[int, int, int] = (0, 0, 0)
    activation: Activation
    label_format: LabelFormat
    category_scheme: CategoryScheme
    output_index: int = Field(default=0, ge=0)  # camie's refined head and eva02's head are not output 0

    @field_validator("fill_color", mode="before")
    @classmethod
    def _fill_color(cls, value: object) -> object:
        return tuple(int(c) for c in value) if value else (0, 0, 0)

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
    merged: dict = {}
    for layer in (_heuristic_profile(tags_file), _embedded_profile(name), _sidecar_profile(model_dir)):
        merged.update(layer)
    return Profile.model_validate(merged)


__all__ = ["Profile", "resolve_profile"]
