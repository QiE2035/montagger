"""Backend abstraction.

A backend owns the model-specific parts: how an image becomes a prepared
tensor (preprocess - called where the bytes are already in memory, with the
decoded Pillow image) and how a prepared tensor becomes tags. The pipeline
only knows this ABC, so further backends (safetensors/PyTorch, LLM APIs)
slot in through the registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

from PIL import Image

if TYPE_CHECKING:
    from montagger.settings import RuntimeState


class Backend(ABC):
    name: str = ""

    @abstractmethod
    def preprocess(self, image: Image.Image) -> Any:
        """Turn the decoded image into a prepared tensor. Must not keep a
        reference to the PIL image afterwards (the pipeline drops it to
        bound memory)."""

    @abstractmethod
    def tag(self, prepared: Any, runtime: "RuntimeState") -> list[str]:
        """Return the monbooru tag strings, e.g. ["1girl", "character:hatsune_miku"]."""

    def reload(self, runtime: "RuntimeState") -> None:
        """Hot-reload (e.g. execution provider switch). Default: nothing."""

    def close(self) -> None:
        """Release the model resources."""


BackendFactory = Callable[["RuntimeState", Any], Backend]
_REGISTRY: dict[str, BackendFactory] = {}


def register(name: str) -> Any:
    def decorate(factory: BackendFactory | type[Backend]) -> BackendFactory:
        if isinstance(factory, type) and issubclass(factory, Backend):
            _REGISTRY[name] = lambda runtime, deps: factory()
        else:
            _REGISTRY[name] = factory
        return factory

    return decorate


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str, runtime: "RuntimeState", deps: Any) -> Backend:
    try:
        return _REGISTRY[name](runtime, deps)
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; available: {available()}") from None


# Importing the modules runs their @register decorators (kept last to avoid
# a circular import while Backend itself is still being defined).
from montagger.backends import heuristic, onnx  # noqa: E402,F401