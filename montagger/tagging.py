"""The relay endpoint contract between monbooru and montagger.

monbooru POSTs the button scope (image_ids) to /relay/tag and gives us 10 s
to answer - so we enqueue and reply immediately. The bulk of the work runs
in the pipeline; the message merely reports how many ids were accepted.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RelayPayload(BaseModel):
    """What monbooru POSTs to a relay button click (payload version 1)."""

    payload: int = 1
    monbooru: str = ""
    gallery: str = ""
    slot: str = ""
    button: str = ""
    image_ids: list[int] = Field(default_factory=list)


def relay_answer(pipeline: Any, payload: RelayPayload) -> dict[str, Any]:
    """Enqueue the ids and build the {ok, message, refresh} answer."""
    new = 0
    known = 0
    if payload.image_ids:
        new, known = pipeline.submit(payload.image_ids)
    message = f"queued {new} image(s)"
    if known:
        message += f", {known} already known"
    return {"ok": True, "message": message, "refresh": False}