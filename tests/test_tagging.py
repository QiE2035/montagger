"""Relay handler: payload parsing, enqueue-and-reply, dedup message."""

from __future__ import annotations

from montagger.tagging import RelayPayload, relay_answer


class FakePipeline:
    def __init__(self) -> None:
        self.submitted: list[list[int]] = []

    def submit(self, ids: list[int]) -> tuple[int, int]:
        self.submitted.append(ids)
        known = len(set(ids) & {2, 3})
        return len(ids) - known, known


def test_basic_answer() -> None:
    pipe = FakePipeline()
    payload = RelayPayload(image_ids=[1, 2, 3, 4])
    answer = relay_answer(pipe, payload)
    assert answer["ok"] is True
    assert answer["refresh"] is False
    assert answer["message"] == "queued 2 image(s), 2 already known"


def test_duplicate_ids_deduped_and_reported() -> None:
    pipe = FakePipeline()
    payload = RelayPayload(image_ids=[2, 3])
    answer = relay_answer(pipe, payload)
    assert "0 image(s)" in answer["message"]
    assert "2 already known" in answer["message"]


def test_empty_payload() -> None:
    answer = relay_answer(FakePipeline(), RelayPayload(image_ids=[]))
    assert answer["ok"] is True
    assert answer["message"] == "queued 0 image(s)"


def test_payload_parsing() -> None:
    payload = RelayPayload.model_validate(
        {
            "payload": 1,
            "monbooru": "0.1.0",
            "gallery": "main",
            "slot": "batch-bar",
            "button": "tag with montagger",
            "image_ids": [11, 22],
        }
    )
    assert payload.image_ids == [11, 22]
    assert payload.gallery == "main"