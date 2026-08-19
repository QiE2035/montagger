"""Store: schema migration, dedup submit, status flow, paging, resume."""

from __future__ import annotations

import pytest

from montagger.store import Store


@pytest.fixture
def store(tmp_path: pytest.TempPathFactory) -> Store:
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def test_submit_dedup(store: Store) -> None:
    inserted, known = store.submit([1, 2, 3, 2, 3])
    assert inserted == 3
    assert known == 2
    inserted, known = store.submit([1])
    assert inserted == 0
    assert known == 1


def test_status_flow(store: Store) -> None:
    store.submit([42])
    store.mark_processing(42)
    store.mark_done(42, ["a", "b"])
    stats = store.stats()
    assert stats["done"] == 1
    rows, total = store.results(1, 50, "done")
    assert total == 1
    assert rows[0]["tags"] == '["a", "b"]'

    store.submit([7])
    store.mark_processing(7)
    store.mark_failed(7, "boom", 1)
    stats = store.stats()
    assert stats["failed"] == 1
    store.clear_results()
    assert store.stats()["done"] == 0
    assert store.stats()["failed"] == 0


def test_paging_and_filter(store: Store) -> None:
    store.submit(list(range(1, 31)))
    for i in range(1, 31):
        store.mark_done(i, [])
    rows, total = store.results(1, 10, "done")
    assert len(rows) == 10
    assert total == 30
    rows, total = store.results(3, 10, "done")
    assert len(rows) == 10
    assert total == 30
    rows, total = store.results(1, 10, "failed")
    assert total == 0
    rows, total = store.results(1, 10, None)
    assert total == 30


def test_resume_ids(store: Store) -> None:
    store.submit([1, 2, 3, 4])
    store.mark_processing(1)
    store.mark_done(2, [])
    store.mark_failed(3, "x", 0)
    ids = store.resume_ids()
    assert sorted(ids) == [1, 3, 4]  # done(2) excluded
    assert store.stats()["pending"] == 3


def test_retry_failed(store: Store) -> None:
    store.submit([5])
    store.mark_failed(5, "boom", 0)
    assert store.retry_failed() == [5]
    assert store.stats()["failed"] == 0
    assert store.stats()["pending"] == 1