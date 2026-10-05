"""The SQLite mirror: job lifecycle, dedup lookup, boot sweep, tokens."""

from pathlib import Path

import pytest

from montagger import store as st


@pytest.fixture
def db(tmp_path):
    return st.Store(tmp_path / "montagger.sqlite")


def mk(db, job_id="j1", sha="a" * 64, model="wd-swinv2", filename="f.jpg", source="api"):
    db.create_job(job_id=job_id, filename=filename, source=source, model=model, sha256=sha, md5="m" * 32, bytes_len=10)
    return job_id


def test_job_lifecycle(db):
    jid = mk(db)
    assert db.get_job(jid).status == st.QUEUED
    db.set_running(jid, "CUDAExecutionProvider")
    assert db.get_job(jid).status == st.RUNNING
    db.finish_done(jid, provider="cuda", width=8, height=8, elapsed_ms=5,
                    tags=[{"name": "1girl", "category": "general", "confidence": 0.9}], rating="general")
    row = db.get_job(jid)
    assert row.status == st.DONE and row.width == 8
    tags = row.tags_json
    assert "1girl" in tags


def test_finish_error_and_lost_sweep(db):
    jid = mk(db)
    db.finish_error(jid, "boom")
    assert db.get_job(jid).status == st.ERROR

    j2 = mk(db, job_id="j2", sha="b" * 64)
    j3 = mk(db, job_id="j3", sha="c" * 64)
    db.set_running(j3, "cpu")
    assert db.mark_unfinished_lost() == 2
    assert db.get_job(j2).status == st.LOST
    assert db.get_job(j3).status == st.LOST
    assert db.get_job(jid).status == st.ERROR  # terminal rows untouched


def test_find_by_hash_dedup(db):
    jid = mk(db)
    db.finish_done(jid, provider="cpu", width=1, height=1, elapsed_ms=1, tags=[], rating="")
    assert db.find_by_hash("a" * 64, "wd-swinv2") is not None
    assert db.find_by_hash("a" * 64, "joytag") is None  # different model
    assert db.find_by_hash("f" * 64, "wd-swinv2") is None
    # a queued row with the same hash does not dedup
    mk(db, job_id="j9", sha="d" * 64)
    assert db.find_by_hash("d" * 64, "wd-swinv2") is None


def test_list_delete_clear(db):
    for i in range(3):
        jid = mk(db, job_id=f"j{i}", sha=f"{i}" * 64)
        db.finish_error(jid, "x")
    rows, total = db.list_jobs()
    assert len(rows) == 3 and total == 3
    # pagination + search
    rows, total = db.list_jobs(limit=2, offset=1)
    assert len(rows) == 2 and total == 3
    rows, total = db.list_jobs(search="f.jpg")
    assert total == 3  # every fixture row uses filename f.jpg
    rows, total = db.list_jobs(search="zzz")
    assert total == 0
    assert db.list_job_ids(status=st.ERROR) == ["j2", "j1", "j0"]  # newest first
    assert db.delete_job("j0")
    assert not db.delete_job("j0")
    assert db.delete_jobs(["j1", "j2", "ghost"]) == 2
    rows, total = db.list_jobs()
    assert rows == [] and total == 0


def test_find_pushed_by_sha(db):
    jid = mk(db)
    db.finish_done(jid, provider="cpu", width=1, height=1, elapsed_ms=1, tags=[], rating="")
    assert db.find_pushed_by_sha("a" * 64) is None  # never pushed
    db.set_pushed(jid, 7)
    hit = db.find_pushed_by_sha("a" * 64)
    assert hit is not None and hit.monbooru_id == 7
    assert db.find_pushed_by_sha("a" * 64, exclude_job_id=jid) is None
    # error rows never count as pushed
    other = mk(db, job_id="k1", sha="b" * 64)
    db.finish_error(other, "x")
    assert db.find_pushed_by_sha("b" * 64) is None


def test_purge_history_keeps_fresh(db):
    jid = mk(db)
    db.finish_error(jid, "x")
    assert db.purge_history(7) == 0  # created now, retention 7d


def test_pushed_marker(db):
    jid = mk(db)
    db.finish_done(jid, provider="cpu", width=1, height=1, elapsed_ms=1, tags=[], rating="")
    db.set_pushed(jid, 42)
    assert db.get_job(jid).monbooru_id == 42
    assert db.get_job(jid).pushed_at is not None


def test_tokens(db):
    token = db.add_token("phone")
    assert db.token_known(token)
    assert db.has_tokens()
    rows = db.list_tokens()
    assert rows[0].name == "phone"
    assert db.delete_token(rows[0].id)
    assert not db.token_known(token)
    assert not db.has_tokens()


def test_list_job_groups_keeps_images_whole(db):
    """Grouped paging is a window over images: every job of the page's
    images comes back together, whatever the page size."""
    for i, sha in enumerate(("a" * 64, "b" * 64, "c" * 64)):
        for j, model in enumerate(("wd-swinv2", "joytag")):
            mk(db, job_id=f"j{i}{j}", sha=sha, model=model, filename=f"img{i}.jpg")
    rows, total = db.list_job_groups(limit=2, offset=0)
    assert total == 3  # images, not jobs
    assert len(rows) == 4  # two whole images' worth of jobs
    shas = {r.sha256 for r in rows}
    assert len(shas) == 2  # no image split across the boundary
    # newest image first, upload model order within an image
    assert rows[0].sha256 == "c" * 64
    assert [r.model for r in rows if r.sha256 == "c" * 64] == ["wd-swinv2", "joytag"]

    rows2, total2 = db.list_job_groups(limit=2, offset=2)
    assert total2 == 3 and len(rows2) == 2
    assert {r.sha256 for r in rows2} == {"a" * 64}


def test_list_job_groups_search_scopes_to_matching_jobs(db):
    mk(db, job_id="j1", sha="a" * 64, model="wd-swinv2")
    mk(db, job_id="j2", sha="a" * 64, model="joytag")
    mk(db, job_id="j3", sha="b" * 64, model="joytag")
    rows, total = db.list_job_groups(limit=10, offset=0, search="joytag")
    assert total == 2  # two images have a joytag job
    assert {r.id for r in rows} == {"j2", "j3"}  # only the matching jobs
