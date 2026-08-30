import asyncio
import os
from io import BytesIO
import time

import pytest

from router.image import upload_sessions


def test_begin_upload_creates_and_resumes_the_same_session(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_sessions, "TMP_UPLOAD_DIR", tmp_path / "tmp")
    payload = {
        "file_name": "river.tif",
        "file_size": 8,
        "chunk_size": 4,
        "total_chunks": 2,
        "file_hash": "e8dc4081b13434b45189a720b77b6818",
    }

    first = upload_sessions.begin_upload_session(7, **payload)
    asyncio.run(
        upload_sessions.save_upload_chunk(7, first["upload_id"], 0, BytesIO(b"abcd"))
    )
    resumed = upload_sessions.begin_upload_session(7, **payload)

    assert resumed["upload_id"] == first["upload_id"]
    assert resumed["uploaded_chunks"] == [0]
    assert resumed["total_chunks"] == 2


def test_save_upload_chunk_writes_the_part_and_reports_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_sessions, "TMP_UPLOAD_DIR", tmp_path / "tmp")

    session = upload_sessions.begin_upload_session(
        7,
        file_name="river.tif",
        file_size=8,
        chunk_size=4,
        total_chunks=2,
        file_hash="e8dc4081b13434b45189a720b77b6818",
    )

    result = asyncio.run(
        upload_sessions.save_upload_chunk(7, session["upload_id"], 0, BytesIO(b"abcd"))
    )

    assert result == {
        "upload_id": session["upload_id"],
        "uploaded_chunks": [0],
        "total_chunks": 2,
    }
    assert (
        upload_sessions.TMP_UPLOAD_DIR
        / session["upload_id"]
        / upload_sessions.CHUNKS_DIR_NAME
        / "0.part"
    ).read_bytes() == b"abcd"


def test_save_upload_chunk_rejects_an_out_of_range_index(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_sessions, "TMP_UPLOAD_DIR", tmp_path / "tmp")
    session = upload_sessions.begin_upload_session(
        7,
        file_name="river.tif",
        file_size=8,
        chunk_size=4,
        total_chunks=2,
        file_hash="e8dc4081b13434b45189a720b77b6818",
    )

    with pytest.raises(upload_sessions.WorkflowInputError, match="chunkIndex"):
        asyncio.run(
            upload_sessions.save_upload_chunk(
                7, session["upload_id"], 2, BytesIO(b"abcd")
            )
        )

    with pytest.raises(upload_sessions.WorkflowInputError, match="分片大小"):
        asyncio.run(
            upload_sessions.save_upload_chunk(
                7, session["upload_id"], 0, BytesIO(b"abc")
            )
        )


def test_cleanup_expired_tmp_uploads_removes_stale_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_sessions, "TMP_UPLOAD_DIR", tmp_path / "tmp")
    session = upload_sessions.begin_upload_session(
        7,
        file_name="river.tif",
        file_size=4,
        chunk_size=4,
        total_chunks=1,
        file_hash="e2fc714c4727ee9395f324cd2e7f331f",
    )
    meta_file = (
        upload_sessions.TMP_UPLOAD_DIR
        / session["upload_id"]
        / upload_sessions.SESSION_META_FILE
    )
    lock_file = meta_file.parent / upload_sessions.COMPLETE_LOCK_FILE
    lock_file.write_text("", encoding="utf-8")
    stale_time = time.time() - upload_sessions.UPLOAD_TTL_SECONDS - 1
    os.utime(lock_file, (stale_time, stale_time))

    upload_sessions.cleanup_expired_tmp_uploads()

    assert meta_file.parent.exists()
    assert not lock_file.exists()

    stale_time = time.time() - upload_sessions.UPLOAD_TTL_SECONDS - 1
    os.utime(meta_file, (stale_time, stale_time))

    upload_sessions.cleanup_expired_tmp_uploads()

    assert not meta_file.parent.exists()
