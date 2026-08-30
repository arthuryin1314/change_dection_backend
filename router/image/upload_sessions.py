import asyncio
import json
import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4


UPLOAD_DIR = Path("uploads")
TMP_UPLOAD_DIR = UPLOAD_DIR / "tmp"
SESSION_META_FILE = "session.json"
CHUNKS_DIR_NAME = "chunks"
COMPLETE_LOCK_FILE = ".complete.lock"
UPLOAD_TTL_SECONDS = 24 * 60 * 60
TMP_CLEANUP_INTERVAL_SECONDS = 60 * 60
_UPLOAD_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_tmp_cleanup_task: asyncio.Task | None = None


class WorkflowInputError(ValueError):
    pass


class WorkflowNotFoundError(LookupError):
    pass


class WorkflowConflictError(RuntimeError):
    pass


def session_dir(upload_id: str) -> Path:
    value = upload_id.strip()
    if not _UPLOAD_ID_RE.fullmatch(value):
        raise WorkflowInputError("uploadId 格式非法")
    UUID(hex=value)
    return TMP_UPLOAD_DIR / value.lower()


def save_session(upload_id: str, meta: dict) -> None:
    meta["updated_at"] = int(time.time())
    meta_file = session_dir(upload_id) / SESSION_META_FILE
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = meta_file.with_name(f".{meta_file.name}.{uuid4().hex}.tmp")
    try:
        temp_file.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        temp_file.replace(meta_file)
    finally:
        temp_file.unlink(missing_ok=True)


def load_session(upload_id: str) -> dict:
    meta_file = session_dir(upload_id) / SESSION_META_FILE
    if not meta_file.exists():
        raise WorkflowNotFoundError("上传会话不存在")
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("上传会话损坏") from exc


def list_uploaded_chunks(upload_id: str) -> list[int]:
    chunks_dir = session_dir(upload_id) / CHUNKS_DIR_NAME
    if not chunks_dir.exists():
        return []
    return sorted(int(path.stem) for path in chunks_dir.glob("*.part") if path.stem.isdigit())


def _session_snapshot(upload_id: str, meta: dict) -> dict:
    return {
        "upload_id": upload_id,
        "uploaded_chunks": list_uploaded_chunks(upload_id),
        "total_chunks": meta["total_chunks"],
    }


def begin_upload_session(
    user_id: int,
    *,
    file_name: str,
    file_size: int,
    chunk_size: int,
    total_chunks: int,
    file_hash: str,
) -> dict:
    TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if total_chunks != (file_size + chunk_size - 1) // chunk_size:
        raise WorkflowInputError("totalChunks 与文件大小不一致")

    expected = {
        "user_id": user_id,
        "file_hash": file_hash.lower(),
        "file_name": Path(file_name).name,
        "file_size": file_size,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
    }
    for meta_file in TMP_UPLOAD_DIR.glob(f"*/{SESSION_META_FILE}"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("status") == "uploading" and all(
            meta.get(key) == value for key, value in expected.items()
        ):
            return _session_snapshot(meta_file.parent.name, meta)

    upload_id = uuid4().hex
    chunks_dir = session_dir(upload_id) / CHUNKS_DIR_NAME
    chunks_dir.mkdir(parents=True)
    meta = {**expected, "status": "uploading", "created_at": int(time.time())}
    save_session(upload_id, meta)
    return _session_snapshot(upload_id, meta)


async def save_upload_chunk(
    user_id: int,
    upload_id: str,
    chunk_index: int,
    chunk: BinaryIO,
) -> dict:
    upload_id = session_dir(upload_id).name
    meta = load_session(upload_id)
    if meta.get("user_id") != user_id:
        raise WorkflowNotFoundError("上传会话不存在")
    if chunk_index < 0 or chunk_index >= meta["total_chunks"]:
        raise WorkflowInputError("chunkIndex 超出范围")

    expected_size = min(
        meta["chunk_size"],
        meta["file_size"] - chunk_index * meta["chunk_size"],
    )
    chunks_dir = session_dir(upload_id) / CHUNKS_DIR_NAME
    chunks_dir.mkdir(parents=True, exist_ok=True)
    part_path = chunks_dir / f"{chunk_index}.part"
    temp_path = chunks_dir / f".{chunk_index}.{uuid4().hex}.tmp"
    written = 0
    try:
        with temp_path.open("wb") as destination:
            while data := chunk.read(1024 * 1024):
                destination.write(data)
                written += len(data)
        if written != expected_size:
            raise WorkflowInputError("分片大小不正确")
        temp_path.replace(part_path)
        save_session(upload_id, meta)
    finally:
        temp_path.unlink(missing_ok=True)

    return _session_snapshot(upload_id, meta)


@contextmanager
def lock_upload_session(upload_id: str):
    directory = session_dir(upload_id)
    if not directory.is_dir():
        raise WorkflowNotFoundError("上传会话不存在")
    lock_path = directory / COMPLETE_LOCK_FILE
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise WorkflowConflictError("上传会话正在处理") from exc
    except FileNotFoundError as exc:
        raise WorkflowNotFoundError("上传会话不存在") from exc
    try:
        os.close(descriptor)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def cleanup_expired_tmp_uploads() -> None:
    if not TMP_UPLOAD_DIR.exists():
        return
    now = time.time()
    for directory in TMP_UPLOAD_DIR.iterdir():
        if not directory.is_dir():
            continue
        lock_file = directory / COMPLETE_LOCK_FILE
        if lock_file.exists():
            if now - lock_file.stat().st_mtime <= UPLOAD_TTL_SECONDS:
                continue
            lock_file.unlink(missing_ok=True)
        meta_file = directory / SESSION_META_FILE
        age_source = meta_file if meta_file.exists() else directory
        if now - age_source.stat().st_mtime > UPLOAD_TTL_SECONDS:
            shutil.rmtree(directory, ignore_errors=True)


async def _periodic_tmp_cleanup() -> None:
    while True:
        await asyncio.to_thread(cleanup_expired_tmp_uploads)
        await asyncio.sleep(TMP_CLEANUP_INTERVAL_SECONDS)


def start_tmp_cleanup_task() -> None:
    global _tmp_cleanup_task
    if _tmp_cleanup_task is None or _tmp_cleanup_task.done():
        _tmp_cleanup_task = asyncio.create_task(_periodic_tmp_cleanup())


async def stop_tmp_cleanup_task() -> None:
    global _tmp_cleanup_task
    task = _tmp_cleanup_task
    _tmp_cleanup_task = None
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
