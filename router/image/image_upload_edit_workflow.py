import asyncio
import hashlib
import json
import os
import re
import shutil
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from crud import images as crud_images
from utils.date_parser import parse_capture_date
from utils.geoserver_utils import delete_geotiff_layer, get_tif_bbox_wgs84, publish_geotiff_layer


UPLOAD_DIR = Path("uploads")
IMAGE_DIR = UPLOAD_DIR / "images"
SHAPEFILE_DIR = UPLOAD_DIR / "shapefiles"
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


for directory in (IMAGE_DIR, SHAPEFILE_DIR, TMP_UPLOAD_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _session_dir(upload_id: str) -> Path:
    value = (upload_id or "").strip()
    if not _UPLOAD_ID_RE.fullmatch(value):
        raise WorkflowInputError("uploadId 格式非法")
    UUID(hex=value)
    return TMP_UPLOAD_DIR / value.lower()


def _save_meta(upload_id: str, meta: dict) -> None:
    meta["updated_at"] = int(time.time())
    meta_file = _session_dir(upload_id) / SESSION_META_FILE
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = meta_file.with_name(f".{meta_file.name}.{uuid4().hex}.tmp")
    try:
        temp_file.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        temp_file.replace(meta_file)
    finally:
        temp_file.unlink(missing_ok=True)


def _load_meta(upload_id: str) -> dict:
    meta_file = _session_dir(upload_id) / SESSION_META_FILE
    if not meta_file.exists():
        raise WorkflowNotFoundError("上传会话不存在")
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("上传会话损坏")


def _list_uploaded_chunks(upload_id: str) -> list[int]:
    chunks_dir = _session_dir(upload_id) / CHUNKS_DIR_NAME
    if not chunks_dir.exists():
        return []
    return sorted(int(path.stem) for path in chunks_dir.glob("*.part") if path.stem.isdigit())


def _session_snapshot(upload_id: str, meta: dict) -> dict:
    return {
        "upload_id": upload_id,
        "uploaded_chunks": _list_uploaded_chunks(upload_id),
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
    chunks_dir = _session_dir(upload_id) / CHUNKS_DIR_NAME
    chunks_dir.mkdir(parents=True)
    meta = {**expected, "status": "uploading", "created_at": int(time.time())}
    _save_meta(upload_id, meta)
    return _session_snapshot(upload_id, meta)


async def save_upload_chunk(
    user_id: int,
    upload_id: str,
    chunk_index: int,
    chunk: BinaryIO,
) -> dict:
    upload_id = _session_dir(upload_id).name
    meta = _load_meta(upload_id)
    if meta.get("user_id") != user_id:
        raise WorkflowNotFoundError("上传会话不存在")
    if chunk_index < 0 or chunk_index >= meta["total_chunks"]:
        raise WorkflowInputError("chunkIndex 超出范围")

    expected_size = min(
        meta["chunk_size"],
        meta["file_size"] - chunk_index * meta["chunk_size"],
    )
    chunks_dir = _session_dir(upload_id) / CHUNKS_DIR_NAME
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
        _save_meta(upload_id, meta)
    finally:
        temp_path.unlink(missing_ok=True)

    return _session_snapshot(upload_id, meta)


@contextmanager
def _session_lock(upload_id: str):
    session_dir = _session_dir(upload_id)
    if not session_dir.is_dir():
        raise WorkflowNotFoundError("上传会话不存在")
    lock_path = session_dir / COMPLETE_LOCK_FILE
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise WorkflowConflictError("上传会话正在处理")
    except FileNotFoundError:
        raise WorkflowNotFoundError("上传会话不存在")
    try:
        os.close(descriptor)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _safe_name(value: str) -> str:
    name = re.sub(r'[^0-9A-Za-z._\-\u4e00-\u9fff]+', "_", Path(value).name).strip(" ._")
    if not name:
        raise WorkflowInputError("文件名非法")
    return name[:128]


def _safe_prefix(region_code: str, image_name: str) -> str:
    return f"{_safe_name(region_code)}_{_safe_name(image_name)}"


def _remove_path(path: str | Path | None) -> bool:
    if not path:
        return True
    try:
        Path(path).unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _boundary_paths(image) -> dict[str, str | None]:
    boundary = (getattr(image, "boundary_files", None) or [None])[0]
    return {
        "shp_path": getattr(boundary, "shp_path", None),
        "dbf_path": getattr(boundary, "dbf_path", None),
        "prj_path": getattr(boundary, "prj_path", None),
    }


def _validate_boundary_files(
    files: list[tuple[str, BinaryIO]] | None,
    required: bool,
) -> dict[str, BinaryIO]:
    if not files:
        if required:
            raise WorkflowInputError("boundaryFiles 必须包含 shp/dbf/prj")
        return {}
    grouped: dict[str, BinaryIO] = {}
    for filename, stream in files:
        suffix = Path(filename).suffix.lower().removeprefix(".")
        if suffix not in {"shp", "dbf", "prj"} or suffix in grouped:
            raise WorkflowInputError("boundaryFiles 必须包含唯一的 shp/dbf/prj")
        grouped[suffix] = stream
    if set(grouped) != {"shp", "dbf", "prj"}:
        raise WorkflowInputError("boundaryFiles 必须包含完整 shp/dbf/prj")
    return grouped


def _save_boundary_files(files: dict[str, BinaryIO], prefix: str) -> tuple[dict[str, str], Path]:
    target_dir = SHAPEFILE_DIR / f"{prefix}_{uuid4().hex}"
    target_dir.mkdir(parents=True, exist_ok=False)
    paths: dict[str, str] = {}
    try:
        for suffix, stream in files.items():
            destination = target_dir / f"boundary.{suffix}"
            temp = destination.with_suffix(f".{suffix}.tmp")
            with temp.open("wb") as output:
                while data := stream.read(1024 * 1024):
                    output.write(data)
            temp.replace(destination)
            paths[f"{suffix}_path"] = str(destination)
        return paths, target_dir
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise


def _merge_tif(upload_id: str, meta: dict) -> tuple[Path, list[float]]:
    required = list(range(meta["total_chunks"]))
    if _list_uploaded_chunks(upload_id) != required:
        raise WorkflowInputError("上传分片不完整")

    final_path = IMAGE_DIR / f"{uuid4().hex}_{_safe_name(meta['file_name'])}"
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    digest = hashlib.md5()
    size = 0
    try:
        with temp_path.open("wb") as output:
            for index in required:
                part_path = _session_dir(upload_id) / CHUNKS_DIR_NAME / f"{index}.part"
                with part_path.open("rb") as source:
                    while data := source.read(1024 * 1024):
                        output.write(data)
                        digest.update(data)
                        size += len(data)
        if size != meta["file_size"] or digest.hexdigest() != meta["file_hash"]:
            raise WorkflowInputError("上传文件 MD5 或大小校验失败")
        temp_path.replace(final_path)
        return final_path, get_tif_bbox_wgs84(final_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise


def _version_layer_name(region_code: str, image_name: str, image_id: int) -> str:
    return f"{_safe_prefix(region_code, image_name)}_{image_id}_{uuid4().hex}"


async def _cleanup_replaced_resources(meta: dict) -> None:
    old_layer = meta.get("old_layer_name")
    if old_layer:
        try:
            await delete_geotiff_layer(old_layer)
        except Exception:
            pass
        else:
            meta.pop("old_layer_name", None)
    old_tif = meta.get("old_tif_path")
    if _remove_path(old_tif):
        meta.pop("old_tif_path", None)
    remaining_boundaries = [
        path
        for path in meta.get("old_boundary_paths", []) or []
        if not _remove_path(path)
    ]
    if remaining_boundaries:
        meta["old_boundary_paths"] = remaining_boundaries
    else:
        meta.pop("old_boundary_paths", None)


async def _completed_image(db, user_id: int, meta: dict, operation: str):
    status = meta.get("status")
    if status not in {"completing", "completed"}:
        return None
    if meta.get("operation") != operation:
        raise WorkflowConflictError("上传会话已用于其他操作")
    image = await crud_images.get_image_by_id(db, meta.get("result_image_id"), user_id)
    expected_tif = meta.get("tif_path")
    if image and expected_tif and str(getattr(image, "img_path", "")) != expected_tif:
        image = None
    if image:
        cleanup_pending = any(
            meta.get(field)
            for field in ("old_tif_path", "old_layer_name", "old_boundary_paths")
        )
        if cleanup_pending:
            await _cleanup_replaced_resources(meta)
        if status == "completing" or cleanup_pending:
            meta["status"] = "completed"
            try:
                _save_meta(meta["upload_id"], meta)
            except OSError:
                pass
        return image
    if status == "completed":
        raise WorkflowConflictError("上传会话结果已不存在")

    layer_name = meta.get("layer_name")
    if layer_name:
        try:
            await delete_geotiff_layer(layer_name)
        except Exception:
            pass
    _remove_path(meta.get("tif_path"))
    boundary_dir = meta.get("boundary_dir")
    if boundary_dir:
        shutil.rmtree(boundary_dir, ignore_errors=True)
    for field in (
        "operation",
        "result_image_id",
        "tif_path",
        "boundary_dir",
        "layer_name",
        "old_tif_path",
        "old_layer_name",
        "old_boundary_paths",
    ):
        meta.pop(field, None)
    meta["status"] = "uploading"
    _save_meta(meta["upload_id"], meta)
    return None


async def create_image_from_upload(
    db,
    user_id: int,
    upload_id: str,
    image_name: str,
    resolution: float,
    capture_date: str,
    satellite: str,
    image_type: str,
    region_code: str,
    boundary_files: list[tuple[str, BinaryIO]],
):
    upload_id = _session_dir(upload_id).name
    boundary_group = _validate_boundary_files(boundary_files, required=True)
    with _session_lock(upload_id):
        meta = _load_meta(upload_id)
        meta["upload_id"] = upload_id
        if meta.get("user_id") != user_id:
            raise WorkflowNotFoundError("上传会话不存在")
        if completed := await _completed_image(db, user_id, meta, "create"):
            return completed

        tif_path = None
        boundary_dir = None
        layer_name = None
        try:
            tif_path, bbox = _merge_tif(upload_id, meta)
            boundary_paths, boundary_dir = _save_boundary_files(
                boundary_group,
                _safe_prefix(region_code, image_name),
            )
            image = await crud_images.create_image(
                db=db,
                user_id=user_id,
                image_name=image_name,
                resolution=resolution,
                capture_date=parse_capture_date(capture_date),
                satellite=satellite,
                image_type=image_type,
                region_code=region_code,
                img_path=str(tif_path),
                bbox=bbox,
            )
            await crud_images.create_boundary_files(
                db=db,
                image_id=image.id,
                file_prefix=boundary_dir.name,
                **boundary_paths,
            )
            layer_name = _version_layer_name(region_code, image_name, image.id)
            wms_url = await publish_geotiff_layer(tif_path, layer_name)
            await crud_images.update_image_fields(
                db,
                image,
                {"layer_name": layer_name, "wms_url": wms_url},
            )
            meta.update(
                status="completing",
                operation="create",
                result_image_id=image.id,
                tif_path=str(tif_path),
                boundary_dir=str(boundary_dir),
                layer_name=layer_name,
            )
            _save_meta(upload_id, meta)
            await db.commit()
        except Exception:
            try:
                await db.rollback()
            except Exception:
                pass
            if layer_name:
                try:
                    await delete_geotiff_layer(layer_name)
                except Exception:
                    pass
            _remove_path(tif_path)
            if boundary_dir:
                shutil.rmtree(boundary_dir, ignore_errors=True)
            raise

        try:
            result = await crud_images.get_image_by_id(db, image.id, user_id) or image
        except Exception:
            result = image
        meta["status"] = "completed"
        try:
            _save_meta(upload_id, meta)
        except OSError:
            pass
        return result


async def edit_image_workflow(
    db,
    user_id: int,
    image_id: int,
    upload_id: str | None,
    image_name: str | None,
    resolution: float | None,
    capture_date: str | None,
    satellite: str | None,
    image_type: str | None,
    region_code: str | None,
    boundary_files: list[tuple[str, BinaryIO]] | None,
):
    image = await crud_images.get_image_by_id(db, image_id, user_id)
    if not image:
        raise WorkflowNotFoundError("影像不存在")
    boundary_group = _validate_boundary_files(boundary_files, required=False)
    normalized_upload_id = _session_dir(upload_id).name if upload_id else None
    context = _session_lock(normalized_upload_id) if normalized_upload_id else nullcontext()

    with context:
        meta = _load_meta(normalized_upload_id) if normalized_upload_id else None
        if meta:
            meta["upload_id"] = normalized_upload_id
        if meta and meta.get("user_id") != user_id:
            raise WorkflowNotFoundError("上传会话不存在")
        if meta and (completed := await _completed_image(db, user_id, meta, f"edit:{image_id}")):
            return completed

        old_tif = image.img_path
        old_layer = image.layer_name
        old_boundaries = _boundary_paths(image)
        tif_path = None
        boundary_dir = None
        new_layer = None
        try:
            updates = {
                key: value for key, value in {
                    "image_name": image_name,
                    "resolution": resolution,
                    "capture_date": parse_capture_date(capture_date) if capture_date else None,
                    "satellite": satellite,
                    "image_type": image_type,
                    "region_code": region_code,
                }.items() if value is not None
            }
            final_name = image_name or image.image_name
            final_region = region_code or image.region_code
            if meta:
                tif_path, bbox = _merge_tif(normalized_upload_id, meta)
                updates.update(img_path=str(tif_path), bbox=bbox)
            boundary_paths = {}
            if boundary_group:
                boundary_paths, boundary_dir = _save_boundary_files(
                    boundary_group,
                    _safe_prefix(final_region, final_name),
                )
            if updates:
                await crud_images.update_image_fields(db, image, updates)
            if boundary_paths:
                await crud_images.upsert_boundary_files(
                    db,
                    image,
                    file_prefix=boundary_dir.name,
                    **boundary_paths,
                )
            if tif_path:
                new_layer = _version_layer_name(final_region, final_name, image.id)
                wms_url = await publish_geotiff_layer(tif_path, new_layer)
                await crud_images.update_image_fields(
                    db,
                    image,
                    {"layer_name": new_layer, "wms_url": wms_url},
                )
            if meta:
                meta.update(
                    status="completing",
                    operation=f"edit:{image_id}",
                    result_image_id=image.id,
                    tif_path=str(tif_path),
                    boundary_dir=str(boundary_dir) if boundary_dir else None,
                    layer_name=new_layer,
                    old_tif_path=old_tif,
                    old_layer_name=old_layer,
                )
                if boundary_dir:
                    meta["old_boundary_paths"] = [
                        path for path in old_boundaries.values() if path
                    ]
                _save_meta(normalized_upload_id, meta)
            await db.commit()
        except Exception:
            try:
                await db.rollback()
            except Exception:
                pass
            if new_layer:
                try:
                    await delete_geotiff_layer(new_layer)
                except Exception:
                    pass
            _remove_path(tif_path)
            if boundary_dir:
                shutil.rmtree(boundary_dir, ignore_errors=True)
            raise

        try:
            result = await crud_images.get_image_by_id(db, image.id, user_id) or image
        except Exception:
            result = image
        if meta:
            await _cleanup_replaced_resources(meta)
            meta["status"] = "completed"
            try:
                _save_meta(normalized_upload_id, meta)
            except OSError:
                pass

    if not meta and boundary_dir:
        for path in old_boundaries.values():
            _remove_path(path)
    return result


async def delete_image_workflow(db, user_id: int, image_id: int) -> bool:
    image = await crud_images.get_image_by_id(db, image_id, user_id)
    if not image:
        return False
    layer_name = image.layer_name
    deleted = await crud_images.delete_image_with_files(db, image_id, user_id)
    await db.commit()
    _remove_path(deleted.get("img_path"))
    for path in deleted.get("boundary_paths", []):
        _remove_path(path)
    if layer_name:
        try:
            await delete_geotiff_layer(layer_name)
        except Exception:
            pass
    return True


def _cleanup_expired_tmp_uploads() -> None:
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
        await asyncio.to_thread(_cleanup_expired_tmp_uploads)
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
