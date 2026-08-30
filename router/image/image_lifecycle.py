import asyncio
import hashlib
import json
import re
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from crud import images as crud_images
from router.image import upload_sessions
from router.image.upload_sessions import (
    CHUNKS_DIR_NAME,
    WorkflowConflictError,
    WorkflowInputError,
    WorkflowNotFoundError,
    list_uploaded_chunks as _list_uploaded_chunks,
    load_session as _load_meta,
    lock_upload_session as _session_lock,
    save_session as _save_meta,
    session_dir as _session_dir,
)
from utils.date_parser import parse_capture_date
from utils.geoserver_utils import delete_geotiff_layer, get_tif_bbox_wgs84, publish_geotiff_layer


UPLOAD_DIR = Path("uploads")
IMAGE_DIR = UPLOAD_DIR / "images"
SHAPEFILE_DIR = UPLOAD_DIR / "shapefiles"
OPERATION_DIR = UPLOAD_DIR / "operations"
ASSET_CLEANUP_INTERVAL_SECONDS = 60 * 60
_OPERATION_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_asset_cleanup_task: asyncio.Task | None = None


for directory in (IMAGE_DIR, SHAPEFILE_DIR, OPERATION_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PreparedUserImageDeletion:
    operation_id: str
    deleted_count: int


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
    return _remove_permanent_asset(str(path))


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
        for path in meta.get("old_boundary_paths", [])
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
        _remove_permanent_asset_directory(boundary_dir)
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


async def create_image(
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
        boundary_paths = {}
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
            await _compensate_new_assets(
                user_id,
                tif_path,
                list(boundary_paths.values()),
                layer_name,
            )
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


async def edit_image(
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
    old_boundaries = _boundary_paths(image)
    _validate_permanent_asset_paths([
        path
        for path in [image.img_path, *old_boundaries.values()]
        if path
    ])
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
        tif_path = None
        boundary_dir = None
        boundary_paths = {}
        new_layer = None
        boundary_cleanup_operation_id = None
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
            if boundary_group:
                boundary_paths, boundary_dir = _save_boundary_files(
                    boundary_group,
                    _safe_prefix(final_region, final_name),
                )
                old_boundary_paths = [
                    path for path in old_boundaries.values() if path
                ]
                if old_boundary_paths:
                    boundary_cleanup_operation_id = _prepare_cleanup(
                        "replacement",
                        user_id,
                        [image.id],
                        [],
                        old_boundary_paths,
                        [],
                        list(boundary_paths.values()),
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
                _save_meta(normalized_upload_id, meta)
            await db.commit()
        except Exception:
            try:
                await db.rollback()
            except Exception:
                pass
            if boundary_cleanup_operation_id:
                await _rollback_replacement_cleanup(
                    boundary_cleanup_operation_id,
                    include_replaced_assets=False,
                )
            await _compensate_new_assets(
                user_id,
                tif_path,
                [] if boundary_cleanup_operation_id else list(boundary_paths.values()),
                new_layer,
            )
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
        if boundary_cleanup_operation_id:
            await finish_cleanup(boundary_cleanup_operation_id)

    return result


def _operation_path(operation_id: str) -> Path:
    value = operation_id.strip()
    if not _OPERATION_ID_RE.fullmatch(value):
        raise WorkflowInputError("清理操作ID格式非法")
    return OPERATION_DIR / f"{value.lower()}.json"


def _save_operation(operation: dict) -> None:
    operation["updated_at"] = int(time.time())
    path = _operation_path(operation["operation_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp_path.write_text(
            json.dumps(operation, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _prepare_cleanup(
    kind: str,
    user_id: int,
    image_ids: list[int],
    image_paths: list[str],
    boundary_paths: list[str],
    layer_names: list[str],
    expected_boundary_paths: list[str],
) -> str:
    operation_id = uuid4().hex
    _save_operation({
        "operation_id": operation_id,
        "kind": kind,
        "status": "prepared",
        "user_id": user_id,
        "image_ids": image_ids,
        "image_paths": image_paths,
        "boundary_paths": boundary_paths,
        "layer_names": layer_names,
        "expected_boundary_paths": expected_boundary_paths,
        "created_at": int(time.time()),
    })
    return operation_id


def _discard_cleanup(operation_id: str) -> None:
    _operation_path(operation_id).unlink(missing_ok=True)


def _load_operation(operation_id: str) -> dict:
    path = _operation_path(operation_id)
    if not path.exists():
        raise WorkflowNotFoundError("清理操作不存在")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("清理操作记录损坏") from exc


def _is_permanent_asset_path(path: Path) -> bool:
    resolved = path.resolve()
    return any(
        resolved.is_relative_to(root.resolve())
        for root in (IMAGE_DIR, SHAPEFILE_DIR)
    )


def _remove_permanent_asset(path_value: str) -> bool:
    path = Path(path_value)
    if not _is_permanent_asset_path(path):
        return False
    try:
        path.unlink(missing_ok=True)
        parent = path.parent
        if parent != IMAGE_DIR and parent != SHAPEFILE_DIR:
            try:
                parent.rmdir()
            except OSError:
                pass
        return True
    except OSError:
        return False


def _remove_permanent_asset_directory(path_value: str) -> bool:
    path = Path(path_value)
    resolved = path.resolve()
    shapefile_root = SHAPEFILE_DIR.resolve()
    if resolved == shapefile_root or not resolved.is_relative_to(shapefile_root):
        return False
    try:
        shutil.rmtree(resolved)
        return True
    except OSError:
        return False


def _validate_permanent_asset_paths(paths: list[str]) -> None:
    if any(not _is_permanent_asset_path(Path(path)) for path in paths):
        raise WorkflowInputError("影像资产路径不在 uploads 目录内")


async def finish_cleanup(operation_id: str) -> bool:
    try:
        operation = _load_operation(operation_id)
    except WorkflowNotFoundError:
        return True

    operation["status"] = "cleanup_pending"
    operation["image_paths"] = [
        path for path in operation["image_paths"]
        if not _remove_permanent_asset(path)
    ]
    operation["boundary_paths"] = [
        path for path in operation["boundary_paths"]
        if not _remove_permanent_asset(path)
    ]
    remaining_layers = []
    for layer_name in operation["layer_names"]:
        try:
            await delete_geotiff_layer(layer_name)
        except Exception:
            remaining_layers.append(layer_name)
    operation["layer_names"] = remaining_layers

    if any(operation[field] for field in ("image_paths", "boundary_paths", "layer_names")):
        _save_operation(operation)
        return False
    _discard_cleanup(operation_id)
    return True


async def _compensate_new_assets(
    user_id: int,
    image_path: Path | None,
    boundary_paths: list[str],
    layer_name: str | None,
) -> None:
    if not image_path and not boundary_paths and not layer_name:
        return
    try:
        operation_id = _prepare_cleanup(
            "compensation",
            user_id,
            [],
            [str(image_path)] if image_path else [],
            boundary_paths,
            [layer_name] if layer_name else [],
            [],
        )
        await finish_cleanup(operation_id)
    except Exception:
        if image_path:
            _remove_path(image_path)
        for path in boundary_paths:
            _remove_path(path)
        if layer_name:
            try:
                await delete_geotiff_layer(layer_name)
            except Exception:
                pass


async def _rollback_replacement_cleanup(
    operation_id: str,
    include_replaced_assets: bool,
) -> None:
    operation = _load_operation(operation_id)
    replacement_paths = operation["boundary_paths"] if include_replaced_assets else []
    operation.update(
        kind="compensation",
        status="prepared",
        boundary_paths=[
            *replacement_paths,
            *operation["expected_boundary_paths"],
        ],
        expected_boundary_paths=[],
    )
    _save_operation(operation)
    await finish_cleanup(operation_id)


async def delete_image(db, user_id: int, image_id: int) -> bool:
    image = await crud_images.get_image_by_id(db, image_id, user_id)
    if not image:
        return False
    old_boundaries = _boundary_paths(image)
    _validate_permanent_asset_paths([
        path
        for path in [image.img_path, *old_boundaries.values()]
        if path
    ])
    layer_name = image.layer_name
    deleted = await crud_images.delete_image_with_files(db, image_id, user_id)
    operation_id = _prepare_cleanup(
        "image_delete",
        user_id,
        [image_id],
        [deleted["img_path"]] if deleted["img_path"] else [],
        deleted["boundary_paths"],
        [layer_name] if layer_name else [],
        [],
    )
    try:
        await db.commit()
    except Exception:
        try:
            await db.rollback()
        finally:
            _discard_cleanup(operation_id)
        raise
    try:
        await finish_cleanup(operation_id)
    except Exception:
        pass
    return True


async def prepare_user_image_deletion(
    db,
    user_id: int,
) -> PreparedUserImageDeletion:
    deleted = await crud_images.delete_images_by_user_with_files(db, user_id)
    _validate_permanent_asset_paths([
        *deleted["img_paths"],
        *deleted["boundary_paths"],
    ])
    operation_id = _prepare_cleanup(
        "user_image_delete",
        user_id,
        deleted["image_ids"],
        deleted["img_paths"],
        deleted["boundary_paths"],
        deleted["layer_names"],
        [],
    )
    return PreparedUserImageDeletion(operation_id, deleted["deleted_count"])


def cancel_cleanup(operation_id: str) -> None:
    _discard_cleanup(operation_id)


async def recover_cleanup_operations(session_factory) -> None:
    async with session_factory() as db:
        for path in OPERATION_DIR.glob("*.json"):
            try:
                operation = _load_operation(path.stem)
            except RuntimeError:
                continue
            if operation["status"] == "prepared" and operation["kind"] == "replacement":
                image = await crud_images.get_image_by_id(
                    db,
                    operation["image_ids"][0],
                    operation["user_id"],
                )
                if image:
                    current_paths = {
                        path for path in _boundary_paths(image).values() if path
                    }
                    if not set(operation["expected_boundary_paths"]).issubset(current_paths):
                        await _rollback_replacement_cleanup(
                            operation["operation_id"],
                            include_replaced_assets=False,
                        )
                        continue
                else:
                    await _rollback_replacement_cleanup(
                        operation["operation_id"],
                        include_replaced_assets=True,
                    )
                    continue
                await finish_cleanup(operation["operation_id"])
                continue
            if operation["status"] == "prepared" and operation["kind"] != "compensation":
                existing_count = await crud_images.count_images_by_ids(
                    db,
                    operation["user_id"],
                    operation["image_ids"],
                )
                if existing_count == len(operation["image_ids"]):
                    _discard_cleanup(operation["operation_id"])
                    continue
                if existing_count:
                    continue
            await finish_cleanup(operation["operation_id"])

        for meta_file in upload_sessions.TMP_UPLOAD_DIR.glob(
            f"*/{upload_sessions.SESSION_META_FILE}"
        ):
            try:
                meta = upload_sessions.load_session(meta_file.parent.name)
            except (RuntimeError, WorkflowNotFoundError):
                continue
            if meta["status"] not in {"completing", "completed"}:
                continue
            operation = meta.get("operation")
            if not operation:
                continue
            try:
                await _completed_image(db, meta["user_id"], meta, operation)
            except WorkflowConflictError:
                continue


async def retry_pending_cleanup() -> None:
    for path in OPERATION_DIR.glob("*.json"):
        try:
            operation = _load_operation(path.stem)
        except RuntimeError:
            continue
        if operation["status"] == "cleanup_pending":
            await finish_cleanup(operation["operation_id"])

    for meta_file in upload_sessions.TMP_UPLOAD_DIR.glob(
        f"*/{upload_sessions.SESSION_META_FILE}"
    ):
        try:
            meta = upload_sessions.load_session(meta_file.parent.name)
        except (RuntimeError, WorkflowNotFoundError):
            continue
        if meta["status"] != "completed" or not any(
            meta.get(field)
            for field in ("old_tif_path", "old_layer_name", "old_boundary_paths")
        ):
            continue
        await _cleanup_replaced_resources(meta)
        upload_sessions.save_session(meta_file.parent.name, meta)


async def _periodic_asset_cleanup() -> None:
    while True:
        await retry_pending_cleanup()
        await asyncio.sleep(ASSET_CLEANUP_INTERVAL_SECONDS)


def start_cleanup_task() -> None:
    global _asset_cleanup_task
    if _asset_cleanup_task is None or _asset_cleanup_task.done():
        _asset_cleanup_task = asyncio.create_task(_periodic_asset_cleanup())


async def stop_cleanup_task() -> None:
    global _asset_cleanup_task
    task = _asset_cleanup_task
    _asset_cleanup_task = None
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
