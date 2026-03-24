from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging
import shutil
from pathlib import Path
import json
import os
import time
from uuid import uuid4

from pydantic import BaseModel

from config.db_config import get_db
from crud import images as crud_images
from schemas.images import ImageResponse
from utils.geoserver_utils import publish_geotiff_layer, get_tif_bbox_wgs84
from utils.response import success_response
from utils.date_parser import parse_capture_date
from utils.get_user_by_token import get_current_user
from utils.geoserver_utils import GEOSERVER_URL, GEOSERVER_WORKSPACE
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/images", tags=["images"])

# 配置上传目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

IMAGE_DIR = UPLOAD_DIR / "images"
IMAGE_DIR.mkdir(exist_ok=True)

SHAPEFILE_DIR = UPLOAD_DIR / "shapefiles"
SHAPEFILE_DIR.mkdir(exist_ok=True)

TMP_UPLOAD_DIR = UPLOAD_DIR / "tmp"
TMP_UPLOAD_DIR.mkdir(exist_ok=True)

SESSION_META_FILE = "session.json"
CHUNKS_DIR_NAME = "chunks"
COMPLETE_LOCK_FILE = ".complete.lock"
UPLOAD_TTL_SECONDS = 24 * 60 * 60


class UploadInitRequest(BaseModel):
    file_hash: str
    file_name: str
    file_size: int
    chunk_size: int
    total_chunks: int


def _session_dir(upload_id: str) -> Path:
    return TMP_UPLOAD_DIR / upload_id


def _meta_path(upload_id: str) -> Path:
    return _session_dir(upload_id) / SESSION_META_FILE


def _chunks_dir(upload_id: str) -> Path:
    return _session_dir(upload_id) / CHUNKS_DIR_NAME


def _save_meta(upload_id: str, meta: dict) -> None:
    meta["updated_at"] = int(time.time())
    meta_file = _meta_path(upload_id)
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = meta_file.with_suffix(".tmp")
    temp_file.write_text(json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8")
    temp_file.replace(meta_file)


def _load_meta(upload_id: str) -> dict:
    meta_file = _meta_path(upload_id)
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="upload_id 不存在")
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取上传会话失败: {str(exc)}")


def _list_uploaded_chunks(upload_id: str) -> List[int]:
    chunks_dir = _chunks_dir(upload_id)
    if not chunks_dir.exists():
        return []

    result: List[int] = []
    for part_file in chunks_dir.glob("*.part"):
        try:
            result.append(int(part_file.stem))
        except ValueError:
            continue
    return sorted(result)


def _cleanup_expired_tmp_uploads() -> None:
    now_ts = time.time()
    for upload_dir in TMP_UPLOAD_DIR.iterdir():
        if not upload_dir.is_dir():
            continue

        meta_file = upload_dir / SESSION_META_FILE
        expire_base = meta_file if meta_file.exists() else upload_dir
        age = now_ts - expire_base.stat().st_mtime

        if age <= UPLOAD_TTL_SECONDS:
            continue

        # 已完成会话不清理，保留用于 complete 幂等返回
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("status") == "completed":
                    continue
            except Exception:
                pass

        shutil.rmtree(upload_dir, ignore_errors=True)


def _find_upload_id_by_hash(file_hash: str, user_id: int) -> Optional[str]:
    latest_upload_id = None
    latest_mtime = -1.0

    for upload_dir in TMP_UPLOAD_DIR.iterdir():
        if not upload_dir.is_dir():
            continue

        meta_file = upload_dir / SESSION_META_FILE
        if not meta_file.exists():
            continue

        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        if meta.get("file_hash") != file_hash or meta.get("user_id") != user_id:
            continue

        mtime = meta_file.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_upload_id = upload_dir.name

    return latest_upload_id


def _save_upload_stream(src, dst, chunk_size: int = 1024 * 1024) -> None:
    while True:
        chunk = src.read(chunk_size)
        if not chunk:
            break
        dst.write(chunk)


async def _save_shp_and_create_records(
    db: AsyncSession,
    user_id: int,
    image_name: str,
    resolution: float,
    capture_date: str,
    satellite: str,
    image_type: str,
    region_code: str,
    tif_path: Path,
    shp_files: List[UploadFile],
    bbox: Optional[List[float]] = None,
    layer_name: Optional[str] = None,
    wms_url: Optional[str] = None,
):
    parsed_capture_date = parse_capture_date(capture_date)

    shp_dir = SHAPEFILE_DIR / f"{region_code}_{image_name}"
    shp_dir.mkdir(exist_ok=True)

    shp_path = None
    dbf_path = None
    prj_path = None

    for shp_file in shp_files:
        if not shp_file.filename:
            continue

        save_path = shp_dir / Path(shp_file.filename).name
        with save_path.open("wb") as buffer:
            _save_upload_stream(shp_file.file, buffer)

        lower_name = shp_file.filename.lower()
        if lower_name.endswith(".shp"):
            shp_path = str(save_path)
        elif lower_name.endswith(".dbf"):
            dbf_path = str(save_path)
        elif lower_name.endswith(".prj"):
            prj_path = str(save_path)

    db_image = await crud_images.create_image(
        db=db,
        user_id=user_id,
        image_name=image_name,
        resolution=resolution,
        capture_date=parsed_capture_date,
        satellite=satellite,
        image_type=image_type,
        region_code=region_code,
        img_path=str(tif_path),
        bbox=bbox,
        layer_name=layer_name,
        wms_url=wms_url,
    )

    await crud_images.create_boundary_files(
        db=db,
        image_id=db_image.id,
        file_prefix=f"{region_code}_{image_name}",
        shp_path=shp_path,
        dbf_path=dbf_path,
        prj_path=prj_path,
    )

    return db_image


def _pick_boundary_paths(image) -> dict:
    boundary_files = getattr(image, "boundary_files", None) or []
    for bf in boundary_files:
        if getattr(bf, "shp_path", None) or getattr(bf, "dbf_path", None) or getattr(bf, "prj_path", None):
            return {
                "shp_path": getattr(bf, "shp_path", None),
                "dbf_path": getattr(bf, "dbf_path", None),
                "prj_path": getattr(bf, "prj_path", None),
            }
    return {"shp_path": None, "dbf_path": None, "prj_path": None}


def _build_layer_name(region_code: Optional[str], image_name: Optional[str], image_id: Optional[int]) -> Optional[str]:
    if not region_code or not image_name or image_id is None:
        return None
    return f"{region_code}_{image_name}_{image_id}".replace(" ", "_")


def _build_wms_url(layer_name: Optional[str]) -> Optional[str]:
    if not layer_name:
        return None
    return (
        f"{GEOSERVER_URL}/{GEOSERVER_WORKSPACE}/wms"
        f"?service=WMS&version=1.1.0&request=GetMap"
        f"&layers={GEOSERVER_WORKSPACE}:{layer_name}"
        f"&format=image/png"
    )


def _build_layer_info(image) -> dict:
    stored_layer_name = getattr(image, "layer_name", None)
    stored_wms_url = getattr(image, "wms_url", None)
    if stored_layer_name or stored_wms_url:
        return {"layer_name": stored_layer_name, "wms_url": stored_wms_url}

    layer_name = _build_layer_name(image.region_code, image.image_name, image.id)
    return {"layer_name": layer_name, "wms_url": _build_wms_url(layer_name)}


def _serialize_image(image) -> dict:
    return {
        "id": image.id,
        "image_name": image.image_name,
        "resolution": image.resolution,
        "capture_date": image.capture_date,
        "satellite": image.satellite,
        "image_type": image.image_type,
        "region_code": image.region_code,
        "img_path": image.img_path,
        "bbox": getattr(image, "bbox", None),
        **_build_layer_info(image),
        **_pick_boundary_paths(image),
        "upload_time": image.upload_time,
    }


@router.post("/upload/init", summary="初始化分片上传")
async def init_upload(payload: UploadInitRequest, current_user=Depends(get_current_user)):
    _cleanup_expired_tmp_uploads()

    if payload.file_size <= 0 or payload.chunk_size <= 0 or payload.total_chunks <= 0:
        raise HTTPException(status_code=422, detail="file_size/chunk_size/total_chunks 必须大于 0")

    upload_id = uuid4().hex
    upload_dir = _session_dir(upload_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    _chunks_dir(upload_id).mkdir(parents=True, exist_ok=True)

    _save_meta(
        upload_id,
        {
            "upload_id": upload_id,
            "user_id": current_user.id,
            "file_hash": payload.file_hash,
            "file_name": payload.file_name,
            "file_size": payload.file_size,
            "chunk_size": payload.chunk_size,
            "total_chunks": payload.total_chunks,
            "status": "uploading",
            "created_at": int(time.time()),
        },
    )

    return success_response(message="初始化成功", data={"upload_id": upload_id})


@router.get("/upload/status", summary="查询分片上传状态")
async def get_upload_status(
    upload_id: Optional[str] = Query(None),
    file_hash: Optional[str] = Query(None),
    current_user=Depends(get_current_user),
):
    _cleanup_expired_tmp_uploads()

    resolved_upload_id = upload_id
    if not resolved_upload_id and file_hash:
        resolved_upload_id = _find_upload_id_by_hash(file_hash, current_user.id)

    if not resolved_upload_id:
        raise HTTPException(status_code=404, detail="未找到上传会话")

    meta = _load_meta(resolved_upload_id)
    if meta.get("user_id") != current_user.id:
        raise HTTPException(status_code=404, detail="未找到上传会话")
    uploaded_chunks = _list_uploaded_chunks(resolved_upload_id)

    return success_response(
        message="获取状态成功",
        data={
            "upload_id": resolved_upload_id,
            "file_hash": meta.get("file_hash"),
            "status": meta.get("status", "uploading"),
            "uploaded_chunks": uploaded_chunks,
            "total_chunks": meta.get("total_chunks"),
        },
    )


@router.post("/upload/chunk", summary="上传单个分片")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk: Optional[UploadFile] = File(None, alias="chunk"),
    chunk_file: Optional[UploadFile] = File(None, alias="chunk_file"),
    current_user=Depends(get_current_user),
):
    _cleanup_expired_tmp_uploads()

    meta = _load_meta(upload_id)
    if meta.get("user_id") != current_user.id:
        raise HTTPException(status_code=404, detail="upload_id 不存在")
    if meta.get("status") == "completed":
        return success_response(message="上传已完成，分片上传跳过", data={"chunk_index": chunk_index})

    if chunk_index < 0 or chunk_index >= int(meta["total_chunks"]):
        raise HTTPException(status_code=422, detail="chunk_index 超出范围")

    chunk_data = chunk or chunk_file
    if not chunk_data:
        raise HTTPException(status_code=422, detail="缺少 chunk 文件")

    part_path = _chunks_dir(upload_id) / f"{chunk_index}.part"
    part_path.parent.mkdir(parents=True, exist_ok=True)

    with part_path.open("wb") as buffer:
        _save_upload_stream(chunk_data.file, buffer)

    meta["status"] = "uploading"
    _save_meta(upload_id, meta)

    return success_response(message="分片上传成功", data={"upload_id": upload_id, "chunk_index": chunk_index})


def _serialize_image_json(image) -> dict:
    return ImageResponse.model_validate(_serialize_image(image)).model_dump(mode="json")


@router.post("/upload/complete", summary="完成分片上传并入库")
async def complete_upload(
    upload_id: str = Form(...),
    image_name: str = Form(...),
    resolution: float = Form(...),
    capture_date: str = Form(...),
    satellite: str = Form(...),
    image_type: str = Form(..., alias="type"),
    region_code: str = Form(...),
    shp_files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    _cleanup_expired_tmp_uploads()

    meta = _load_meta(upload_id)
    if meta.get("user_id") != current_user.id:
        raise HTTPException(status_code=404, detail="upload_id 不存在")

    # complete 幂等：重复调用直接返回上次结果
    if meta.get("status") == "completed" and meta.get("result"):
        return success_response(message="上传已完成", data=meta["result"])

    lock_path = _session_dir(upload_id) / COMPLETE_LOCK_FILE
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        raise HTTPException(status_code=409, detail="该上传正在合并，请稍后重试")

    tif_path: Optional[Path] = None
    db_image = None
    try:
        total_chunks = int(meta["total_chunks"])
        chunk_size = int(meta["chunk_size"])
        file_size = int(meta["file_size"])

        uploaded = _list_uploaded_chunks(upload_id)
        required = list(range(total_chunks))
        if uploaded != required:
            missing = sorted(set(required) - set(uploaded))
            raise HTTPException(status_code=422, detail=f"分片不完整，缺失: {missing[:20]}")

        merged_size = 0
        for idx in required:
            part_path = _chunks_dir(upload_id) / f"{idx}.part"
            part_size = part_path.stat().st_size

            if idx < total_chunks - 1 and part_size != chunk_size:
                raise HTTPException(status_code=422, detail=f"分片 {idx} 大小不正确")
            if idx == total_chunks - 1 and not (0 < part_size <= chunk_size):
                raise HTTPException(status_code=422, detail="最后一片大小不正确")

            merged_size += part_size

        if merged_size != file_size:
            raise HTTPException(status_code=422, detail="分片总大小与文件大小不一致")

        original_name = Path(str(meta.get("file_name", "merged.tif"))).name
        if not original_name.lower().endswith((".tif", ".tiff")):
            raise HTTPException(status_code=422, detail="最终文件必须是 .tif/.tiff")

        tif_filename = f"{region_code}_{image_name}_{original_name}"
        tif_path = IMAGE_DIR / tif_filename
        with tif_path.open("wb") as merged_file:
            for idx in required:
                part_path = _chunks_dir(upload_id) / f"{idx}.part"
                with part_path.open("rb") as part_file:
                    _save_upload_stream(part_file, merged_file)

        if tif_path.stat().st_size != file_size:
            raise HTTPException(status_code=422, detail="合并后文件大小校验失败")

        bbox = get_tif_bbox_wgs84(tif_path)
        db_image = await _save_shp_and_create_records(
            db=db,
            user_id=current_user.id,
            image_name=image_name,
            resolution=resolution,
            capture_date=capture_date,
            satellite=satellite,
            image_type=image_type,
            region_code=region_code,
            tif_path=tif_path,
            shp_files=shp_files,
            bbox=bbox,
        )
        await db.commit()

        layer_name = _build_layer_name(region_code, image_name, db_image.id)
        wms_url = None
        published = False
        geoserver_error = None

        if layer_name and tif_path is not None:
            try:
                wms_url = await publish_geotiff_layer(tif_path=tif_path, layer_name=layer_name)
                await crud_images.update_image_fields(
                    db,
                    db_image,
                    {"layer_name": layer_name, "wms_url": wms_url},
                )
                await db.commit()
                published = True
            except Exception as exc:
                await db.rollback()
                geoserver_error = str(exc)
                logger.exception(
                    "GeoServer publish failed in chunk complete: upload_id=%s image_id=%s layer_name=%s user_id=%s",
                    upload_id,
                    getattr(db_image, "id", None),
                    layer_name,
                    getattr(current_user, "id", None),
                )

        refreshed = await crud_images.get_image_by_id(db, db_image.id, current_user.id)
        payload_image = refreshed or db_image
        result = {
            **_serialize_image_json(payload_image),
            "upload_id": upload_id,
            "published": published,
        }
        if geoserver_error:
            result["geoserver_error"] = geoserver_error
        meta["status"] = "completed"
        meta["result"] = result
        _save_meta(upload_id, meta)

        msg = "影像上传成功" if published else "影像已入库，GeoServer 图层发布失败，请稍后手动发布"
        return success_response(message=msg, data=result)

    except HTTPException:
        await db.rollback()
        raise
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        await db.rollback()
        # GeoServer 发布失败或其他提交后异常：主记录可能已入库，尽量返回降级成功结果
        if db_image is not None:
            logger.exception(
                "Upload complete fallback triggered: upload_id=%s image_id=%s layer_name=%s user_id=%s",
                upload_id,
                getattr(db_image, "id", None),
                _build_layer_name(region_code, image_name, getattr(db_image, "id", None)),
                getattr(current_user, "id", None),
            )
            payload_image = await crud_images.get_image_by_id(db, db_image.id, current_user.id)
            payload_image = payload_image or db_image
            return success_response(
                message="影像已入库，GeoServer 图层发布失败，请稍后手动发布",
                data={
                    **(_serialize_image_json(payload_image) if payload_image else {}),
                    "published": False,
                    "geoserver_error": str(exc),
                },
            )
        # 入库前出错：清理已落盘的 TIF
        if tif_path and tif_path.exists():
            tif_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"上传失败: {str(exc)}")


@router.post("/upload", summary="上传影像（整文件，兼容旧流程）")
async def upload_image(
    image_name: str = Form(...),
    resolution: float = Form(...),
    capture_date: str = Form(...),
    satellite: str = Form(...),
    image_type: str = Form(..., alias="type"),
    region_code: str = Form(...),
    image_file: UploadFile = File(...),
    shp_files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    tif_path: Optional[Path] = None
    db_image = None

    try:
        if not image_file.filename or not image_file.filename.lower().endswith((".tif", ".tiff")):
            raise ValueError("image_file 必须是 .tif 或 .tiff 文件")

        # ── 1. 保存 TIF ──────────────────────────────────────────────────
        tif_filename = f"{region_code}_{image_name}_{Path(image_file.filename).name}"
        tif_path = IMAGE_DIR / tif_filename        # E:\change_detection\...\images\xxx.tif
        with tif_path.open("wb") as buffer:
            _save_upload_stream(image_file.file, buffer)

        bbox = get_tif_bbox_wgs84(tif_path)

        # ── 2. 写入数据库 ────────────────────────────────────────────────
        db_image = await _save_shp_and_create_records(
            db=db,
            user_id=current_user.id,
            image_name=image_name,
            resolution=resolution,
            capture_date=capture_date,
            satellite=satellite,
            image_type=image_type,
            region_code=region_code,
            tif_path=tif_path,
            shp_files=shp_files,
            bbox=bbox,
        )
        await db.commit()

        # ── 3. 发布到 GeoServer ──────────────────────────────────────────
        layer_name = _build_layer_name(region_code, image_name, db_image.id)
        wms_url = None

        if not layer_name:
            raise RuntimeError("无法生成图层名称")

        wms_url = await publish_geotiff_layer(
            tif_path=tif_path,
            layer_name=layer_name,
        )
        await crud_images.update_image_fields(
            db,
            db_image,
            {"layer_name": layer_name, "wms_url": wms_url},
        )
        await db.commit()

        refreshed = await crud_images.get_image_by_id(db, db_image.id, current_user.id)
        payload_image = refreshed or db_image
        return success_response(
            message="影像上传成功",
            data={
                **_serialize_image_json(payload_image),
                "published": True,
            },
        )

    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception as exc:
        await db.rollback()
        # GeoServer 发布失败或其他提交后异常：主记录可能已入库，尽量返回降级成功结果
        if db_image is not None:
            logger.exception(
                "Upload fallback triggered: image_id=%s layer_name=%s user_id=%s",
                getattr(db_image, "id", None),
                _build_layer_name(region_code, image_name, getattr(db_image, "id", None)),
                getattr(current_user, "id", None),
            )
            payload_image = await crud_images.get_image_by_id(db, db_image.id, current_user.id)
            payload_image = payload_image or db_image
            return success_response(
                message="影像已入库，GeoServer 图层发布失败，请稍后手动发布",
                data={
                    **(_serialize_image_json(payload_image) if payload_image else {}),
                    "published": False,
                    "geoserver_error": str(exc),
                },
            )
        # 入库前出错：清理已落盘的 TIF
        if tif_path and tif_path.exists():
            tif_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"上传失败: {str(exc)}")


@router.get("/list", summary="获取分页影像列表")
async def get_images_list(
        page:int=Query(1,ge=1,description='页码从1开始'),
        page_size:int=Query(10,ge=1,description='每页默认展示10条',alias='pageSize'),
        db: AsyncSession = Depends(get_db),
        current_user=Depends(get_current_user)
):
    """
    获取当前用户的影像记录，按上传时间降序排列
    需要Bearer token认证
    """
    try:
        offset = (page - 1) * page_size
        images,total_count = await crud_images.get_paginated_images(db, current_user.id,offset, page_size)
        data = [
            ImageResponse.model_validate(_serialize_image(image)).model_dump(mode="json")
            for image in images
        ]
        return success_response(message='获取成功',data={
            "items":data,
            "total":total_count,
            "page":page,
            "page_size":page_size,
            "total_pages":(total_count + page_size - 1) // page_size

        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取影像列表失败: {str(exc)}")


@router.get("/search", summary="搜索影像")
async def search_images(
    q: Optional[str] = Query(None, description="影像名称关键字"),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """按名称模糊搜索当前用户影像，返回数组。"""
    try:
        images = await crud_images.search_images(db, current_user.id, q)
        data = [
            ImageResponse.model_validate(_serialize_image(image)).model_dump(mode="json")
            for image in images
        ]
        return success_response(message="搜索成功", data=data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"搜索影像失败: {str(exc)}")


@router.get("/{image_id}", response_model=ImageResponse, summary="根据ID获取影像")
async def get_image(image_id: int, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """
    根据ID获取当前用户的影像记录
    """
    image = await crud_images.get_image_by_id(db, image_id, current_user.id)
    if not image:
        raise HTTPException(status_code=404, detail="影像不存在")
    return ImageResponse.model_validate(_serialize_image(image))


def _safe_unlink(file_path: Optional[str]) -> None:
    if not file_path:
        return
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception:
        pass


def _save_upload_file(upload: UploadFile, target_dir: Path, filename: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    save_path = target_dir / filename
    with save_path.open("wb") as buffer:
        _save_upload_stream(upload.file, buffer)
    return save_path


def _extract_shp_group(
    shp_files: Optional[List[UploadFile]],
    shp_file: Optional[UploadFile],
    dbf_file: Optional[UploadFile],
    prj_file: Optional[UploadFile],
) -> dict:
    result = {
        "shp": shp_file,
        "dbf": dbf_file,
        "prj": prj_file,
    }
    for file_item in shp_files or []:
        if not file_item or not file_item.filename:
            continue
        lower_name = file_item.filename.lower()
        if lower_name.endswith(".shp"):
            result["shp"] = file_item
        elif lower_name.endswith(".dbf"):
            result["dbf"] = file_item
        elif lower_name.endswith(".prj"):
            result["prj"] = file_item
    return result


@router.put("/update/{image_id}", summary="编辑影像")
async def edit_image(
    image_id: int,
    image_name: Optional[str] = Form(None, description="影像名称"),
    resolution: Optional[float] = Form(None, description="影像分辨率"),
    capture_date: Optional[str] = Form(None, description="拍摄日期，支持 YYYY-MM-DD 或 JS Date 字符串"),
    satellite: Optional[str] = Form(None, description="卫星名称"),
    image_type: Optional[str] = Form(None, alias="type", description="影像类型"),
    region_code: Optional[str] = Form(None, description="区域代码"),
    image_file: Optional[UploadFile] = File(None, description="遥感影像文件（可选，.tif/.tiff）"),
    shp_files: Optional[List[UploadFile]] = File(None, description="边界文件（可选，支持 .shp/.dbf/.prj）"),
    shp_file: Optional[UploadFile] = File(None, description="单独上传 .shp（可选）"),
    dbf_file: Optional[UploadFile] = File(None, description="单独上传 .dbf（可选）"),
    prj_file: Optional[UploadFile] = File(None, description="单独上传 .prj（可选）"),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    image = await crud_images.get_image_by_id(db, image_id, current_user.id)
    if not image:
        raise HTTPException(status_code=404, detail="影像不存在")

    new_files: List[Path] = []
    old_files_to_delete: List[str] = []

    try:
        updates = {}
        if image_name is not None:
            updates["image_name"] = image_name
        if resolution is not None:
            updates["resolution"] = resolution
        if capture_date is not None:
            updates["capture_date"] = parse_capture_date(capture_date)
        if satellite is not None:
            updates["satellite"] = satellite
        if image_type is not None:
            updates["image_type"] = image_type
        if region_code is not None:
            updates["region_code"] = region_code

        final_image_name = image_name if image_name is not None else image.image_name
        final_region_code = region_code if region_code is not None else image.region_code
        file_prefix = f"{final_region_code}_{final_image_name}"

        if image_file is not None:
            if not image_file.filename or not image_file.filename.lower().endswith((".tif", ".tiff")):
                raise HTTPException(status_code=422, detail="image_file 必须是 .tif 或 .tiff 文件")

            tif_filename = f"{file_prefix}_{image.id}_{Path(image_file.filename).name}"
            tif_path = _save_upload_file(image_file, IMAGE_DIR, tif_filename)
            new_files.append(tif_path)

            if image.img_path and image.img_path != str(tif_path):
                old_files_to_delete.append(image.img_path)
            updates["img_path"] = str(tif_path)
            updates["bbox"] = get_tif_bbox_wgs84(tif_path)

        shp_group = _extract_shp_group(shp_files, shp_file, dbf_file, prj_file)
        boundary_updates = {}
        shp_dir = SHAPEFILE_DIR / file_prefix

        if shp_group["shp"] is not None:
            shp_upload = shp_group["shp"]
            if not shp_upload.filename or not shp_upload.filename.lower().endswith(".shp"):
                raise HTTPException(status_code=422, detail="shp_file 必须是 .shp 文件")
            shp_path = _save_upload_file(shp_upload, shp_dir, Path(shp_upload.filename).name)
            new_files.append(shp_path)
            boundary_updates["shp_path"] = str(shp_path)

        if shp_group["dbf"] is not None:
            dbf_upload = shp_group["dbf"]
            if not dbf_upload.filename or not dbf_upload.filename.lower().endswith(".dbf"):
                raise HTTPException(status_code=422, detail="dbf_file 必须是 .dbf 文件")
            dbf_path = _save_upload_file(dbf_upload, shp_dir, Path(dbf_upload.filename).name)
            new_files.append(dbf_path)
            boundary_updates["dbf_path"] = str(dbf_path)

        if shp_group["prj"] is not None:
            prj_upload = shp_group["prj"]
            if not prj_upload.filename or not prj_upload.filename.lower().endswith(".prj"):
                raise HTTPException(status_code=422, detail="prj_file 必须是 .prj 文件")
            prj_path = _save_upload_file(prj_upload, shp_dir, Path(prj_upload.filename).name)
            new_files.append(prj_path)
            boundary_updates["prj_path"] = str(prj_path)

        old_boundary = _pick_boundary_paths(image)
        if updates:
            await crud_images.update_image_fields(db, image, updates)

        if boundary_updates or updates:
            await crud_images.upsert_boundary_files(
                db,
                image,
                file_prefix=file_prefix,
                shp_path=boundary_updates.get("shp_path"),
                dbf_path=boundary_updates.get("dbf_path"),
                prj_path=boundary_updates.get("prj_path"),
            )

        await db.commit()

        if "shp_path" in boundary_updates and old_boundary.get("shp_path") and old_boundary.get("shp_path") != boundary_updates["shp_path"]:
            old_files_to_delete.append(old_boundary["shp_path"])
        if "dbf_path" in boundary_updates and old_boundary.get("dbf_path") and old_boundary.get("dbf_path") != boundary_updates["dbf_path"]:
            old_files_to_delete.append(old_boundary["dbf_path"])
        if "prj_path" in boundary_updates and old_boundary.get("prj_path") and old_boundary.get("prj_path") != boundary_updates["prj_path"]:
            old_files_to_delete.append(old_boundary["prj_path"])

        for old_file in old_files_to_delete:
            _safe_unlink(old_file)

        refreshed = await crud_images.get_image_by_id(db, image_id, current_user.id)
        if not refreshed:
            raise HTTPException(status_code=404, detail="影像不存在")

        image_payload = ImageResponse.model_validate(_serialize_image(refreshed))
        return success_response(message="影像编辑成功", data=image_payload.model_dump(mode="json"))

    except HTTPException:
        await db.rollback()
        for file_path in new_files:
            _safe_unlink(str(file_path))
        raise
    except Exception as exc:
        await db.rollback()
        for file_path in new_files:
            _safe_unlink(str(file_path))
        raise HTTPException(status_code=500, detail=f"编辑影像失败: {str(exc)}")


@router.delete('/delete/{image_id}', summary='删除影像')
async def delete_image(
    image_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        deleted = await crud_images.delete_image_with_files(db, image_id, current_user.id)
        if not deleted:
            raise HTTPException(status_code=404, detail="影像不存在")

        await db.commit()

        _safe_unlink(deleted.get("img_path"))
        for file_path in deleted.get("boundary_paths", []):
            _safe_unlink(file_path)

        return success_response(
            message="影像删除成功",
            data={"id": image_id},
        )
    except HTTPException:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"删除影像失败: {str(exc)}")
