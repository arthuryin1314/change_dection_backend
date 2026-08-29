import logging
from pathlib import Path
from typing import NoReturn, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from config.db_config import get_db
from crud import images as crud_images
from router.image.image_upload_edit_workflow import (
    WorkflowConflictError,
    WorkflowNotFoundError,
    begin_upload_session,
    create_image_from_upload,
    delete_image_workflow,
    edit_image_workflow,
    save_upload_chunk,
    start_tmp_cleanup_task,
    stop_tmp_cleanup_task,
)
from schemas.images import (
    ImagePageData,
    ImagePageResponse,
    ImageRecordResponse,
    ImageResponse,
    UploadSessionData,
    UploadSessionRequest,
    UploadSessionResponse,
)
from utils.geoserver_utils import GEOSERVER_URL, GEOSERVER_WORKSPACE
from utils.get_user_by_token import get_current_user
from utils.response import success_response


router = APIRouter(prefix="/api/images", tags=["images"])
logger = logging.getLogger(__name__)


def _raise_http_error(exc: Exception) -> NoReturn:
    if isinstance(exc, WorkflowNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, WorkflowConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.exception("影像工作流执行失败")
    raise HTTPException(status_code=500, detail="影像处理失败，请稍后重试") from exc


def _session_data(session: dict) -> dict:
    return UploadSessionData.model_validate(session).model_dump(mode="json", by_alias=True)


def _to_public_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    normalized = str(path_value).replace("\\", "/")
    marker_index = normalized.lower().find("/uploads/")
    if marker_index >= 0:
        return normalized[marker_index + 1 :]
    if normalized.lower().startswith("uploads/"):
        return normalized
    return Path(normalized).name


def _pick_boundary_paths(image) -> dict:
    for boundary in getattr(image, "boundary_files", None) or []:
        if any(
            getattr(boundary, field, None)
            for field in ("shp_path", "dbf_path", "prj_path")
        ):
            return {
                field: _to_public_path(getattr(boundary, field, None))
                for field in ("shp_path", "dbf_path", "prj_path")
            }
    return {"shp_path": None, "dbf_path": None, "prj_path": None}


def _layer_info(image) -> dict:
    layer_name = getattr(image, "layer_name", None)
    wms_url = getattr(image, "wms_url", None)
    if layer_name or wms_url:
        return {"layer_name": layer_name, "wms_url": wms_url}
    if not image.region_code or not image.image_name:
        return {"layer_name": None, "wms_url": None}
    layer_name = f"{image.region_code}_{image.image_name}_{image.id}".replace(" ", "_")
    return {
        "layer_name": layer_name,
        "wms_url": (
            f"{GEOSERVER_URL}/{GEOSERVER_WORKSPACE}/wms"
            f"?service=WMS&version=1.1.0&request=GetMap"
            f"&layers={GEOSERVER_WORKSPACE}:{layer_name}&format=image/png"
        ),
    }


def _serialize_image(image) -> dict:
    return {
        "id": image.id,
        "image_name": image.image_name,
        "resolution": image.resolution,
        "capture_date": image.capture_date,
        "satellite": image.satellite,
        "image_type": image.image_type,
        "region_code": image.region_code,
        "img_path": _to_public_path(getattr(image, "img_path", None)),
        "bbox": getattr(image, "bbox", None),
        **_layer_info(image),
        **_pick_boundary_paths(image),
        "upload_time": image.upload_time,
    }


def _image_data(image) -> dict:
    return ImageResponse.model_validate(_serialize_image(image)).model_dump(mode="json")


@router.post("/uploads", response_model=UploadSessionResponse, summary="初始化或恢复影像上传")
async def begin_upload(
    payload: UploadSessionRequest,
    current_user=Depends(get_current_user),
):
    try:
        session = begin_upload_session(
            current_user.id,
            file_name=payload.file_name,
            file_size=payload.file_size,
            chunk_size=payload.chunk_size,
            total_chunks=payload.total_chunks,
            file_hash=payload.file_hash,
        )
    except Exception as exc:
        _raise_http_error(exc)
    return success_response(message="上传会话已就绪", data=_session_data(session))


@router.put(
    "/uploads/{upload_id}/chunks/{chunk_index}",
    response_model=UploadSessionResponse,
    summary="上传影像分片",
)
async def upload_chunk_part(
    upload_id: str,
    chunk_index: int,
    chunk: UploadFile = File(..., alias="chunk"),
    current_user=Depends(get_current_user),
):
    try:
        session = await save_upload_chunk(
            current_user.id,
            upload_id,
            chunk_index,
            chunk.file,
        )
    except Exception as exc:
        _raise_http_error(exc)
    return success_response(message="分片上传成功", data=_session_data(session))


@router.post("", response_model=ImageRecordResponse, summary="创建影像")
async def create_image(
    upload_id: str = Form(..., alias="uploadId"),
    image_name: str = Form(..., alias="imageName"),
    resolution: float = Form(...),
    capture_date: str = Form(..., alias="captureDate"),
    satellite: str = Form(...),
    image_type: str = Form(..., alias="imageType"),
    region_code: str = Form(..., alias="regionCode"),
    boundary_files: list[UploadFile] = File(..., alias="boundaryFiles"),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        image = await create_image_from_upload(
            db,
            current_user.id,
            upload_id,
            image_name,
            resolution,
            capture_date,
            satellite,
            image_type,
            region_code,
            [(item.filename or "", item.file) for item in boundary_files],
        )
    except Exception as exc:
        _raise_http_error(exc)
    return success_response(message="影像创建成功", data=_image_data(image))


@router.get("", response_model=ImagePageResponse, summary="查询影像目录")
async def query_images(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100, alias="pageSize"),
    keyword: Optional[str] = Query(None, max_length=100),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        items, total = await crud_images.query_images(
            db,
            current_user.id,
            page,
            page_size,
            keyword,
        )
    except Exception as exc:
        logger.exception("获取影像目录失败")
        raise HTTPException(status_code=500, detail="获取影像目录失败，请稍后重试") from exc
    data = ImagePageData(
        items=[_serialize_image(image) for image in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )
    return success_response(
        message="获取成功",
        data=data.model_dump(mode="json", by_alias=True),
    )


@router.put("/{image_id}", response_model=ImageRecordResponse, summary="编辑影像")
async def edit_image(
    image_id: int,
    upload_id: Optional[str] = Form(None, alias="uploadId"),
    image_name: Optional[str] = Form(None, alias="imageName"),
    resolution: Optional[float] = Form(None),
    capture_date: Optional[str] = Form(None, alias="captureDate"),
    satellite: Optional[str] = Form(None),
    image_type: Optional[str] = Form(None, alias="imageType"),
    region_code: Optional[str] = Form(None, alias="regionCode"),
    boundary_files: Optional[list[UploadFile]] = File(None, alias="boundaryFiles"),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        image = await edit_image_workflow(
            db,
            current_user.id,
            image_id,
            upload_id,
            image_name,
            resolution,
            capture_date,
            satellite,
            image_type,
            region_code,
            [(item.filename or "", item.file) for item in boundary_files]
            if boundary_files
            else None,
        )
    except Exception as exc:
        _raise_http_error(exc)
    return success_response(message="影像编辑成功", data=_image_data(image))


@router.get("/{image_id}", response_model=ImageRecordResponse, summary="根据ID获取影像")
async def get_image(
    image_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    image = await crud_images.get_image_by_id(db, image_id, current_user.id)
    if not image:
        raise HTTPException(status_code=404, detail="影像不存在")
    return success_response(message="获取成功", data=_image_data(image))


@router.delete("/delete/{image_id}", summary="删除影像")
async def delete_image(
    image_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        deleted = await delete_image_workflow(db, current_user.id, image_id)
    except Exception as exc:
        _raise_http_error(exc)
    if not deleted:
        raise HTTPException(status_code=404, detail="影像不存在")
    return success_response(message="影像删除成功", data={"id": image_id})
