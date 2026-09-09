import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from affine import Affine
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from config.db_config import AsyncSessionLocal, get_db
from crud import classification_results as crud_results
from crud import images as crud_images
from crud import ml_models as crud_ml_models
from crud.identification_results import SqlAlchemyClaimStore
from services.classification_generation import GenerationRequest, run_generation
from services.generation_lifecycle import SqlAlchemyGenerationLifecycle
from services.identification_results import (
    PROCESSING,
    SUCCEEDED,
    ResultIdentity,
    claim_identification_result,
)
from utils.classification_render import RenderBoundsError, render_classification_png
from utils.classification_result_lock import classification_result_lock
from utils.classification_contract import inference_parameters, ordered_class_definitions
from utils.classification_area import (
    summarize_classification_area_m2,
    validate_class_area_m2,
)
from utils.classification_storage import RasterGrid, StoredClassification, validate_stored_classification
from utils.content_hash import ContentHash, resolve_content_sha256
from utils.get_user_by_token import get_current_user
from utils.response import api_response


router = APIRouter(prefix="/api/identification-results", tags=["identification-results"])
logger = logging.getLogger(__name__)
RESULT_STORAGE_ROOT = Path(
    os.environ.get("CLASSIFICATION_RESULT_DIR", "uploads/classification_results")
)
_generation_tasks: dict[str, set[asyncio.Task]] = {}


class CreateIdentificationResultRequest(BaseModel):
    image_id: int = Field(..., gt=0)
    model_id: int = Field(..., gt=0)


@dataclass(frozen=True)
class ContentHashSource:
    path: str
    cached_sha256: str | None
    cached_size: int | None
    cached_mtime_ns: int | None


@dataclass(frozen=True)
class IdentificationResolution:
    status: str
    row: object | None = None
    reason: str | None = None


def _validate_sources(image, model) -> None:
    if not image.img_path or not os.path.isfile(image.img_path):
        raise HTTPException(status_code=422, detail="影像源文件不存在或不可访问")
    suffix = Path(model.weight_file_path).suffix.lower()
    if (
        model.model_type != "semantic_segmentation"
        or model.framework != "PyTorch"
        or suffix not in {".pth", ".pt"}
        or not os.path.isfile(model.weight_file_path)
    ):
        raise HTTPException(status_code=422, detail="模型不兼容")


async def _calculate_content_hashes(
    image: ContentHashSource,
    model: ContentHashSource,
) -> tuple[ContentHash, ContentHash]:
    return await asyncio.gather(
        asyncio.to_thread(
            resolve_content_sha256,
            image.path,
            cached_sha256=image.cached_sha256,
            cached_size=image.cached_size,
            cached_mtime_ns=image.cached_mtime_ns,
        ),
        asyncio.to_thread(
            resolve_content_sha256,
            model.path,
            cached_sha256=model.cached_sha256,
            cached_size=model.cached_size,
            cached_mtime_ns=model.cached_mtime_ns,
        ),
    )


async def _resolve_source_hashes(db, image, model) -> tuple[str, str]:
    image_hash, weight_hash = await _calculate_content_hashes(
        ContentHashSource(
            path=image.img_path,
            cached_sha256=image.content_sha256,
            cached_size=image.content_sha256_size,
            cached_mtime_ns=image.content_sha256_mtime_ns,
        ),
        ContentHashSource(
            path=model.weight_file_path,
            cached_sha256=model.weight_content_sha256,
            cached_size=model.weight_content_sha256_size,
            cached_mtime_ns=model.weight_content_sha256_mtime_ns,
        ),
    )
    image.content_sha256 = image_hash.sha256
    image.content_sha256_size = image_hash.size
    image.content_sha256_mtime_ns = image_hash.mtime_ns
    model.weight_content_sha256 = weight_hash.sha256
    model.weight_content_sha256_size = weight_hash.size
    model.weight_content_sha256_mtime_ns = weight_hash.mtime_ns
    await db.flush()
    return image_hash.sha256, weight_hash.sha256


def _generation_done(result_id: str, task: asyncio.Task) -> None:
    tasks = _generation_tasks.get(result_id)
    if tasks is not None:
        tasks.discard(task)
        if not tasks:
            _generation_tasks.pop(result_id)
    if task.cancelled():
        return
    exception = task.exception()
    if exception is not None:
        logger.error(
            "完整分类结果生成失败: result_id=%s",
            result_id,
            exc_info=(type(exception), exception, exception.__traceback__),
        )


def _schedule_generation(request: GenerationRequest, lease_owner: str) -> None:
    lifecycle = SqlAlchemyGenerationLifecycle(
        AsyncSessionLocal,
        request.result_id,
        lease_owner,
    )
    task = asyncio.create_task(run_generation(request, lifecycle))
    _generation_tasks.setdefault(request.result_id, set()).add(task)
    task.add_done_callback(lambda done: _generation_done(request.result_id, done))


def _stored_grid(row) -> tuple[StoredClassification, RasterGrid] | None:
    if (
        row.classes_path is None
        or row.valid_mask_path is None
        or row.crs is None
        or row.transform is None
        or row.raster_width is None
        or row.raster_height is None
    ):
        return None
    directory = Path(row.classes_path).parent
    return (
        StoredClassification(
            directory=directory,
            classes_path=Path(row.classes_path),
            valid_mask_path=Path(row.valid_mask_path),
        ),
        RasterGrid(
            width=row.raster_width,
            height=row.raster_height,
            crs=row.crs,
            transform=Affine(*row.transform),
        ),
    )


def _row_is_reusable(row, *, timeout_seconds: float = 2) -> bool:
    stored_grid = _stored_grid(row)
    if stored_grid is None:
        return False
    with classification_result_lock(
        stored_grid[0].directory,
        timeout_seconds=timeout_seconds,
    ):
        return validate_stored_classification(*stored_grid, verify_pixels=False)


async def is_result_reusable(db, result_id: str, user_id: int) -> bool:
    row = await crud_results.get_result_by_id(db, result_id, user_id)
    if row is None or row.status != SUCCEEDED:
        return False
    return await asyncio.to_thread(_row_is_reusable, row)


@router.post("")
async def create_identification_result(
    payload: CreateIdentificationResultRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    image = await crud_images.get_image_by_id(db, payload.image_id, current_user.id)
    if image is None:
        raise HTTPException(status_code=404, detail="影像不存在")
    model = await crud_ml_models.get_ml_model_by_id(db, payload.model_id, current_user.id)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    _validate_sources(image, model)

    image_sha256, weight_sha256 = await _resolve_source_hashes(db, image, model)
    identity = ResultIdentity(
        user_id=current_user.id,
        image_content_sha256=image_sha256,
        weight_content_sha256=weight_sha256,
        inference_parameters=inference_parameters(),
    )
    lease_owner = f"{os.getpid()}-{uuid4().hex}"
    store = SqlAlchemyClaimStore(
        db,
        source_image_id=image.id,
        source_model_id=model.id,
    )
    claim = await claim_identification_result(store, identity, lease_owner, _utc_now())
    await db.commit()

    if claim.record.status == SUCCEEDED:
        if await is_result_reusable(db, claim.record.result_id, current_user.id):
            return api_response(
                200,
                "识别结果已存在",
                {"result_id": claim.record.result_id, "status": SUCCEEDED},
            )
        await crud_results.invalidate_succeeded_result(
            db,
            claim.record.result_id,
            current_user.id,
        )
        await db.commit()
        claim = await claim_identification_result(store, identity, lease_owner, _utc_now())
        await db.commit()

    if claim.should_start:
        request = GenerationRequest(
            result_id=claim.record.result_id,
            image_path=image.img_path,
            weight_file_path=model.weight_file_path,
            weight_sha256=weight_sha256,
            storage_root=RESULT_STORAGE_ROOT,
        )
        _schedule_generation(request, lease_owner)

    return api_response(
        202,
        "识别结果生成中",
        {"result_id": claim.record.result_id, "status": PROCESSING},
    )


def _utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _iso(value):
    return value.isoformat() if value is not None else None


def _serialize_result(row) -> dict:
    data = {
        "result_id": row.id,
        "status": row.status,
        "source_image_id": row.source_image_id,
        "source_model_id": row.source_model_id,
        "identity_sha256": row.identity_sha256,
        "image_content_sha256": row.image_content_sha256,
        "weight_content_sha256": row.weight_content_sha256,
        "inference_parameters": row.inference_parameters,
        "classification_scheme_version": row.classification_scheme_version,
        "classes": ordered_class_definitions(),
        "pipeline_version": row.pipeline_version,
        "grid_policy_version": row.grid_policy_version,
        "started_at": _iso(row.started_at),
        "completed_at": _iso(row.completed_at),
        "failure_detail": row.failure_detail,
        "area_status": row.area_status,
        "class_area_m2": row.class_area_m2,
        "area_completed_at": _iso(row.area_completed_at),
        "area_failure_detail": row.area_failure_detail,
    }
    if row.status == SUCCEEDED:
        data["grid"] = {
            "crs": row.crs,
            "transform": row.transform,
            "width": row.raster_width,
            "height": row.raster_height,
            "resolution": row.resolution,
            "bounds": row.bounds,
        }
    return data


async def resolve_identification_for_change(
    db: AsyncSession,
    *,
    user_id: int,
    image_id: int,
    model_id: int,
) -> IdentificationResolution:
    image = await crud_images.get_image_by_id(db, image_id, user_id)
    if image is None:
        raise HTTPException(status_code=404, detail="影像不存在")
    model = await crud_ml_models.get_ml_model_by_id(db, model_id, user_id)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    _validate_sources(image, model)

    image_source = ContentHashSource(
        path=image.img_path,
        cached_sha256=image.content_sha256,
        cached_size=image.content_sha256_size,
        cached_mtime_ns=image.content_sha256_mtime_ns,
    )
    model_source = ContentHashSource(
        path=model.weight_file_path,
        cached_sha256=model.weight_content_sha256,
        cached_size=model.weight_content_sha256_size,
        cached_mtime_ns=model.weight_content_sha256_mtime_ns,
    )
    source_image_id = image.id
    source_model_id = model.id
    await db.rollback()
    image_hash, weight_hash = await _calculate_content_hashes(
        image_source,
        model_source,
    )
    identity = ResultIdentity(
        user_id=user_id,
        image_content_sha256=image_hash.sha256,
        weight_content_sha256=weight_hash.sha256,
        inference_parameters=inference_parameters(),
    )
    store = SqlAlchemyClaimStore(
        db,
        source_image_id=source_image_id,
        source_model_id=source_model_id,
    )
    matched = await store.get_by_identity(user_id, identity.sha256())
    if matched is None:
        previous = await crud_results.get_latest_by_sources(
            db,
            user_id,
            source_image_id,
            source_model_id,
        )
        reason = "VERSION_MISMATCH" if previous is not None else "MISSING"
        return IdentificationResolution(status="MISSING", reason=reason)

    row = await crud_results.get_result_by_id(db, matched.result_id, user_id)
    if row is None:
        return IdentificationResolution(status="MISSING", reason="MISSING")
    if row.status == PROCESSING:
        return IdentificationResolution(status=PROCESSING, row=row, reason=PROCESSING)
    if row.status == "FAILED":
        return IdentificationResolution(status="FAILED", row=row, reason=row.failure_detail)
    if not await asyncio.to_thread(_row_is_reusable, row):
        return IdentificationResolution(status="UNAVAILABLE", row=row, reason="UNAVAILABLE")
    return IdentificationResolution(status=SUCCEEDED, row=row)


@router.get("/resolve")
async def resolve_identification_result(
    image_id: int = Query(..., gt=0),
    model_id: int = Query(..., gt=0),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        resolution = await resolve_identification_for_change(
            db,
            user_id=current_user.id,
            image_id=image_id,
            model_id=model_id,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if resolution.status == "MISSING":
        return api_response(200, "未找到匹配的识别结果", {"status": "MISSING"})
    if resolution.status == PROCESSING:
        return api_response(200, "识别结果处理中", {"status": PROCESSING})
    if resolution.status == "FAILED":
        return api_response(
            200,
            "识别结果失败",
            {"status": "FAILED", "failure_detail": resolution.reason},
        )
    if resolution.status == "UNAVAILABLE":
        return api_response(200, "识别结果文件不可用", {"status": "UNAVAILABLE"})
    return api_response(
        200,
        "查询成功",
        {"status": SUCCEEDED, "result": _serialize_result(resolution.row)},
    )

def _serialize_area(
    result_id,
    identity_sha256,
    status,
    values,
    completed_at,
    failure_detail=None,
):
    return {
        "result_id": result_id,
        "identity_sha256": identity_sha256,
        "area_status": status,
        "class_area_m2": values,
        "area_completed_at": _iso(completed_at),
        "area_failure_detail": failure_detail,
    }


@router.post("/{result_id}/areas")
async def calculate_identification_result_areas(
    result_id: str,
    identity_sha256: str = Query(
        ...,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    ),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    user_id = current_user.id
    row = await crud_results.get_result_by_id(db, result_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="识别结果不存在")
    if row.identity_sha256 != identity_sha256:
        raise HTTPException(status_code=409, detail="识别结果身份已更新，请重新查询")
    if row.status != SUCCEEDED:
        raise HTTPException(status_code=409, detail="识别结果尚不可用")
    if row.area_status == SUCCEEDED:
        areas = validate_class_area_m2(row.class_area_m2)
        return api_response(
            200,
            "面积汇总已存在",
            _serialize_area(
                row.id,
                identity_sha256,
                SUCCEEDED,
                areas,
                row.area_completed_at,
            ),
        )
    if (
        row.classes_path is None
        or row.valid_mask_path is None
        or row.completed_at is None
    ):
        raise HTTPException(status_code=409, detail="识别结果文件尚不可用")

    classes_path = row.classes_path
    valid_mask_path = row.valid_mask_path
    lease_owner = row.lease_owner
    generation_completed_at = row.completed_at
    await db.rollback()
    try:
        areas = await asyncio.to_thread(
            summarize_classification_area_m2,
            classes_path,
            valid_mask_path,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        area_completed_at = _utc_now()
        changed = await crud_results.mark_area_failed(
            db,
            result_id=result_id,
            user_id=user_id,
            lease_owner=lease_owner,
            completed_at=generation_completed_at,
            detail=str(exc),
            area_completed_at=area_completed_at,
        )
        if not changed:
            await db.rollback()
            raise HTTPException(status_code=409, detail="识别结果已更新，请重新查询") from exc
        await db.commit()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    area_completed_at = _utc_now()
    changed = await crud_results.mark_area_succeeded(
        db,
        result_id=result_id,
        user_id=user_id,
        lease_owner=lease_owner,
        completed_at=generation_completed_at,
        class_area_m2=areas,
        area_completed_at=area_completed_at,
    )
    if not changed:
        await db.rollback()
        raise HTTPException(status_code=409, detail="识别结果已更新，请重新查询")
    await db.commit()
    return api_response(
        200,
        "面积统计完成",
        _serialize_area(
            result_id,
            identity_sha256,
            SUCCEEDED,
            areas,
            area_completed_at,
        ),
    )


@router.get("/{result_id}")
async def get_identification_result(
    result_id: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    row = await crud_results.get_result_by_id(db, result_id, current_user.id)
    if row is None:
        raise HTTPException(status_code=404, detail="识别结果不存在")
    if row.status == SUCCEEDED and not await asyncio.to_thread(_row_is_reusable, row):
        await crud_results.invalidate_succeeded_result(db, row.id, current_user.id)
        await db.commit()
        row.status = "FAILED"
        row.failure_detail = "识别结果文件缺失或损坏"
    return api_response(200, "查询成功", _serialize_result(row))


@router.get("/{result_id}/render")
async def render_identification_result(
    result_id: str,
    bbox: str,
    width: int = Query(..., ge=1, le=4096),
    height: int = Query(..., ge=1, le=4096),
    srs: str = "EPSG:4326",
    classes: list[int] | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    selected = [1, 2, 3, 4, 5] if classes is None else list(dict.fromkeys(classes))
    invalid = [class_id for class_id in selected if class_id < 0 or class_id > 5]
    if invalid:
        raise HTTPException(status_code=422, detail="classes 中的类别 ID 必须在 0-5 之间")

    row = await crud_results.get_result_by_id(db, result_id, current_user.id)
    if row is None:
        raise HTTPException(status_code=404, detail="识别结果不存在")
    if row.status != SUCCEEDED or not await asyncio.to_thread(_row_is_reusable, row):
        raise HTTPException(status_code=409, detail="识别结果尚不可用")
    try:
        def render_under_lock():
            with classification_result_lock(Path(row.classes_path).parent):
                return render_classification_png(
                    row.classes_path,
                    row.valid_mask_path,
                    bbox=bbox,
                    width=width,
                    height=height,
                    srs=srs,
                    classes=selected,
                )

        image = await asyncio.to_thread(render_under_lock)
    except (RenderBoundsError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(content=image, media_type="image/png")


async def stop_generation_tasks() -> None:
    tasks = [
        task
        for result_tasks in _generation_tasks.values()
        for task in result_tasks
    ]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
