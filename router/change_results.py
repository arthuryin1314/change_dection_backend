import hashlib
import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from config.db_config import get_db
from crud import change_results as crud_results
from crud import images as crud_images
from crud import ml_models as crud_models
from services.change_orchestration import schedule_change_orchestration
from utils.change_result_errors import (
    CHANGE_RESULT_NOT_RETRYABLE,
    REQUEST_ID_CONFLICT,
)
from utils.classification_contract import (
    CLASSIFICATION_SCHEME_VERSION,
    GRID_POLICY_VERSION as CLASSIFICATION_GRID_POLICY_VERSION,
    PIPELINE_VERSION,
    inference_parameters,
    ordered_class_definitions,
)
from utils.get_user_by_token import get_current_user
from utils.response import api_response
from utils.transition_matrix import (
    CALCULATION_VERSION as ANALYSIS_CALCULATION_VERSION,
    GRID_POLICY_VERSION as ANALYSIS_GRID_POLICY_VERSION,
)


router = APIRouter(prefix="/api/change-results", tags=["change-results"])


class CreateChangeResultRequest(BaseModel):
    request_id: UUID
    before_image_id: int = Field(..., gt=0)
    after_image_id: int = Field(..., gt=0)
    model_id: int = Field(..., gt=0)


class RetryChangeResultRequest(BaseModel):
    request_id: UUID


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value):
    return value.isoformat() if value is not None else None


def _canonical_sha256(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _request_fingerprint(payload: CreateChangeResultRequest) -> str:
    return _canonical_sha256(
        {
            "before_image_id": payload.before_image_id,
            "after_image_id": payload.after_image_id,
            "model_id": payload.model_id,
        }
    )


def _source_descriptor(row, path_name: str, hash_prefix: str) -> dict:
    return {
        "name": row.image_name,
        "path": getattr(row, path_name),
        "cached_sha256": getattr(row, f"{hash_prefix}_sha256"),
        "cached_size": getattr(row, f"{hash_prefix}_sha256_size"),
        "cached_mtime_ns": getattr(row, f"{hash_prefix}_sha256_mtime_ns"),
    }


async def _submitted_inputs(
    db: AsyncSession,
    user_id: int,
    payload: CreateChangeResultRequest,
) -> dict:
    before = await crud_images.get_image_by_id(db, payload.before_image_id, user_id)
    after = await crud_images.get_image_by_id(db, payload.after_image_id, user_id)
    model = await crud_models.get_ml_model_by_id(db, payload.model_id, user_id)
    if before is None or after is None:
        raise HTTPException(status_code=404, detail="影像不存在")
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    if model.model_type != "semantic_segmentation" or model.framework != "PyTorch":
        raise HTTPException(status_code=422, detail="模型不兼容")
    return {
        "before_image_id": before.id,
        "after_image_id": after.id,
        "model_id": model.id,
        "before": _source_descriptor(before, "img_path", "content"),
        "after": _source_descriptor(after, "img_path", "content"),
        "model": {
            "name": model.model_name,
            "weight_path": model.weight_file_path,
            "cached_sha256": model.weight_content_sha256,
            "cached_size": model.weight_content_sha256_size,
            "cached_mtime_ns": model.weight_content_sha256_mtime_ns,
        },
        "identity_contract": {
            "inference_parameters": inference_parameters(),
            "classification_scheme_version": CLASSIFICATION_SCHEME_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "grid_policy_version": CLASSIFICATION_GRID_POLICY_VERSION,
        },
        "analysis_contract": {
            "calculation_version": ANALYSIS_CALCULATION_VERSION,
            "grid_policy_version": ANALYSIS_GRID_POLICY_VERSION,
        },
    }


def _preparation_key(user_id: int, submitted: dict) -> str:
    return _canonical_sha256({"user_id": user_id, "inputs": submitted})


def _period_status(row, period: str) -> dict:
    result_id = getattr(row, f"{period}_result_id")
    identity = getattr(row, f"{period}_identity_sha256")
    if result_id is not None:
        return {
            "status": "SUCCEEDED",
            "result_id": result_id,
            "identity_sha256": identity,
        }
    phase = row.phase
    if phase == f"ENSURING_{period.upper()}":
        return {"status": "PROCESSING"}
    return {"status": "PENDING"}


def _serialize(row, request_id: str) -> dict:
    data = {
        "result_id": row.id,
        "request_id": request_id,
        "status": row.status,
        "phase": row.phase,
        "inputs": {
            "before_image_id": row.before_image_id,
            "after_image_id": row.after_image_id,
            "model_id": row.source_model_id,
        },
        "periods": {
            "before": _period_status(row, "before"),
            "after": _period_status(row, "after"),
        },
        "started_at": _iso(row.started_at),
        "heartbeat_at": _iso(row.heartbeat_at),
        "completed_at": _iso(row.completed_at),
    }
    if row.status == crud_results.SUCCEEDED:
        data.update(
            {
                "classes": ordered_class_definitions(),
                "before": row.before_snapshot,
                "after": row.after_snapshot,
                "matrix_m2": row.matrix_m2,
                "common_valid_area_m2": row.common_valid_area_m2,
                "before_window": row.before_window,
                "after_window": row.after_window,
                "analysis": {
                    "calculation_version": row.calculation_version,
                    "grid_policy_version": row.grid_policy_version,
                    "identity_sha256": row.analysis_identity_sha256,
                    "resolution": (row.analysis_metadata or {}).get("resolution"),
                    **(row.analysis_metadata or {}),
                },
                "calculated_at": _iso(row.calculated_at),
            }
        )
        grid_values = (
            row.crs,
            row.transform,
            row.raster_width,
            row.raster_height,
            row.bounds,
        )
        if all(value is not None for value in grid_values):
            data["grid"] = {
                "crs": row.crs,
                "transform": row.transform,
                "width": row.raster_width,
                "height": row.raster_height,
                "bounds": row.bounds,
            }
    elif row.status == crud_results.FAILED:
        data.update(
            {
                "error_code": row.error_code,
                "error_message": row.error_message,
                "error_data": row.error_data,
            }
        )
    return data


def _history_result_status(row) -> str:
    if row.matrix_m2 is None or row.calculated_at is None:
        return "INCOMPLETE"
    return "AVAILABLE"


def _serialize_history_item(row) -> dict:
    return {
        "result_id": row.id,
        "request_id": row.request_id,
        "completed_at": _iso(row.completed_at),
        "calculated_at": _iso(row.calculated_at),
        "result_status": _history_result_status(row),
        "before": row.before_snapshot,
        "after": row.after_snapshot,
    }


@router.get("/history")
async def list_change_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    rows, total = await crud_results.list_succeeded_history(
        db,
        current_user.id,
        offset=(page - 1) * page_size,
        limit=page_size,
    )
    return api_response(
        200,
        "查询成功",
        {
            "items": [_serialize_history_item(row) for row in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
        },
    )


@router.get("/history/{result_id}")
async def get_change_history(
    result_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    row = await crud_results.get_succeeded_history_by_id(
        db,
        result_id,
        current_user.id,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="历史记录不存在或无权访问")
    if _history_result_status(row) == "INCOMPLETE":
        return api_response(
            409,
            "变化历史结果数据不完整",
            {"result_id": result_id, "status": "INCOMPLETE"},
        )
    return api_response(200, "查询成功", _serialize(row, row.request_id))


def _failure(row, request_id: str):
    details = dict(row.error_data or {})
    details.update(
        {
            "request_id": request_id,
            "status": row.status,
            "phase": row.phase,
            "error_code": row.error_code,
        }
    )
    return api_response(row.error_http_status, row.error_message, details)


def _response(row, request_id: str):
    if row.status == crud_results.PROCESSING:
        return api_response(202, "变化分析处理中", _serialize(row, request_id))
    if row.status == crud_results.FAILED:
        return _failure(row, request_id)
    return api_response(200, "变化分析完成", _serialize(row, request_id))


async def _claim(
    db: AsyncSession,
    *,
    request_id: str,
    request_fingerprint: str,
    submitted: dict,
    user_id: int,
    retry_of_request_id: str | None = None,
    before_result_id: str | None = None,
    before_identity_sha256: str | None = None,
    after_result_id: str | None = None,
    after_identity_sha256: str | None = None,
):
    owner = f"{uuid4().hex}"
    claim = await crud_results.claim_submission(
        db,
        request_id=request_id,
        request_fingerprint=request_fingerprint,
        preparation_key=_preparation_key(user_id, submitted),
        submitted_inputs=submitted,
        user_id=user_id,
        owner=owner,
        now=_now(),
        retry_of_request_id=retry_of_request_id,
        before_result_id=before_result_id,
        before_identity_sha256=before_identity_sha256,
        after_result_id=after_result_id,
        after_identity_sha256=after_identity_sha256,
    )
    if claim.request_conflict:
        await db.rollback()
        return api_response(
            409,
            "request_id 已用于另一组变化分析输入",
            {"request_id": request_id, "error_code": REQUEST_ID_CONFLICT},
        )
    await db.commit()
    if claim.should_start:
        schedule_change_orchestration(
            claim.row.id,
            user_id,
            claim.row.lease_owner,
        )
    return api_response(202, "变化分析任务已接收", _serialize(claim.row, request_id))


@router.post("")
async def create_change_result(
    payload: CreateChangeResultRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    user_id = current_user.id
    request_id = str(payload.request_id)
    fingerprint = _request_fingerprint(payload)
    binding = await crud_results.get_request_binding(db, request_id, user_id)
    if binding is not None:
        if binding.request_fingerprint == fingerprint:
            row = await crud_results.get_by_id(db, binding.change_result_id)
        elif binding.request_fingerprint.startswith("legacy:"):
            row = await crud_results.get_by_id(db, binding.change_result_id)
            matching_inputs = (
                row.before_image_id == payload.before_image_id
                and row.after_image_id == payload.after_image_id
                and row.source_model_id == payload.model_id
            )
            if not matching_inputs:
                return api_response(
                    409,
                    "request_id 已用于另一组变化分析输入",
                    {"request_id": request_id, "error_code": REQUEST_ID_CONFLICT},
                )
            binding.request_fingerprint = fingerprint
            await db.commit()
        else:
            return api_response(
                409,
                "request_id 已用于另一组变化分析输入",
                {"request_id": request_id, "error_code": REQUEST_ID_CONFLICT},
            )
        return api_response(202, "变化分析任务已接收", _serialize(row, request_id))
    submitted = await _submitted_inputs(db, user_id, payload)
    return await _claim(
        db,
        request_id=request_id,
        request_fingerprint=fingerprint,
        submitted=submitted,
        user_id=user_id,
    )


@router.post("/{request_id}/retry")
async def retry_change_result(
    request_id: UUID,
    payload: RetryChangeResultRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    user_id = current_user.id
    original_request_id = str(request_id)
    new_request_id = str(payload.request_id)
    if new_request_id == original_request_id:
        return api_response(
            409,
            "重试必须使用新的 request_id",
            {"request_id": original_request_id, "error_code": REQUEST_ID_CONFLICT},
        )
    await crud_results.expire_request(
        db,
        request_id=original_request_id,
        user_id=user_id,
        now=_now(),
    )
    await db.commit()
    row = await crud_results.get_by_request(db, original_request_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="变化分析结果不存在")
    if row.status != crud_results.FAILED:
        return api_response(
            409,
            "只有失败的变化分析可以重试",
            {
                "request_id": original_request_id,
                "error_code": CHANGE_RESULT_NOT_RETRYABLE,
            },
        )
    submitted = dict(row.submitted_inputs)
    if "before" not in submitted or "after" not in submitted or "model" not in submitted:
        return api_response(
            409,
            "原任务没有可重试的冻结输入描述",
            {
                "request_id": original_request_id,
                "error_code": CHANGE_RESULT_NOT_RETRYABLE,
            },
        )
    retry_payload = CreateChangeResultRequest(
        request_id=payload.request_id,
        before_image_id=submitted["before_image_id"],
        after_image_id=submitted["after_image_id"],
        model_id=submitted["model_id"],
    )
    return await _claim(
        db,
        request_id=new_request_id,
        request_fingerprint=_request_fingerprint(retry_payload),
        submitted=submitted,
        user_id=user_id,
        retry_of_request_id=original_request_id,
        before_result_id=row.before_result_id,
        before_identity_sha256=row.before_identity_sha256,
        after_result_id=row.after_result_id,
        after_identity_sha256=row.after_identity_sha256,
    )


@router.get("/{request_id}")
async def get_change_result(
    request_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    value = str(request_id)
    await crud_results.expire_request(
        db,
        request_id=value,
        user_id=current_user.id,
        now=_now(),
    )
    await db.commit()
    row = await crud_results.get_by_request(db, value, current_user.id)
    if row is None:
        raise HTTPException(status_code=404, detail="变化分析结果不存在")
    return _response(row, value)
