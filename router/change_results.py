import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from config.db_config import AsyncSessionLocal, get_db
from crud import change_results as crud_results
from router.identification_results import resolve_identification_for_change
from utils.change_result_errors import (
    CHANGE_RESULT_FAILED,
    IDENTIFICATION_RESULT_BUSY,
    IDENTIFICATION_RESULT_UNAVAILABLE,
    REQUEST_ID_CONFLICT,
)
from utils.classification_contract import ordered_class_definitions
from utils.get_user_by_token import get_current_user
from utils.response import api_response
from utils.transition_matrix import TransitionMatrixError, compute_transition_matrix_m2


router = APIRouter(prefix="/api/change-results", tags=["change-results"])
HEARTBEAT_INTERVAL_SECONDS = 15


class CreateChangeResultRequest(BaseModel):
    request_id: UUID
    before_image_id: int = Field(..., gt=0)
    after_image_id: int = Field(..., gt=0)
    model_id: int = Field(..., gt=0)
    before_result_id: str = Field(..., min_length=1, max_length=32)
    before_identity_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    after_result_id: str = Field(..., min_length=1, max_length=32)
    after_identity_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ResolvedPeriod:
    id: str
    identity_sha256: str
    source_image_id: int
    source_model_id: int
    image_content_sha256: str
    weight_content_sha256: str
    inference_parameters: dict
    classification_scheme_version: str
    pipeline_version: str
    grid_policy_version: str
    completed_at: datetime
    classes_path: str
    valid_mask_path: str

    @classmethod
    def from_row(cls, row):
        return cls(
            id=row.id,
            identity_sha256=row.identity_sha256,
            source_image_id=row.source_image_id,
            source_model_id=row.source_model_id,
            image_content_sha256=row.image_content_sha256,
            weight_content_sha256=row.weight_content_sha256,
            inference_parameters=dict(row.inference_parameters),
            classification_scheme_version=row.classification_scheme_version,
            pipeline_version=row.pipeline_version,
            grid_policy_version=row.grid_policy_version,
            completed_at=row.completed_at,
            classes_path=row.classes_path,
            valid_mask_path=row.valid_mask_path,
        )

    @property
    def snapshot(self) -> dict:
        return {
            "result_id": self.id,
            "identity_sha256": self.identity_sha256,
            "source_image_id": self.source_image_id,
            "source_model_id": self.source_model_id,
            "image_content_sha256": self.image_content_sha256,
            "weight_content_sha256": self.weight_content_sha256,
            "inference_parameters": self.inference_parameters,
            "classification_scheme_version": self.classification_scheme_version,
            "pipeline_version": self.pipeline_version,
            "grid_policy_version": self.grid_policy_version,
            "completed_at": self.completed_at.isoformat(),
        }


@dataclass(frozen=True)
class ResolvedChangeInputs:
    before: ResolvedPeriod
    after: ResolvedPeriod

    @classmethod
    def from_rows(cls, before, after):
        return cls(
            before=ResolvedPeriod.from_row(before),
            after=ResolvedPeriod.from_row(after),
        )

    @property
    def before_snapshot(self) -> dict:
        return self.before.snapshot

    @property
    def after_snapshot(self) -> dict:
        return self.after.snapshot


class ChangeInputError(ValueError):
    def __init__(self, message: str, periods: dict):
        super().__init__(message)
        self.periods = periods


def _now():
    return datetime.now(timezone.utc)


def _serialize(row) -> dict:
    data = {
        "request_id": row.request_id,
        "status": row.status,
        "started_at": row.started_at.isoformat(),
        "heartbeat_at": row.heartbeat_at.isoformat(),
        "completed_at": row.completed_at.isoformat() if row.completed_at is not None else None,
    }
    if row.status == crud_results.SUCCEEDED:
        data.update(
            {
                "classes": ordered_class_definitions(),
                "before": row.before_snapshot,
                "after": row.after_snapshot,
                "matrix_m2": row.matrix_m2,
                "common_valid_area_m2": row.common_valid_area_m2,
                "grid": {
                    "crs": row.crs,
                    "transform": row.transform,
                    "width": row.raster_width,
                    "height": row.raster_height,
                    "bounds": row.bounds,
                },
                "before_window": row.before_window,
                "after_window": row.after_window,
                "calculated_at": row.calculated_at.isoformat(),
            }
        )
    return data


def _failure(row) -> JSONResponse:
    details = dict(row.error_data)
    details.update({"request_id": row.request_id, "error_code": row.error_code})
    return api_response(row.error_http_status, row.error_message, details)


def _same_inputs(row, payload: CreateChangeResultRequest) -> bool:
    return (
        row.before_image_id == payload.before_image_id
        and row.after_image_id == payload.after_image_id
        and row.source_model_id == payload.model_id
        and row.before_result_id == payload.before_result_id
        and row.before_identity_sha256 == payload.before_identity_sha256
        and row.after_result_id == payload.after_result_id
        and row.after_identity_sha256 == payload.after_identity_sha256
    )


def _existing(row) -> JSONResponse:
    if row.status == crud_results.PROCESSING:
        return api_response(202, "变化分析计算中", _serialize(row))
    if row.status == crud_results.FAILED:
        return _failure(row)
    return api_response(200, "变化分析完成", _serialize(row))


async def _heartbeat(request_id: str, user_id: int, owner: str) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        async with AsyncSessionLocal() as heartbeat_db:
            active = await crud_results.heartbeat(
                request_id,
                user_id,
                owner,
                _now(),
                heartbeat_db,
            )
            if not active:
                await heartbeat_db.rollback()
                return
            await heartbeat_db.commit()


async def resolve_change_inputs(db, user_id: int, payload: CreateChangeResultRequest):
    requested = {
        "before": (payload.before_image_id, payload.before_result_id, payload.before_identity_sha256),
        "after": (payload.after_image_id, payload.after_result_id, payload.after_identity_sha256),
    }
    rows = {}
    failures = {}
    for period, (image_id, result_id, identity_sha256) in requested.items():
        resolution = await resolve_identification_for_change(
            db,
            user_id=user_id,
            image_id=image_id,
            model_id=payload.model_id,
        )
        if resolution.status != "SUCCEEDED":
            reason = (
                resolution.reason
                if resolution.reason in {"MISSING", "VERSION_MISMATCH"}
                else "INCOMPLETE"
            )
            failures[period] = {"reason": reason, "status": resolution.status}
        elif resolution.row.id != result_id or resolution.row.identity_sha256 != identity_sha256:
            failures[period] = {"reason": "VERSION_MISMATCH", "status": "UNAVAILABLE"}
        else:
            rows[period] = ResolvedPeriod.from_row(resolution.row)
    if failures:
        raise ChangeInputError("两期识别结果尚不可用于变化分析", failures)
    return ResolvedChangeInputs(before=rows["before"], after=rows["after"])


async def _persist_failure(db, **values):
    return await crud_results.mark_failed(db, now=_now(), **values)


@router.post("")
async def create_change_result(
    payload: CreateChangeResultRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    user_id = current_user.id
    request_id = str(payload.request_id)
    owner = uuid4().hex
    claim = await crud_results.claim_request(
        db, payload=payload, user_id=user_id, owner=owner, now=_now()
    )
    if not claim.should_start:
        if not _same_inputs(claim.row, payload):
            return api_response(
                409,
                "request_id 已用于另一组变化分析输入",
                {
                    "request_id": request_id,
                    "error_code": REQUEST_ID_CONFLICT,
                },
            )
        return _existing(claim.row)
    await db.commit()

    try:
        resolved = await resolve_change_inputs(db, user_id, payload)
    except ChangeInputError as exc:
        row = await _persist_failure(
            db,
            request_id=request_id,
            user_id=user_id,
            owner=owner,
            http_status=422,
            error_code=IDENTIFICATION_RESULT_UNAVAILABLE,
            message=str(exc),
            error_data={"periods": exc.periods},
        )
        return _failure(row)
    except TimeoutError:
        row = await _persist_failure(
            db,
            request_id=request_id,
            user_id=user_id,
            owner=owner,
            http_status=409,
            error_code=IDENTIFICATION_RESULT_BUSY,
            message="识别结果正在使用，请稍后重试",
            error_data={},
        )
        return _failure(row)
    except HTTPException as exc:
        row = await _persist_failure(
            db,
            request_id=request_id,
            user_id=user_id,
            owner=owner,
            http_status=exc.status_code,
            error_code=IDENTIFICATION_RESULT_UNAVAILABLE,
            message=str(exc.detail),
            error_data={},
        )
        return _failure(row)

    await db.rollback()
    heartbeat_task = asyncio.create_task(_heartbeat(request_id, user_id, owner))
    try:
        result = await asyncio.to_thread(
            compute_transition_matrix_m2,
            resolved.before.classes_path,
            resolved.before.valid_mask_path,
            resolved.after.classes_path,
            resolved.after.valid_mask_path,
        )
        row = await crud_results.mark_succeeded(
            db,
            request_id=request_id,
            user_id=user_id,
            owner=owner,
            resolved=resolved,
            result=result,
            now=_now(),
        )
        if row is None:
            return _existing(await crud_results.get_by_request(db, request_id, user_id))
        return api_response(200, "变化分析完成", _serialize(row))
    except TransitionMatrixError as exc:
        http_status = 409 if exc.error_code == IDENTIFICATION_RESULT_BUSY else 422
        row = await _persist_failure(
            db,
            request_id=request_id,
            user_id=user_id,
            owner=owner,
            http_status=http_status,
            error_code=exc.error_code,
            message=str(exc),
            error_data={},
        )
        return _failure(row)
    except Exception:
        await _persist_failure(
            db,
            request_id=request_id,
            user_id=user_id,
            owner=owner,
            http_status=500,
            error_code=CHANGE_RESULT_FAILED,
            message="变化分析计算失败",
            error_data={},
        )
        raise
    finally:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task


@router.get("/{request_id}")
async def get_change_result(
    request_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    await crud_results.expire_stale(db, now=_now())
    await db.commit()
    row = await crud_results.get_by_request(db, str(request_id), current_user.id)
    if row is None:
        raise HTTPException(status_code=404, detail="变化分析结果不存在")
    return _existing(row)
