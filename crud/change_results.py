import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import exists, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from models.change_results import ChangeResult
from models.classification_results import ClassificationResult
from utils.change_result_errors import CHANGE_RESULT_INTERRUPTED
from utils.transition_matrix import CALCULATION_VERSION, GRID_POLICY_VERSION


PROCESSING = "PROCESSING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
STALE_AFTER = timedelta(seconds=120)


@dataclass(frozen=True)
class RequestClaim:
    row: ChangeResult
    should_start: bool


async def get_by_request(db: AsyncSession, request_id: str, user_id: int):
    result = await db.execute(
        select(ChangeResult).where(
            ChangeResult.request_id == request_id,
            ChangeResult.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def expire_stale(db: AsyncSession, *, now: datetime) -> int:
    result = await db.execute(
        update(ChangeResult)
        .where(
            ChangeResult.status == PROCESSING,
            ChangeResult.heartbeat_at <= now - STALE_AFTER,
        )
        .values(
            status=FAILED,
            completed_at=now,
            error_http_status=409,
            error_code=CHANGE_RESULT_INTERRUPTED,
            error_message="变化分析执行已中断",
            error_data={"error_code": CHANGE_RESULT_INTERRUPTED},
        )
    )
    return result.rowcount


async def claim_request(db: AsyncSession, *, payload, user_id: int, owner: str, now: datetime):
    row = ChangeResult(
        request_id=str(payload.request_id),
        user_id=user_id,
        before_image_id=payload.before_image_id,
        after_image_id=payload.after_image_id,
        source_model_id=payload.model_id,
        before_result_id=payload.before_result_id,
        after_result_id=payload.after_result_id,
        before_identity_sha256=payload.before_identity_sha256,
        after_identity_sha256=payload.after_identity_sha256,
        calculation_version=CALCULATION_VERSION,
        grid_policy_version=GRID_POLICY_VERSION,
        status=PROCESSING,
        lease_owner=owner,
        started_at=now,
        heartbeat_at=now,
    )
    db.add(row)
    try:
        await db.flush()
        return RequestClaim(row=row, should_start=True)
    except IntegrityError:
        await db.rollback()
        existing = await get_by_request(db, str(payload.request_id), user_id)
        if existing is None:
            raise
        if existing.status == PROCESSING and existing.heartbeat_at <= now - STALE_AFTER:
            await expire_stale(db, now=now)
            await db.commit()
            existing = await get_by_request(db, str(payload.request_id), user_id)
        return RequestClaim(row=existing, should_start=False)


async def heartbeat(request_id: str, user_id: int, owner: str, now: datetime, db):
    result = await db.execute(
        update(ChangeResult)
        .where(
            ChangeResult.request_id == request_id,
            ChangeResult.user_id == user_id,
            ChangeResult.status == PROCESSING,
            ChangeResult.lease_owner == owner,
        )
        .values(heartbeat_at=now)
    )
    return result.rowcount == 1


def _owned_processing_request(request_id: str, user_id: int, owner: str):
    return update(ChangeResult).where(
        ChangeResult.request_id == request_id,
        ChangeResult.user_id == user_id,
        ChangeResult.status == PROCESSING,
        ChangeResult.lease_owner == owner,
    )


async def mark_succeeded(db, *, request_id, user_id, owner, resolved, result, now):
    identity_payload = {
        "before_identity_sha256": resolved.before.identity_sha256,
        "after_identity_sha256": resolved.after.identity_sha256,
        "calculation_version": CALCULATION_VERSION,
        "grid_policy_version": GRID_POLICY_VERSION,
        "grid": {
            "crs": str(result.grid.crs),
            "transform": list(result.grid.transform)[:6],
            "width": result.grid.width,
            "height": result.grid.height,
            "bounds": result.bounds,
        },
        "analysis": result.analysis,
    }
    analysis_identity_sha256 = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    values = {
        "status": SUCCEEDED,
        "completed_at": now,
        "calculated_at": now,
        "heartbeat_at": now,
        "before_snapshot": resolved.before_snapshot,
        "after_snapshot": resolved.after_snapshot,
        "matrix_m2": result.matrix_m2,
        "common_valid_area_m2": result.common_valid_area_m2,
        "crs": str(result.grid.crs),
        "transform": list(result.grid.transform)[:6],
        "raster_width": result.grid.width,
        "raster_height": result.grid.height,
        "bounds": result.bounds,
        "before_window": result.before_window,
        "after_window": result.after_window,
        "calculation_version": CALCULATION_VERSION,
        "grid_policy_version": GRID_POLICY_VERSION,
        "analysis_identity_sha256": analysis_identity_sha256,
        "analysis_metadata": result.analysis,
    }
    def source_is_current(source):
        return exists().where(
            ClassificationResult.id == source.id,
            ClassificationResult.user_id == user_id,
            ClassificationResult.status == SUCCEEDED,
            ClassificationResult.identity_sha256 == source.identity_sha256,
            ClassificationResult.completed_at == source.completed_at,
        )

    changed = await db.execute(
        _owned_processing_request(request_id, user_id, owner)
        .where(source_is_current(resolved.before), source_is_current(resolved.after))
        .values(**values)
    )
    if changed.rowcount != 1:
        await db.rollback()
        return None
    await db.commit()
    return await get_by_request(db, request_id, user_id)


async def mark_failed(
    db,
    *,
    request_id,
    user_id,
    owner,
    http_status,
    error_code,
    message,
    error_data,
    now,
):
    changed = await db.execute(
        _owned_processing_request(request_id, user_id, owner).values(
            status=FAILED,
            heartbeat_at=now,
            completed_at=now,
            error_http_status=http_status,
            error_code=error_code,
            error_message=message[:4000],
            error_data=error_data,
        )
    )
    if changed.rowcount != 1:
        await db.rollback()
        return None
    await db.commit()
    return await get_by_request(db, request_id, user_id)
