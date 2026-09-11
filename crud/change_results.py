import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import exists, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from models.change_results import ChangeRequest, ChangeResult
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


@dataclass(frozen=True)
class SubmissionClaim:
    row: ChangeResult
    should_start: bool
    request_conflict: bool = False


async def get_by_request(db: AsyncSession, request_id: str, user_id: int):
    result = await db.execute(
        select(ChangeResult)
        .join(ChangeRequest, ChangeRequest.change_result_id == ChangeResult.id)
        .where(ChangeRequest.request_id == request_id, ChangeRequest.user_id == user_id)
    )
    row = result.scalar_one_or_none()
    if row is not None:
        return row
    legacy = await db.execute(
        select(ChangeResult).where(
            ChangeResult.request_id == request_id,
            ChangeResult.user_id == user_id,
        )
    )
    return legacy.scalar_one_or_none()


async def get_by_id(db: AsyncSession, result_id: int):
    result = await db.execute(select(ChangeResult).where(ChangeResult.id == result_id))
    return result.scalar_one_or_none()


async def get_request_binding(db: AsyncSession, request_id: str, user_id: int):
    result = await db.execute(
        select(ChangeRequest).where(
            ChangeRequest.request_id == request_id,
            ChangeRequest.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def claim_submission(
    db: AsyncSession,
    *,
    request_id: str,
    request_fingerprint: str,
    preparation_key: str,
    submitted_inputs: dict,
    user_id: int,
    owner: str,
    now: datetime,
    retry_of_request_id: str | None = None,
    before_result_id: str | None = None,
    before_identity_sha256: str | None = None,
    after_result_id: str | None = None,
    after_identity_sha256: str | None = None,
) -> SubmissionClaim:
    binding = await get_request_binding(db, request_id, user_id)
    if binding is not None:
        row = await get_by_id(db, binding.change_result_id)
        return SubmissionClaim(
            row=row,
            should_start=False,
            request_conflict=binding.request_fingerprint != request_fingerprint,
        )

    statement = (
        insert(ChangeResult)
        .values(
            request_id=request_id,
            user_id=user_id,
            before_image_id=submitted_inputs["before_image_id"],
            after_image_id=submitted_inputs["after_image_id"],
            source_model_id=submitted_inputs["model_id"],
            before_result_id=before_result_id,
            after_result_id=after_result_id,
            before_identity_sha256=before_identity_sha256,
            after_identity_sha256=after_identity_sha256,
            calculation_version=CALCULATION_VERSION,
            grid_policy_version=GRID_POLICY_VERSION,
            preparation_key=preparation_key,
            orchestration_identity_sha256=None,
            phase="PREPARING",
            submitted_inputs=submitted_inputs,
            status=PROCESSING,
            lease_owner=owner,
            started_at=now,
            heartbeat_at=now,
        )
        .on_conflict_do_nothing(
            index_elements=["user_id", "preparation_key"],
            index_where=text("status = 'PROCESSING'"),
        )
        .returning(ChangeResult)
    )
    created = (await db.execute(statement)).scalar_one_or_none()
    if created is None:
        active = await db.execute(
            select(ChangeResult).where(
                ChangeResult.user_id == user_id,
                ChangeResult.preparation_key == preparation_key,
                ChangeResult.status == PROCESSING,
            )
        )
        row = active.scalar_one()
        should_start = False
    else:
        row = created
        should_start = True

    binding_statement = (
        insert(ChangeRequest)
        .values(
            user_id=user_id,
            request_id=request_id,
            request_fingerprint=request_fingerprint,
            change_result_id=row.id,
            retry_of_request_id=retry_of_request_id,
        )
        .on_conflict_do_nothing(index_elements=["user_id", "request_id"])
        .returning(ChangeRequest.id)
    )
    inserted_binding = (await db.execute(binding_statement)).scalar_one_or_none()
    if inserted_binding is None:
        binding = await get_request_binding(db, request_id, user_id)
        bound_row = await get_by_id(db, binding.change_result_id)
        return SubmissionClaim(
            row=bound_row,
            should_start=False,
            request_conflict=binding.request_fingerprint != request_fingerprint,
        )
    return SubmissionClaim(row=row, should_start=should_start)


async def set_frozen_identity(
    db: AsyncSession,
    *,
    result_id: int,
    user_id: int,
    owner: str,
    orchestration_identity_sha256: str,
    before_identity_sha256: str,
    after_identity_sha256: str,
    frozen_inputs: dict,
    now: datetime,
):
    reusable = await db.execute(
        select(ChangeResult)
        .where(
            ChangeResult.user_id == user_id,
            ChangeResult.orchestration_identity_sha256 == orchestration_identity_sha256,
            ChangeResult.status == SUCCEEDED,
        )
        .order_by(ChangeResult.completed_at.desc())
        .limit(1)
    )
    winner = reusable.scalar_one_or_none()
    if winner is None:
        try:
            async with db.begin_nested():
                changed = await db.execute(
                    _owned_processing_request_by_id(result_id, user_id, owner).values(
                        orchestration_identity_sha256=orchestration_identity_sha256,
                        before_identity_sha256=before_identity_sha256,
                        after_identity_sha256=after_identity_sha256,
                        frozen_inputs=frozen_inputs,
                        phase="WAITING_BEFORE",
                        heartbeat_at=now,
                    )
                )
                if changed.rowcount != 1:
                    return None
        except IntegrityError:
            competing = await db.execute(
                select(ChangeResult).where(
                    ChangeResult.user_id == user_id,
                    ChangeResult.orchestration_identity_sha256
                    == orchestration_identity_sha256,
                    ChangeResult.status == PROCESSING,
                )
            )
            winner = competing.scalar_one()
        else:
            await db.commit()
            return await get_by_id(db, result_id)

    await db.execute(
        update(ChangeRequest)
        .where(ChangeRequest.change_result_id == result_id)
        .values(change_result_id=winner.id)
    )
    await db.execute(
        _owned_processing_request_by_id(result_id, user_id, owner).values(
            status=FAILED,
            phase="MERGED",
            completed_at=now,
            heartbeat_at=now,
            error_http_status=409,
            error_code="CHANGE_RESULT_MERGED",
            error_message="变化分析请求已合并",
            error_data={"merged_into": winner.id},
        )
    )
    await db.commit()
    return winner


async def set_phase(
    db: AsyncSession,
    *,
    result_id: int,
    user_id: int,
    owner: str,
    phase: str,
    now: datetime,
) -> bool:
    changed = await db.execute(
        _owned_processing_request_by_id(result_id, user_id, owner).values(
            phase=phase,
            heartbeat_at=now,
        )
    )
    await db.commit()
    return changed.rowcount == 1


async def mark_period_ready(
    db: AsyncSession,
    *,
    result_id: int,
    user_id: int,
    owner: str,
    period: str,
    result,
    now: datetime,
) -> bool:
    values = {
        f"{period}_result_id": result.id,
        f"{period}_identity_sha256": result.identity_sha256,
        "heartbeat_at": now,
    }
    changed = await db.execute(
        _owned_processing_request_by_id(result_id, user_id, owner).values(**values)
    )
    await db.commit()
    return changed.rowcount == 1


async def expire_stale(db: AsyncSession, *, now: datetime) -> int:
    result = await db.execute(
        update(ChangeResult)
        .where(
            ChangeResult.status == PROCESSING,
            ChangeResult.heartbeat_at <= now - STALE_AFTER,
        )
        .values(
            status=FAILED,
            phase="FAILED",
            completed_at=now,
            error_http_status=409,
            error_code=CHANGE_RESULT_INTERRUPTED,
            error_message="变化分析执行已中断",
            error_data={
                "error_code": CHANGE_RESULT_INTERRUPTED,
                "retryable": True,
            },
        )
    )
    return result.rowcount


async def expire_request(
    db: AsyncSession,
    *,
    request_id: str,
    user_id: int,
    now: datetime,
) -> int:
    row = await get_by_request(db, request_id, user_id)
    if (
        row is None
        or row.status != PROCESSING
        or row.heartbeat_at > now - STALE_AFTER
    ):
        return 0
    result = await db.execute(
        update(ChangeResult)
        .where(ChangeResult.id == row.id, ChangeResult.status == PROCESSING)
        .values(
            status=FAILED,
            phase="FAILED",
            completed_at=now,
            error_http_status=409,
            error_code=CHANGE_RESULT_INTERRUPTED,
            error_message="变化分析执行已中断",
            error_data={
                "error_code": CHANGE_RESULT_INTERRUPTED,
                "retryable": True,
            },
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


def _owned_processing_request_by_id(result_id: int, user_id: int, owner: str):
    return update(ChangeResult).where(
        ChangeResult.id == result_id,
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
        "phase": "SUCCEEDED",
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
            phase="FAILED",
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
