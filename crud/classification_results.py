from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.classification_results import ClassificationResult
from services.identification_results import FAILED, SUCCEEDED
from utils.classification_area import validate_class_area_m2


async def get_result_by_id(
    db: AsyncSession,
    result_id: str,
    user_id: int,
) -> ClassificationResult | None:
    result = await db.execute(
        select(ClassificationResult).where(
            ClassificationResult.id == result_id,
            ClassificationResult.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def invalidate_succeeded_result(
    db: AsyncSession,
    result_id: str,
    user_id: int,
) -> bool:
    result = await db.execute(
        update(ClassificationResult)
        .where(
            ClassificationResult.id == result_id,
            ClassificationResult.user_id == user_id,
            ClassificationResult.status == SUCCEEDED,
        )
        .values(
            status=FAILED,
            failure_detail="识别结果文件缺失或损坏",
            completed_at=datetime.now(timezone.utc),
            class_area_m2=None,
            area_status="NOT_COMPUTED",
            area_completed_at=None,
            area_failure_detail=None,
        )
    )
    return result.rowcount == 1


def _current_generation(result_id, user_id, lease_owner, completed_at):
    return update(ClassificationResult).where(
        ClassificationResult.id == result_id,
        ClassificationResult.user_id == user_id,
        ClassificationResult.status == SUCCEEDED,
        ClassificationResult.lease_owner == lease_owner,
        ClassificationResult.completed_at == completed_at,
    )


async def mark_area_succeeded(
    db: AsyncSession,
    *,
    result_id: str,
    user_id: int,
    lease_owner: str,
    completed_at: datetime,
    class_area_m2,
    area_completed_at: datetime,
) -> bool:
    areas = validate_class_area_m2(class_area_m2)
    result = await db.execute(
        _current_generation(result_id, user_id, lease_owner, completed_at).values(
            class_area_m2=areas,
            area_status=SUCCEEDED,
            area_completed_at=area_completed_at,
            area_failure_detail=None,
        )
    )
    return result.rowcount == 1


async def mark_area_failed(
    db: AsyncSession,
    *,
    result_id: str,
    user_id: int,
    lease_owner: str,
    completed_at: datetime,
    detail: str,
    area_completed_at: datetime,
) -> bool:
    result = await db.execute(
        _current_generation(result_id, user_id, lease_owner, completed_at).values(
            class_area_m2=None,
            area_status=FAILED,
            area_completed_at=area_completed_at,
            area_failure_detail=detail[:4000],
        )
    )
    return result.rowcount == 1

