from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.classification_results import ClassificationResult
from services.identification_results import FAILED, SUCCEEDED


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
        )
    )
    return result.rowcount == 1

