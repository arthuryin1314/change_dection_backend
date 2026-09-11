from typing import TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession


Row = TypeVar("Row")


async def list_for_user(
    db: AsyncSession,
    model: type[Row],
    user_id: int,
    *,
    offset: int,
    limit: int,
) -> tuple[list[Row], int]:
    filters = (model.user_id == user_id, model.status == "SUCCEEDED")
    total = await db.scalar(select(func.count()).select_from(model).where(*filters))
    result = await db.execute(
        select(model)
        .where(*filters)
        .order_by(model.completed_at.desc(), model.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars()), total


async def get_for_user(
    db: AsyncSession,
    model: type[Row],
    result_id: str | int,
    user_id: int,
) -> Row | None:
    result = await db.execute(
        select(model).where(
            model.id == result_id,
            model.user_id == user_id,
            model.status == "SUCCEEDED",
        )
    )
    return result.scalar_one_or_none()
