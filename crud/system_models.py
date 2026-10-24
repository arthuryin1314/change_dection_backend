from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.system_models import SystemModel, SystemModelVersion


async def list_system_models(
    db: AsyncSession, *, page: int, page_size: int,
    keyword: str, model_type: str | None,
) -> tuple[list[tuple[SystemModel, SystemModelVersion | None]], int]:
    # Without a production version, expose the latest version's public status
    # so uploading/building/failed models remain visible in the catalog.
    latest_version = (
        select(func.max(SystemModelVersion.id))
        .where(SystemModelVersion.model_id == SystemModel.id)
        .correlate(SystemModel).scalar_subquery()
    )
    query = select(SystemModel, SystemModelVersion).outerjoin(
        SystemModelVersion,
        SystemModelVersion.id == func.coalesce(SystemModel.production_version_id, latest_version),
    )
    if keyword:
        query = query.where(SystemModel.model_name.contains(keyword, autoescape=True))
    if model_type is not None:
        query = query.where(SystemModel.model_type == model_type)
    total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar_one()
    rows = await db.execute(query.order_by(SystemModel.id.desc()).offset((page - 1) * page_size).limit(page_size))
    return [(model, version) for model, version in rows], total
