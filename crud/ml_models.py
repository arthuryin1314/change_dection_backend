from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.ml_models import MLModel


async def create_ml_model(
    db: AsyncSession,
    user_id: int,
    model_name: str,
    model_type: str,
    framework: str,
    weight_file_path: str,
    model_file_path: str,
    description: str,
) -> MLModel:
    """创建新的模型上传记录"""
    db_model = MLModel(
        user_id=user_id,
        model_name=model_name,
        model_type=model_type,
        framework=framework,
        weight_file_path=weight_file_path,
        model_file_path=model_file_path,
        description=description,
    )
    db.add(db_model)
    await db.flush()
    await db.refresh(db_model)
    return db_model


async def get_ml_models(
    db: AsyncSession,
    user_id: int,
    page: int,
    page_size: int,
    keyword: str = "",
) -> tuple[list[MLModel], int]:
    """分页查询当前用户上传的模型记录"""
    stmt = select(MLModel).where(MLModel.user_id == user_id)

    if keyword:
        stmt = stmt.where(MLModel.model_name.ilike(f"%{keyword}%"))

    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    stmt = (
        stmt.order_by(MLModel.upload_time.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    items = result.scalars().all()
    return list(items), total or 0
