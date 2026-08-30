from typing import Optional

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
    model_type: Optional[str] = None,
    framework: Optional[str] = None,
) -> tuple[list[MLModel], int]:
    """分页查询当前用户上传的模型记录"""
    stmt = select(MLModel).where(MLModel.user_id == user_id)

    if keyword:
        stmt = stmt.where(MLModel.model_name.ilike(f"%{keyword}%"))
    if model_type:
        stmt = stmt.where(MLModel.model_type == model_type)
    if framework:
        stmt = stmt.where(MLModel.framework == framework)

    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    stmt = (
        stmt.order_by(MLModel.upload_time.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    items = result.scalars().all()
    return list(items), total or 0


async def get_ml_model_by_id(
    db: AsyncSession,
    model_id: int,
    user_id: int,
) -> Optional[MLModel]:
    """根据ID获取当前用户的模型记录"""
    result = await db.execute(
        select(MLModel).where(MLModel.id == model_id, MLModel.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def update_ml_model(
    db: AsyncSession,
    model_id: int,
    user_id: int,
    **kwargs: str,
) -> Optional[MLModel]:
    """局部更新当前用户的模型元数据"""
    record = await get_ml_model_by_id(db, model_id, user_id)
    if record is None:
        return None

    for key, value in kwargs.items():
        setattr(record, key, value)

    await db.flush()
    await db.refresh(record)
    return record


async def delete_ml_model(
    db: AsyncSession,
    model_id: int,
    user_id: int,
) -> Optional[tuple[str, str]]:
    """删除当前用户的模型记录，返回待清理的文件路径"""
    record = await get_ml_model_by_id(db, model_id, user_id)
    if record is None:
        return None

    w_path = record.weight_file_path
    m_path = record.model_file_path
    await db.delete(record)
    return w_path, m_path
