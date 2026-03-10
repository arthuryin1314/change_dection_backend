from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.images import Image, BoundaryFile
from datetime import datetime
from typing import List, Optional


async def create_image(
    db: AsyncSession,
    image_name: str,
    resolution: float,
    capture_date,
    satellite: str,
    image_type: str,
    region_code: str,
    img_path: str
) -> Image:
    """创建新的影像记录"""
    db_image = Image(
        image_name=image_name,
        resolution=resolution,
        capture_date=capture_date,
        satellite=satellite,
        image_type=image_type,
        region_code=region_code,
        img_path=img_path,
        upload_time=datetime.now()
    )
    db.add(db_image)
    await db.flush()
    await db.refresh(db_image)
    return db_image


async def create_boundary_files(
    db: AsyncSession,
    image_id: int,
    file_prefix: str,
    shp_path: Optional[str] = None,
    dbf_path: Optional[str] = None,
    prj_path: Optional[str] = None
) -> BoundaryFile:
    """创建边界文件记录"""
    db_boundary = BoundaryFile(
        image_id=image_id,
        file_prefix=file_prefix,
        shp_path=shp_path,
        dbf_path=dbf_path,
        prj_path=prj_path
    )
    db.add(db_boundary)
    await db.flush()
    await db.refresh(db_boundary)
    return db_boundary


async def get_all_images(db: AsyncSession) -> List[Image]:
    """获取所有影像记录"""
    result = await db.execute(
        select(Image).order_by(Image.upload_time.desc())
    )
    return result.scalars().all()


async def get_image_by_id(db: AsyncSession, image_id: int) -> Optional[Image]:
    """根据ID获取影像记录"""
    result = await db.execute(
        select(Image).where(Image.id == image_id)
    )
    return result.scalar_one_or_none()

