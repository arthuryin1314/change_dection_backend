from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select,func
from sqlalchemy.orm import selectinload
from models.images import Image, BoundaryFile
from datetime import datetime
from typing import List, Optional, Any, Dict


async def create_image(
    db: AsyncSession,
    user_id: int,
    image_name: str,
    resolution: float,
    capture_date,
    satellite: str,
    image_type: str,
    region_code: str,
    img_path: str,
    bbox: Optional[List[float]] = None,
    layer_name: Optional[str] = None,
    wms_url: Optional[str] = None,
    content_sha256: Optional[str] = None,
    content_sha256_size: Optional[int] = None,
    content_sha256_mtime_ns: Optional[int] = None,
) -> Image:
    """创建新的影像记录"""
    db_image = Image(
        user_id=user_id,
        image_name=image_name,
        resolution=resolution,
        capture_date=capture_date,
        satellite=satellite,
        image_type=image_type,
        region_code=region_code,
        img_path=img_path,
        bbox=bbox,
        layer_name=layer_name,
        wms_url=wms_url,
        content_sha256=content_sha256,
        content_sha256_size=content_sha256_size,
        content_sha256_mtime_ns=content_sha256_mtime_ns,
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


async def query_images(
    db: AsyncSession,
    user_id: int,
    page: int,
    page_size: int,
    keyword: Optional[str] = None,
) -> tuple[List[Image], int]:
    filters = [Image.user_id == user_id]
    keyword = (keyword or "").strip()
    if keyword:
        escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        filters.append(Image.image_name.ilike(f"%{escaped}%", escape="\\"))

    count_result = await db.execute(select(func.count()).select_from(Image).where(*filters))
    total = count_result.scalar() or 0
    result = await db.execute(
        select(Image)
        .options(selectinload(Image.boundary_files))
        .where(*filters)
        .order_by(Image.upload_time.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(result.scalars().all()), total


async def get_image_by_id(db: AsyncSession, image_id: int, user_id: int) -> Optional[Image]:
    """根据ID获取当前用户的影像记录"""
    result = await db.execute(
        select(Image)
        .options(selectinload(Image.boundary_files))
        .where(Image.id == image_id, Image.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def count_images_by_ids(db: AsyncSession, user_id: int, image_ids: List[int]) -> int:
    if not image_ids:
        return 0
    result = await db.execute(
        select(func.count())
        .select_from(Image)
        .where(Image.user_id == user_id, Image.id.in_(image_ids))
    )
    return result.scalar_one()


async def update_image_fields(db: AsyncSession, image: Image, updates: Dict[str, Any]) -> Image:
    """局部更新影像主表字段"""
    for key, value in updates.items():
        setattr(image, key, value)
    await db.flush()
    await db.refresh(image)
    return image


async def upsert_boundary_files(
    db: AsyncSession,
    image: Image,
    file_prefix: Optional[str] = None,
    shp_path: Optional[str] = None,
    dbf_path: Optional[str] = None,
    prj_path: Optional[str] = None,
) -> BoundaryFile:
    """更新或创建边界文件记录，仅覆盖传入的路径字段"""
    boundary = image.boundary_files[0] if image.boundary_files else None
    if not boundary:
        boundary = BoundaryFile(image_id=image.id)
        db.add(boundary)

    if file_prefix is not None:
        boundary.file_prefix = file_prefix
    if shp_path is not None:
        boundary.shp_path = shp_path
    if dbf_path is not None:
        boundary.dbf_path = dbf_path
    if prj_path is not None:
        boundary.prj_path = prj_path

    await db.flush()
    await db.refresh(boundary)
    return boundary


async def delete_image_with_files(db: AsyncSession, image_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    """删除当前用户影像及其边界文件记录，返回待清理的文件路径"""
    image = await get_image_by_id(db, image_id, user_id)
    if not image:
        return None

    boundary_paths: List[str] = []
    for bf in image.boundary_files:
        if bf.shp_path:
            boundary_paths.append(bf.shp_path)
        if bf.dbf_path:
            boundary_paths.append(bf.dbf_path)
        if bf.prj_path:
            boundary_paths.append(bf.prj_path)

    for bf in image.boundary_files:
        await db.delete(bf)

    img_path = image.img_path
    await db.delete(image)
    await db.flush()

    return {
        "image_id": image_id,
        "img_path": img_path,
        "boundary_paths": boundary_paths,
    }


async def delete_images_by_user_with_files(db: AsyncSession, user_id: int) -> Dict[str, Any]:
    """删除当前用户全部影像及边界文件记录，返回待清理文件路径。"""
    result = await db.execute(
        select(Image)
        .options(selectinload(Image.boundary_files))
        .where(Image.user_id == user_id)
    )
    images = list(result.scalars().all())

    image_paths: List[str] = []
    boundary_paths: List[str] = []
    layer_names: List[str] = []

    for image in images:
        if image.img_path:
            image_paths.append(image.img_path)
        if image.layer_name:
            layer_names.append(image.layer_name)

        for bf in image.boundary_files:
            if bf.shp_path:
                boundary_paths.append(bf.shp_path)
            if bf.dbf_path:
                boundary_paths.append(bf.dbf_path)
            if bf.prj_path:
                boundary_paths.append(bf.prj_path)
            await db.delete(bf)

        await db.delete(image)

    await db.flush()
    return {
        "deleted_count": len(images),
        "image_ids": [image.id for image in images],
        "img_paths": image_paths,
        "boundary_paths": boundary_paths,
        "layer_names": layer_names,
    }
