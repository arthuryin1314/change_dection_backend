import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

from config.db_config import get_db
from crud.ml_models import (
    create_ml_model,
    delete_ml_model,
    get_ml_model_by_id,
    get_ml_models,
    update_ml_model,
)
from schemas.ml_models import (
    MLModelListItem,
    MLModelListResponse,
    MLModelResponse,
)
from utils.file_storage import save_upload_file
from utils.get_user_by_token import get_current_user
from utils.response import error_response, success_response


router = APIRouter(prefix="/api/models", tags=["models"])

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHT_EXTENSIONS = {".pth", ".pt", ".h5", ".onnx", ".pdparams"}
MODEL_FILE_EXTENSIONS = {".py", ".zip"}
MAX_WEIGHT_SIZE = 1 * 1024 * 1024 * 1024
MAX_MODEL_FILE_SIZE = 100 * 1024 * 1024


def _get_suffix(upload_file: UploadFile) -> str:
    return Path(upload_file.filename or "").suffix.lower()


def _safe_unlink(file_path: Optional[Path]) -> None:
    if file_path is None:
        return
    try:
        file_path.unlink(missing_ok=True)
    except Exception:
        pass


@router.post("/upload", summary="上传模型")
async def upload_model(
    model_name: str = Form(...),
    model_type: str = Form(...),
    framework: str = Form(...),
    weight: UploadFile = File(...),
    model_file: UploadFile = File(...),
    description: str = Form(...),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if _get_suffix(weight) not in WEIGHT_EXTENSIONS or _get_suffix(model_file) not in MODEL_FILE_EXTENSIONS:
        return error_response(400, "不支持的文件类型")

    weight_path: Optional[Path] = None
    model_file_path: Optional[Path] = None

    try:
        weight_path = await save_upload_file(
            weight,
            BASE_DIR / "uploads" / "weights" / str(current_user.id),
        )
        model_file_path = await save_upload_file(
            model_file,
            BASE_DIR / "uploads" / "model_files" / str(current_user.id),
        )

        db_record = await create_ml_model(
            db=db,
            user_id=current_user.id,
            model_name=model_name,
            model_type=model_type,
            framework=framework,
            weight_file_path=str(weight_path),
            model_file_path=str(model_file_path),
            description=description,
        )
        await db.commit()

        data = MLModelResponse.model_validate(db_record).model_dump(mode="json")
        return success_response(message="上传成功", data=data)
    except Exception as e:
        logger.exception("上传模型失败: %s", e)
        await db.rollback()
        _safe_unlink(weight_path)
        _safe_unlink(model_file_path)
        return error_response(500, "上传失败")


@router.get("/list", summary="获取模型列表")
async def list_models(
    page: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    keyword: str = Query(default=""),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        items, total = await get_ml_models(db, current_user.id, page, pageSize, keyword)
        response = MLModelListResponse(
            items=[
                MLModelListItem(
                    id=item.id,
                    model_name=item.model_name,
                    model_type=item.model_type,
                    framework=item.framework,
                    weight=Path(item.weight_file_path).name,
                    model_file=Path(item.model_file_path).name,
                    description=item.description,
                    update_date=item.updated_time.strftime("%Y-%m-%d") if item.updated_time else None,
                )
                for item in items
            ],
            total=total,
            page=page,
            pageSize=pageSize,
        )
        return success_response(data=response.model_dump())
    except Exception as e:
        logger.exception("获取模型列表失败: %s", e)
        return error_response(500, "获取模型列表失败")


@router.put("/{model_id}", summary="更新模型")
async def update_model(
    model_id: int,
    model_name: Optional[str] = Form(None),
    model_type: Optional[str] = Form(None),
    framework: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    weight: Optional[UploadFile] = File(None),
    model_file: Optional[UploadFile] = File(None),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    record = await get_ml_model_by_id(db, model_id, current_user.id)
    if record is None:
        return error_response(404, "模型不存在")

    old_w_path = record.weight_file_path if weight else None
    old_m_path = record.model_file_path if model_file else None

    update_kwargs = {}
    if model_name is not None:
        update_kwargs["model_name"] = model_name
    if model_type is not None:
        update_kwargs["model_type"] = model_type
    if framework is not None:
        update_kwargs["framework"] = framework
    if description is not None:
        update_kwargs["description"] = description

    new_w_path: Optional[Path] = None
    new_m_path: Optional[Path] = None

    try:
        if weight:
            if _get_suffix(weight) not in WEIGHT_EXTENSIONS:
                return error_response(400, "不支持的权重文件类型")
            if weight.size and weight.size > MAX_WEIGHT_SIZE:
                return error_response(400, "权重文件超过大小限制（最大 1 GB）")

        if model_file:
            if _get_suffix(model_file) not in MODEL_FILE_EXTENSIONS:
                return error_response(400, "不支持的模型文件类型")
            if model_file.size and model_file.size > MAX_MODEL_FILE_SIZE:
                return error_response(400, "模型文件超过大小限制（最大 100 MB）")

        if weight:
            new_w_path = await save_upload_file(
                weight,
                BASE_DIR / "uploads" / "weights" / str(current_user.id),
            )
            update_kwargs["weight_file_path"] = str(new_w_path)

        if model_file:
            new_m_path = await save_upload_file(
                model_file,
                BASE_DIR / "uploads" / "model_files" / str(current_user.id),
            )
            update_kwargs["model_file_path"] = str(new_m_path)

        if not update_kwargs:
            return success_response(message="无改动")

        await update_ml_model(db, model_id, current_user.id, **update_kwargs)

        await db.commit()

        if old_w_path:
            _safe_unlink(Path(old_w_path))
        if old_m_path:
            _safe_unlink(Path(old_m_path))

        return success_response(message="更新成功")
    except Exception as e:
        logger.exception("更新模型失败: %s", e)
        await db.rollback()
        _safe_unlink(new_w_path)
        _safe_unlink(new_m_path)
        return error_response(500, "更新失败")


@router.delete("/{model_id}", summary="删除模型")
async def delete_model(
    model_id: int,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        paths = await delete_ml_model(db, model_id, current_user.id)
        if paths is None:
            return error_response(404, "模型不存在")

        w_path, m_path = paths
        await db.commit()

        if w_path:
            _safe_unlink(Path(w_path))
        if m_path:
            _safe_unlink(Path(m_path))

        return success_response(message="删除成功")
    except Exception as e:
        logger.exception("删除模型失败: %s", e)
        await db.rollback()
        return error_response(500, "删除失败")
