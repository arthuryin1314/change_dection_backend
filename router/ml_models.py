import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
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
from utils.model_asset_storage import ModelAssetTooLargeError, save_upload_file
from utils.get_user_by_token import get_current_user
from utils.response import error_response, success_response


router = APIRouter(prefix="/api/models", tags=["models"])

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHT_EXTENSIONS = {".pth", ".pt", ".h5", ".onnx", ".pdparams"}
MODEL_FILE_EXTENSIONS = {".py", ".zip"}
MAX_WEIGHT_SIZE = 1 * 1024 * 1024 * 1024
MAX_MODEL_FILE_SIZE = 100 * 1024 * 1024
MODEL_TYPES = {"semantic_segmentation", "change_detection", "target_extraction"}
FRAMEWORKS = {"PyTorch", "TensorFlow", "PaddlePaddle", "ONNX"}


def _get_suffix(upload_file: UploadFile) -> str:
    if upload_file.filename is None:
        return ""
    return Path(upload_file.filename).suffix.lower()


def _validate_metadata(
    model_name: Optional[str],
    model_type: Optional[str],
    framework: Optional[str],
    description: Optional[str],
    require_all: bool,
) -> Optional[str]:
    if model_name is not None:
        if not 2 <= len(model_name) <= 40:
            return "模型名称长度必须为 2-40 个字符"
    elif require_all:
        return "模型名称不能为空"

    if model_type is not None:
        if model_type not in MODEL_TYPES:
            return "模型类型不合法"
    elif require_all:
        return "模型类型不能为空"

    if framework is not None:
        if framework not in FRAMEWORKS:
            return "框架不合法"
    elif require_all:
        return "框架不能为空"

    if description is not None:
        minimum = 5 if require_all else 0
        if not minimum <= len(description) <= 200:
            return "模型描述长度不合法"
    elif require_all:
        return "模型描述不能为空"

    return None


def _validate_file(
    upload_file: UploadFile,
    extensions: set[str],
    max_size: int,
    invalid_message: str,
    size_message: str,
) -> Optional[str]:
    if _get_suffix(upload_file) not in extensions:
        return invalid_message
    if upload_file.size is not None and upload_file.size > max_size:
        return size_message
    return None


def _safe_unlink(file_path: Optional[Path]) -> None:
    if file_path is None:
        return
    try:
        file_path.unlink(missing_ok=True)
        file_path.parent.rmdir()
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
    validation_error = _validate_metadata(
        model_name,
        model_type,
        framework,
        description,
        require_all=True,
    )
    if validation_error is not None:
        return error_response(400, validation_error)

    validation_error = _validate_file(
        weight,
        WEIGHT_EXTENSIONS,
        MAX_WEIGHT_SIZE,
        "不支持的权重文件类型",
        "权重文件超过大小限制（最大 1 GB）",
    )
    if validation_error is not None:
        return error_response(400, validation_error)

    validation_error = _validate_file(
        model_file,
        MODEL_FILE_EXTENSIONS,
        MAX_MODEL_FILE_SIZE,
        "不支持的模型文件类型",
        "模型文件超过大小限制（最大 100 MB）",
    )
    if validation_error is not None:
        return error_response(400, validation_error)

    weight_path: Optional[Path] = None
    model_file_path: Optional[Path] = None

    try:
        weight_path = await save_upload_file(
            weight,
            BASE_DIR / "uploads" / "model_assets",
            current_user.id,
            MAX_WEIGHT_SIZE,
        )
        model_file_path = await save_upload_file(
            model_file,
            BASE_DIR / "uploads" / "model_assets",
            current_user.id,
            MAX_MODEL_FILE_SIZE,
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
        data = MLModelResponse.model_validate(db_record).model_dump(mode="json")
        await db.commit()
        return success_response(message="上传成功", data=data)
    except ModelAssetTooLargeError as e:
        logger.exception("上传模型文件超过大小限制: %s", e)
        await db.rollback()
        _safe_unlink(weight_path)
        _safe_unlink(model_file_path)
        if e.max_size == MAX_WEIGHT_SIZE:
            return error_response(400, "权重文件超过大小限制（最大 1 GB）")
        return error_response(400, "模型文件超过大小限制（最大 100 MB）")
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
    modelType: Optional[str] = Query(default=None),
    framework: Optional[str] = Query(default=None),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        items, total = await get_ml_models(
            db,
            current_user.id,
            page,
            pageSize,
            keyword,
            modelType,
            framework,
        )
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
    request: Request,
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

    form = await request.form()
    if model_name is None and "model_name" in form:
        model_name = ""
    if model_type is None and "model_type" in form:
        model_type = ""
    if framework is None and "framework" in form:
        framework = ""
    if description is None and "description" in form:
        description = ""

    validation_error = _validate_metadata(
        model_name,
        model_type,
        framework,
        description,
        require_all=False,
    )
    if validation_error is not None:
        return error_response(400, validation_error)

    if weight is not None:
        validation_error = _validate_file(
            weight,
            WEIGHT_EXTENSIONS,
            MAX_WEIGHT_SIZE,
            "不支持的权重文件类型",
            "权重文件超过大小限制（最大 1 GB）",
        )
        if validation_error is not None:
            return error_response(400, validation_error)

    if model_file is not None:
        validation_error = _validate_file(
            model_file,
            MODEL_FILE_EXTENSIONS,
            MAX_MODEL_FILE_SIZE,
            "不支持的模型文件类型",
            "模型文件超过大小限制（最大 100 MB）",
        )
        if validation_error is not None:
            return error_response(400, validation_error)

    old_w_path = Path(record.weight_file_path) if weight is not None else None
    old_m_path = Path(record.model_file_path) if model_file is not None else None

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
        if weight is not None:
            new_w_path = await save_upload_file(
                weight,
                BASE_DIR / "uploads" / "model_assets",
                current_user.id,
                MAX_WEIGHT_SIZE,
            )
            update_kwargs["weight_file_path"] = str(new_w_path)

        if model_file is not None:
            new_m_path = await save_upload_file(
                model_file,
                BASE_DIR / "uploads" / "model_assets",
                current_user.id,
                MAX_MODEL_FILE_SIZE,
            )
            update_kwargs["model_file_path"] = str(new_m_path)

        if not update_kwargs:
            return success_response(message="无改动")

        await update_ml_model(db, model_id, current_user.id, **update_kwargs)

        await db.commit()

        _safe_unlink(old_w_path)
        _safe_unlink(old_m_path)

        return success_response(message="更新成功")
    except ModelAssetTooLargeError as e:
        logger.exception("更新模型文件超过大小限制: %s", e)
        await db.rollback()
        _safe_unlink(new_w_path)
        _safe_unlink(new_m_path)
        if e.max_size == MAX_WEIGHT_SIZE:
            return error_response(400, "权重文件超过大小限制（最大 1 GB）")
        return error_response(400, "模型文件超过大小限制（最大 100 MB）")
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
