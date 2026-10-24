from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from config.db_config import get_db
from crud.system_models import list_system_models
from models.system_models import SystemModel
from models.users import User
from schemas.ml_models import MLModelListItem, MLModelListResponse, MLModelUpdateRequest, ModelType, ModelVersionSummary
from utils.get_user_by_token import get_current_user
from utils.model_permissions import require_model_admin
from utils.response import error_response, success_response


router = APIRouter(prefix="/api/models", tags=["models"])
Database = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.get("/list", summary="系统模型目录")
async def list_models(
    db: Database,
    current_user: CurrentUser,
    page: Annotated[int, Query(ge=1)] = 1,
    pageSize: Annotated[int, Query(ge=1, le=100)] = 10,
    keyword: Annotated[str, Query(max_length=255)] = "",
    modelType: ModelType | None = None,
):
    rows, total = await list_system_models(
        db, page=page, page_size=pageSize, keyword=keyword.strip(), model_type=modelType,
    )
    data = MLModelListResponse(
        items=[MLModelListItem(
            id=model.id, model_name=model.model_name, model_type=model.model_type,
            description=model.description, tags=model.tags,
            production_version_id=model.production_version_id,
            version=ModelVersionSummary.model_validate(version) if version is not None else None,
            # Runtime/selector integration belongs to later tickets. Technical
            # availability alone must not enable a business action.
            business_capabilities=[], updated_at=model.updated_at,
        ) for model, version in rows],
        total=total, page=page, pageSize=pageSize, can_manage=current_user.is_admin,
    )
    return success_response(data=data.model_dump(mode="json"))


@router.put("/{model_id}", summary="编辑模型目录信息", dependencies=[Depends(require_model_admin)])
async def update_model(model_id: int, payload: MLModelUpdateRequest, db: Database):
    model = await db.get(SystemModel, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    model.model_name = payload.model_name
    model.description = payload.description
    model.tags = payload.tags
    await db.commit()
    return success_response(message="目录信息已更新")


@router.post("/upload", summary="模型包上传（尚未开放）", dependencies=[Depends(require_model_admin)])
async def upload_model():
    return error_response(503, "模型包上传将在后续版本开放")


@router.delete("/{model_id}", summary="不支持物理删除模型", dependencies=[Depends(require_model_admin)])
async def delete_model(model_id: int):
    return error_response(405, "不支持物理删除模型，请使用后续开放的版本下线功能")
