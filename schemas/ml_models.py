from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class MLModelResponse(BaseModel):
    """模型上传响应"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    model_name: str
    model_type: str
    framework: str
    weight_file_path: str
    model_file_path: str
    description: Optional[str]
    upload_time: Optional[datetime]
    updated_time: Optional[datetime]


class MLModelListItem(BaseModel):
    """模型列表单条记录响应"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_name: str
    model_type: str
    framework: str
    weight: str
    model_file: str
    description: Optional[str]
    update_date: Optional[str]


class MLModelListResponse(BaseModel):
    """模型列表分页响应"""

    items: list[MLModelListItem]
    total: int
    page: int
    pageSize: int


class MLModelUpdateRequest(BaseModel):
    """模型编辑请求"""

    model_name: Optional[str] = None
    model_type: Optional[str] = None
    framework: Optional[str] = None
    description: Optional[str] = None
