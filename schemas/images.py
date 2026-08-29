from datetime import date, datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ImageCreate(BaseModel):
    """影像创建请求"""
    ImgName: str = Field(..., description="影像名称")
    ImgResolution: float = Field(..., description="影像分辨率（米）")
    ImgDate: date = Field(..., description="影像日期")
    satellite: str = Field(..., description="卫星名称")
    type: str = Field(..., description="影像类型")
    region_code: str = Field(..., description="区域代码")


class ImageResponse(BaseModel):
    """影像响应"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_name: str
    resolution: Optional[float]
    capture_date: Optional[date]
    satellite: Optional[str]
    image_type: Optional[str]
    region_code: Optional[str]
    img_path: Optional[str]
    bbox: Optional[list[float]] = None
    layer_name: Optional[str] = None
    wms_url: Optional[str] = None
    shp_path: Optional[str] = None
    dbf_path: Optional[str] = None
    prj_path: Optional[str] = None
    upload_time: Optional[datetime]


class ImagePageData(BaseModel):
    items: List[ImageResponse]
    total: int
    page: int
    page_size: int = Field(serialization_alias="pageSize")
    total_pages: int = Field(serialization_alias="totalPages")


class ImagePageResponse(BaseModel):
    code: int
    message: str
    data: ImagePageData


class ImageRecordResponse(BaseModel):
    code: int
    message: str
    data: ImageResponse


class UploadSessionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_name: str = Field(alias="fileName", min_length=1, max_length=128)
    file_size: int = Field(alias="fileSize", gt=0)
    chunk_size: int = Field(alias="chunkSize", gt=0)
    total_chunks: int = Field(alias="totalChunks", gt=0)
    file_hash: str = Field(
        alias="fileHash",
        min_length=32,
        max_length=32,
        pattern=r"^[0-9a-fA-F]{32}$",
    )


class UploadSessionData(BaseModel):
    upload_id: str = Field(serialization_alias="uploadId")
    uploaded_chunks: List[int] = Field(serialization_alias="uploadedChunks")
    total_chunks: int = Field(serialization_alias="totalChunks")


class UploadSessionResponse(BaseModel):
    code: int
    message: str
    data: UploadSessionData


class BoundaryFileResponse(BaseModel):
    """边界文件响应"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_id: int
    file_prefix: Optional[str]
    shp_path: Optional[str]
    dbf_path: Optional[str]
    prj_path: Optional[str]
