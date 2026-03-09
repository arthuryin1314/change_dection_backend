from datetime import date

from pydantic import BaseModel,Field


class Image(BaseModel):
    ImgName: str = Field(..., description="影像名称")
    ImgResolution: float = Field(..., description="影像分辨率（米）")
    ImgDate: date = Field(..., description="影像日期")
    satellite: str = Field(..., description="卫星名称")
    type: str = Field(..., description="影像类型")
    region_code: str = Field(..., description="区域代码")
    img_path: str = Field(..., description="影像路径")
