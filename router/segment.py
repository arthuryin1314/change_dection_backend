import asyncio
import base64
import io
import logging
import os
import re
from contextlib import contextmanager
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from config.db_config import get_db
from crud import images as crud_images
from crud import ml_models as crud_ml_models
from utils import deeplab_service
from utils.get_user_by_token import get_current_user
from utils.response import success_response
from utils.tif_reader import NoOverlapError, UnsupportedSrsError, _cap_dimensions, read_tif_rgb_window


@contextmanager
def _allow_large_intermediate_image():
    """临时禁用 PIL 解压炸弹检查，仅用于打开我们自己生成的可信中间 PNG。"""
    saved = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = saved


router = APIRouter(prefix="/api", tags=["segment"])
logger = logging.getLogger(__name__)

_SRS_RE = re.compile(r"^EPSG:\d{4,6}$", re.IGNORECASE)
SUPPORTED_MODEL_TYPE = "semantic_segmentation"
SUPPORTED_FRAMEWORK = "PyTorch"
SUPPORTED_WEIGHT_SUFFIXES = {".pth", ".pt"}
MODEL_INCOMPATIBLE_DETAIL = "模型不兼容"
SEGMENT_FAILURE_DETAIL = "地物识别失败，请稍后重试"


def _is_readable_file(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.R_OK)


class SegmentRequest(BaseModel):
    image_id: int = Field(..., gt=0)
    model_id: int = Field(..., gt=0)
    bbox: str
    width: int = Field(..., ge=1, le=4096)
    height: int = Field(..., ge=1, le=4096)
    srs: str = Field(default="EPSG:4326")
    classes: Optional[list[int]] = Field(default=None)

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: Optional[list[int]]) -> Optional[list[int]]:
        if value is None:
            return value
        invalid = [class_id for class_id in value if class_id < 0 or class_id > 5]
        if invalid:
            raise ValueError(f"classes 中的类别 ID 必须在 0-5 之间，非法值：{invalid}")
        return list(dict.fromkeys(value))

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: str) -> str:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox 必须为 minx,miny,maxx,maxy")

        try:
            minx, miny, maxx, maxy = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError("bbox 必须包含 4 个数字") from exc

        if minx >= maxx or miny >= maxy:
            raise ValueError("bbox 范围非法")

        return ",".join(parts)

    @field_validator("srs")
    @classmethod
    def validate_srs(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not _SRS_RE.fullmatch(normalized):
            raise ValueError("srs 必须为 EPSG 编码，例如 EPSG:4326")
        return normalized


@router.post("/segment", summary="地物识别")
async def segment_image(
    payload: SegmentRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user=Depends(get_current_user),
):
    try:
        image = await crud_images.get_image_by_id(db, payload.image_id, current_user.id)
        if image is None:
            raise HTTPException(status_code=404, detail="影像不存在")

        model = await crud_ml_models.get_ml_model_by_id(db, payload.model_id, current_user.id)
        if model is None:
            raise HTTPException(status_code=404, detail="模型不存在")

        if (
            model.model_type != SUPPORTED_MODEL_TYPE
            or model.framework != SUPPORTED_FRAMEWORK
            or os.path.splitext(model.weight_file_path)[1].lower() not in SUPPORTED_WEIGHT_SUFFIXES
            or not _is_readable_file(model.weight_file_path)
        ):
            raise HTTPException(status_code=422, detail=MODEL_INCOMPATIBLE_DETAIL)

        if not image.img_path or not _is_readable_file(image.img_path):
            raise HTTPException(status_code=422, detail="影像源文件不存在或不可访问")

        try:
            tif_result = await asyncio.to_thread(read_tif_rgb_window, image.img_path, payload.bbox, payload.srs)
        except UnsupportedSrsError as exc:
            raise HTTPException(status_code=422, detail=f"srs 不被支持: {payload.srs}") from exc
        except NoOverlapError as exc:
            raise HTTPException(status_code=422, detail="bbox 与影像无重叠") from exc

        if tif_result.capped:
            logger.info(
                "Segment native window capped: image_id=%s native=%sx%s read=%sx%s cap=%s",
                payload.image_id,
                tif_result.requested_native_width,
                tif_result.requested_native_height,
                tif_result.read_width,
                tif_result.read_height,
                max(_cap_dimensions(tif_result.requested_native_width, tif_result.requested_native_height)[:2]),
            )

        try:
            result_png = await asyncio.to_thread(
                deeplab_service.segment_rgba_png,
                tif_result.image,
                payload.model_id,
                model.weight_file_path,
                payload.classes,
                weight_sha256=model.weight_content_sha256,
            )
        except deeplab_service.ModelLoadError as exc:
            logger.warning(
                "模型权重不兼容: model_id=%s weight_file_path=%s",
                payload.model_id,
                model.weight_file_path,
                exc_info=True,
            )
            raise HTTPException(status_code=422, detail=MODEL_INCOMPATIBLE_DETAIL) from exc
        result_image = await asyncio.to_thread(
            _compose_and_resize_result_png,
            result_png,
            tif_result,
            payload.width,
            payload.height,
        )
        encoded = base64.b64encode(result_image).decode("ascii")

        return success_response(
            message="识别成功",
            data={
                "image": f"data:image/png;base64,{encoded}",
                "bbox": payload.bbox,
            },
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "地物识别失败: image_id=%s model_id=%s user_id=%s",
            payload.image_id,
            payload.model_id,
            current_user.id,
        )
        raise HTTPException(status_code=500, detail=SEGMENT_FAILURE_DETAIL)


def _compose_and_resize_result_png(result_png: bytes, tif_result, width: int, height: int) -> bytes:
    with _allow_large_intermediate_image():
        segment_image = Image.open(io.BytesIO(result_png)).convert("RGBA")
        segment_image.load()

    canvas_width, canvas_height, _ = _cap_dimensions(
        tif_result.requested_native_width,
        tif_result.requested_native_height,
    )
    scale_x = canvas_width / tif_result.requested_native_width
    scale_y = canvas_height / tif_result.requested_native_height
    offset_x = round(tif_result.effective_offset[0] * scale_x)
    offset_y = round(tif_result.effective_offset[1] * scale_y)
    paste_width = max(1, round(tif_result.effective_size[0] * scale_x))
    paste_height = max(1, round(tif_result.effective_size[1] * scale_y))

    if segment_image.size != (paste_width, paste_height):
        segment_image = segment_image.resize((paste_width, paste_height), Image.NEAREST)

    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    canvas.paste(segment_image, (offset_x, offset_y))
    if canvas.size != (width, height):
        canvas = canvas.resize((width, height), Image.NEAREST)

    output = io.BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()
