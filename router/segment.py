import asyncio
import base64
import io
import re
from typing import Annotated, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from config.db_config import get_db
from crud import images as crud_images
from utils import deeplab_service
from utils.geoserver_utils import GEOSERVER_PASSWORD, GEOSERVER_USER, GEOSERVER_WORKSPACE
from utils.get_user_by_token import get_current_user
from utils.response import success_response

router = APIRouter(prefix="/api", tags=["segment"])

_SRS_RE = re.compile(r"^EPSG:\d{4,6}$", re.IGNORECASE)
_WMS_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
_WMS_RETRYABLE_CODES = {429, 502, 503, 504}
_WMS_MAX_RETRIES = 3
_WMS_BASE_DELAY = 0.5


class SegmentRequest(BaseModel):
    image_id: int = Field(..., gt=0)
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
        invalid = [class_id for class_id in value if class_id < 1 or class_id > 5]
        if invalid:
            raise ValueError(f"classes 中的类别 ID 必须在 1-5 之间，非法值：{invalid}")
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


def _qualified_layer_name(layer_name: str) -> str:
    name = layer_name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="影像缺少 GeoServer 图层名称")
    return name if ":" in name else f"{GEOSERVER_WORKSPACE}:{name}"


def _build_getmap_url(
    wms_url: str,
    layer_name: str,
    bbox: str,
    width: int,
    height: int,
    srs: str,
) -> str:
    if not wms_url:
        raise HTTPException(status_code=422, detail="影像缺少 WMS 地址")

    split = urlsplit(wms_url)
    query = dict(parse_qsl(split.query, keep_blank_values=True))
    query.update(
        {
            "service": "WMS",
            "version": query.get("version") or "1.1.0",
            "request": "GetMap",
            "layers": query.get("layers") or _qualified_layer_name(layer_name),
            "styles": query.get("styles", ""),
            "format": "image/png",
            "transparent": "true",
            "bbox": bbox,
            "width": str(width),
            "height": str(height),
            "srs": srs,
        }
    )
    query.pop("crs", None)
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(query), split.fragment))


async def _fetch_wms_png(url: str) -> bytes:
    auth = (GEOSERVER_USER, GEOSERVER_PASSWORD) if GEOSERVER_USER and GEOSERVER_PASSWORD else None
    last_response: httpx.Response | None = None

    async with httpx.AsyncClient(timeout=_WMS_TIMEOUT) as client:
        for attempt in range(_WMS_MAX_RETRIES):
            try:
                response = await client.get(url, auth=auth)
                if response.status_code in _WMS_RETRYABLE_CODES and attempt < _WMS_MAX_RETRIES - 1:
                    last_response = response
                    await asyncio.sleep(_WMS_BASE_DELAY * (2 ** attempt))
                    continue
                last_response = response
                break
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError):
                if attempt < _WMS_MAX_RETRIES - 1:
                    await asyncio.sleep(_WMS_BASE_DELAY * (2 ** attempt))
                    continue
                raise

    if last_response.status_code != 200:
        detail = last_response.text[:300] if last_response.text else last_response.reason_phrase
        raise HTTPException(status_code=502, detail=f"GeoServer GetMap 失败 [{last_response.status_code}]: {detail}")

    content_type = last_response.headers.get("content-type", "").lower()
    if "image" not in content_type and not last_response.content.startswith(b"\x89PNG"):
        detail = last_response.text[:300] if last_response.text else "响应不是图片"
        raise HTTPException(status_code=502, detail=f"GeoServer GetMap 响应异常: {detail}")

    return last_response.content


def _open_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=502, detail="GeoServer 返回的图片无法解析") from exc


@router.post("/segment", summary="地物识别")
async def segment_image(
    payload: SegmentRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user=Depends(get_current_user),
):
    image = await crud_images.get_image_by_id(db, payload.image_id, current_user.id)
    if image is None:
        raise HTTPException(status_code=404, detail="影像不存在")
    if not image.wms_url or not image.layer_name:
        raise HTTPException(status_code=422, detail="影像未发布到 GeoServer，无法识别")

    getmap_url = _build_getmap_url(
        wms_url=image.wms_url,
        layer_name=image.layer_name,
        bbox=payload.bbox,
        width=payload.width,
        height=payload.height,
        srs=payload.srs,
    )

    source_png = await _fetch_wms_png(getmap_url)
    source_image = await asyncio.to_thread(_open_image, source_png)
    result_png = await asyncio.to_thread(deeplab_service.segment_rgba_png, source_image, payload.classes)
    encoded = base64.b64encode(result_png).decode("ascii")

    return success_response(
        message="识别成功",
        data={
            "image": f"data:image/png;base64,{encoded}",
            "bbox": payload.bbox,
        },
    )
