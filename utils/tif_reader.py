import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import CRSError, WindowError
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds


_DEFAULT_NATIVE_PIXEL_CAP = 16384


def _load_native_pixel_cap() -> int:
    raw = os.environ.get("SEGMENT_NATIVE_PIXEL_CAP")
    if not raw:
        return _DEFAULT_NATIVE_PIXEL_CAP
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_NATIVE_PIXEL_CAP
    return value if value > 0 else _DEFAULT_NATIVE_PIXEL_CAP


NATIVE_PIXEL_CAP = _load_native_pixel_cap()


class UnsupportedSrsError(ValueError):
    pass


class NoOverlapError(ValueError):
    pass


@dataclass(frozen=True)
class TifReadResult:
    image: Image.Image
    requested_native_width: int
    requested_native_height: int
    native_width: int
    native_height: int
    read_width: int
    read_height: int
    effective_offset: tuple[int, int]
    effective_size: tuple[int, int]
    capped: bool


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = [float(part.strip()) for part in bbox.split(",")]
    return minx, miny, maxx, maxy


def _cap_dimensions(width: int, height: int) -> tuple[int, int, bool]:
    if width <= NATIVE_PIXEL_CAP and height <= NATIVE_PIXEL_CAP:
        return width, height, False

    scale = min(NATIVE_PIXEL_CAP / width, NATIVE_PIXEL_CAP / height)
    capped_width = max(1, round(width * scale))
    capped_height = max(1, round(height * scale))
    return capped_width, capped_height, True


def read_tif_rgb_window(tif_path: str | Path, bbox: str, srs: str) -> TifReadResult:
    try:
        request_crs = CRS.from_string(srs)
    except CRSError as exc:
        raise UnsupportedSrsError(srs) from exc

    with rasterio.open(str(tif_path)) as ds:
        if ds.crs is None:
            raise RuntimeError(f"TIF 文件缺少坐标系信息: {tif_path}")
        if ds.count < 3:
            raise RuntimeError(f"TIF 文件少于 3 个波段: {tif_path}")
        if any(dtype != "uint8" for dtype in ds.dtypes[:3]):
            raise RuntimeError(f"TIF 前 3 个波段必须为 uint8，实际为 {ds.dtypes[:3]}")

        minx, miny, maxx, maxy = _parse_bbox(bbox)
        left, bottom, right, top = transform_bounds(
            request_crs,
            ds.crs,
            minx,
            miny,
            maxx,
            maxy,
            densify_pts=21,
        )

        requested_window = from_bounds(left, bottom, right, top, transform=ds.transform).round_offsets().round_lengths()
        requested_native_width = int(requested_window.width)
        requested_native_height = int(requested_window.height)
        if requested_native_width <= 0 or requested_native_height <= 0:
            raise NoOverlapError("bbox 与影像无重叠")

        full_window = Window(0, 0, ds.width, ds.height)
        try:
            window = requested_window.intersection(full_window)
        except WindowError as exc:
            raise NoOverlapError("bbox 与影像无重叠") from exc

        window = window.round_offsets().round_lengths()
        native_width = int(window.width)
        native_height = int(window.height)
        if native_width <= 0 or native_height <= 0:
            raise NoOverlapError("bbox 与影像无重叠")

        effective_offset = (
            int(window.col_off - requested_window.col_off),
            int(window.row_off - requested_window.row_off),
        )
        effective_size = (native_width, native_height)

        requested_read_width, requested_read_height, capped = _cap_dimensions(
            requested_native_width,
            requested_native_height,
        )
        if capped:
            scale_x = requested_read_width / requested_native_width
            scale_y = requested_read_height / requested_native_height
            read_width = max(1, round(native_width * scale_x))
            read_height = max(1, round(native_height * scale_y))
        else:
            read_width = native_width
            read_height = native_height

        read_kwargs = {"window": window}
        if capped:
            read_kwargs["out_shape"] = (3, read_height, read_width)
            read_kwargs["resampling"] = Resampling.bilinear

        arr = ds.read([1, 2, 3], **read_kwargs)

    arr = np.moveaxis(arr, 0, -1)

    return TifReadResult(
        image=Image.fromarray(arr, "RGB"),
        requested_native_width=requested_native_width,
        requested_native_height=requested_native_height,
        native_width=native_width,
        native_height=native_height,
        read_width=read_width,
        read_height=read_height,
        effective_offset=effective_offset,
        effective_size=effective_size,
        capped=capped,
    )
