import io

import numpy as np
import rasterio
from PIL import Image
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds


CLASS_COLORS = np.array(
    [
        (0, 0, 0),
        (0, 0, 255),
        (0, 128, 0),
        (128, 128, 128),
        (0, 255, 0),
        (255, 0, 0),
    ],
    dtype=np.uint8,
)


class RenderBoundsError(ValueError):
    pass


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(part.strip()) for part in bbox.split(","))
    except ValueError as exc:
        raise RenderBoundsError("bbox 必须包含 4 个数字") from exc
    if len(values) != 4:
        raise RenderBoundsError("bbox 必须为 minx,miny,maxx,maxy")
    left, bottom, right, top = values
    if left >= right or bottom >= top:
        raise RenderBoundsError("bbox 范围非法")
    return values


def render_classification_png(
    classes_path: str,
    valid_mask_path: str,
    *,
    bbox: str,
    width: int,
    height: int,
    srs: str,
    classes: list[int],
) -> bytes:
    request_crs = CRS.from_user_input(srs)
    requested_bounds = _parse_bbox(bbox)
    with rasterio.open(classes_path) as classes_ds, rasterio.open(
        valid_mask_path
    ) as valid_ds:
        source_bounds = transform_bounds(
            classes_ds.crs,
            request_crs,
            *classes_ds.bounds,
            densify_pts=21,
        )
        if (
            requested_bounds[2] <= source_bounds[0]
            or requested_bounds[0] >= source_bounds[2]
            or requested_bounds[3] <= source_bounds[1]
            or requested_bounds[1] >= source_bounds[3]
        ):
            raise RenderBoundsError("bbox 与识别结果无重叠")

        destination_transform = from_bounds(*requested_bounds, width, height)
        class_data = np.zeros((height, width), dtype=np.uint8)
        valid_data = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=rasterio.band(classes_ds, 1),
            destination=class_data,
            src_transform=classes_ds.transform,
            src_crs=classes_ds.crs,
            dst_transform=destination_transform,
            dst_crs=request_crs,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
        )
        reproject(
            source=rasterio.band(valid_ds, 1),
            destination=valid_data,
            src_transform=valid_ds.transform,
            src_crs=valid_ds.crs,
            dst_transform=destination_transform,
            dst_crs=request_crs,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
        )

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = CLASS_COLORS[class_data]
    selected = np.isin(class_data, classes)
    rgba[:, :, 3] = np.where((valid_data == 1) & selected, 180, 0)
    output = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(output, format="PNG")
    return output.getvalue()

