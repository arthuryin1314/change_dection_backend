from functools import lru_cache
from math import pi

import numpy as np
from affine import Affine
from pyproj import CRS, Proj, Transformer
from rasterio.windows import Window


BROADCAST_VARIATION_RTOL = 1e-6
MAX_MATERIALIZED_AREA_CELLS = 16_000_000


class UnsupportedAreaGridError(ValueError):
    pass


def _require_positive_finite(values) -> None:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array) & (array > 0)):
        raise UnsupportedAreaGridError("像元地表面积及比例因子必须为有限正数")


def _transform_key(transform: Affine) -> tuple[float, float, float, float, float, float]:
    return (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )


def _north_up_transform(values) -> Affine:
    transform = Affine(*values)
    if transform.b != 0 or transform.d != 0:
        raise UnsupportedAreaGridError("#7 不支持带旋转或错切的面积网格")
    if transform.a <= 0 or transform.e >= 0:
        raise UnsupportedAreaGridError("#7 只支持北向栅格")
    return transform


def _projected_factors(crs: CRS, transform: Affine, columns, row_position):
    x = transform.c + (np.asarray(columns, dtype=np.float64) + 0.5) * transform.a
    y = np.full_like(x, transform.f + row_position * transform.e)
    to_geographic = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    longitude, latitude = to_geographic.transform(x, y)
    return np.asarray(
        Proj(crs).get_factors(longitude, latitude).areal_scale,
        dtype=np.float64,
    )


@lru_cache(maxsize=8)
def _projected_column_areas(
    crs_wkt: str,
    transform_values: tuple[float, float, float, float, float, float],
    height: int,
    width: int,
) -> np.ndarray:
    crs = CRS.from_wkt(crs_wkt)
    transform = _north_up_transform(transform_values)
    columns = np.arange(width, dtype=np.float64)
    center_factors = _projected_factors(crs, transform, columns, height / 2)
    _require_positive_finite(center_factors)

    sampled_columns = np.unique(np.array([0, width // 2, width - 1]))
    sampled_center = center_factors[sampled_columns]
    for row_position in (0.5, height - 0.5):
        sampled = _projected_factors(
            crs,
            transform,
            sampled_columns,
            row_position,
        )
        _require_positive_finite(sampled)
        relative = np.max(np.abs(sampled - sampled_center) / sampled_center)
        if relative > BROADCAST_VARIATION_RTOL:
            raise UnsupportedAreaGridError(
                "投影面积比例在南北方向变化过大，不能按列广播"
            )

    x_to_m = crs.axis_info[0].unit_conversion_factor
    y_to_m = crs.axis_info[1].unit_conversion_factor
    projected_cell_area = abs(transform.a * transform.e) * x_to_m * y_to_m
    areas = projected_cell_area / center_factors
    _require_positive_finite(areas)
    areas.setflags(write=False)
    return areas


@lru_cache(maxsize=8)
def _geographic_row_areas(
    crs_wkt: str,
    transform_values: tuple[float, float, float, float, float, float],
    height: int,
    width: int,
) -> np.ndarray:
    crs = CRS.from_wkt(crs_wkt)
    transform = _north_up_transform(transform_values)
    x_to_degrees = crs.axis_info[0].unit_conversion_factor * 180 / pi
    y_to_degrees = crs.axis_info[1].unit_conversion_factor * 180 / pi
    left = transform.c * x_to_degrees
    right = (transform.c + transform.a) * x_to_degrees
    geod = crs.get_geod()
    areas = np.empty(height, dtype=np.float64)
    for row in range(height):
        top = (transform.f + row * transform.e) * y_to_degrees
        bottom = (transform.f + (row + 1) * transform.e) * y_to_degrees
        area, _ = geod.polygon_area_perimeter(
            [left, right, right, left],
            [top, top, bottom, bottom],
        )
        areas[row] = abs(area)
    _require_positive_finite(areas)
    areas.setflags(write=False)
    return areas


def _window_slices(
    grid_shape: tuple[int, int],
    window: Window | None,
) -> tuple[slice, slice]:
    height, width = grid_shape
    if height <= 0 or width <= 0:
        raise ValueError("grid_shape 必须为正整数")
    if window is None:
        return slice(0, height), slice(0, width)

    values = (window.col_off, window.row_off, window.width, window.height)
    if any(int(value) != value for value in values):
        raise ValueError("面积窗口必须落在整数像元边界")
    col_off, row_off, window_width, window_height = map(int, values)
    if (
        col_off < 0
        or row_off < 0
        or window_width <= 0
        or window_height <= 0
        or col_off + window_width > width
        or row_off + window_height > height
    ):
        raise ValueError("面积窗口超出栅格范围")
    return (
        slice(row_off, row_off + window_height),
        slice(col_off, col_off + window_width),
    )


def pixel_area_m2(
    crs: str | CRS,
    transform: Affine,
    grid_shape: tuple[int, int],
    window: Window | None = None,
) -> np.ndarray:
    spatial_ref = CRS.from_user_input(crs)
    row_slice, column_slice = _window_slices(grid_shape, window)
    result_height = row_slice.stop - row_slice.start
    result_width = column_slice.stop - column_slice.start
    if result_height * result_width > MAX_MATERIALIZED_AREA_CELLS:
        raise UnsupportedAreaGridError("大面积网格必须使用窗口分块调用")
    height, width = grid_shape
    transform_values = _transform_key(transform)

    if spatial_ref.is_projected:
        columns = _projected_column_areas(
            spatial_ref.to_wkt(),
            transform_values,
            height,
            width,
        )[column_slice]
        result = np.broadcast_to(
            columns[np.newaxis, :],
            (result_height, columns.size),
        )
    elif spatial_ref.is_geographic:
        rows = _geographic_row_areas(
            spatial_ref.to_wkt(),
            transform_values,
            height,
            width,
        )[row_slice]
        result = np.broadcast_to(
            rows[:, np.newaxis],
            (rows.size, result_width),
        )
    else:
        raise UnsupportedAreaGridError("CRS 必须是投影坐标系或经纬度坐标系")
    return np.array(result, dtype=np.float64, copy=True)
