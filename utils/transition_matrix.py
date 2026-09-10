from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, hypot, isfinite
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError
from rasterio.errors import RasterioError
from rasterio.warp import calculate_default_transform, transform_bounds
from rasterio.windows import Window

from utils.change_result_errors import (
    AREA_GRID_UNSUPPORTED,
    GRID_MISMATCH,
    IDENTIFICATION_RESULT_BUSY,
    NO_COMMON_VALID_AREA,
    RESULT_CONTRACT_BROKEN,
    SPATIAL_METADATA_UNAVAILABLE,
)
from utils.classification_storage import RasterGrid
from utils.classification_result_lock import classification_result_lock
from utils.pixel_area import UnsupportedAreaGridError, pixel_area_axis_m2


CLASS_COUNT = 6
ALIGNMENT_TOLERANCE_PIXELS = 1e-6
PIXEL_SIZE_RTOL = 1e-9
WINDOW_SIZE = 256
GDAL_CACHE_BYTES = 128 * 1024 * 1024
GRID_POLICY_VERSION = "aligned-grid-v1"
CALCULATION_VERSION = "transition-matrix-v2"
EDGE_RULE = "target_pixel_center_full_cell"


class TransitionMatrixError(ValueError):
    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class TransitionMatrixResult:
    matrix_m2: list[list[float]]
    common_valid_area_m2: float
    grid: RasterGrid
    bounds: list[float]
    before_window: list[int] | None
    after_window: list[int] | None
    analysis: dict


def _same_crs(before, after) -> bool:
    return CRS.from_user_input(before) == CRS.from_user_input(after)


def _close_pixel_size(before: float, after: float) -> bool:
    return abs(before - after) <= PIXEL_SIZE_RTOL * max(abs(before), abs(after))


def _require_supported_pair_transform(before: Affine, after: Affine) -> None:
    if before.b != after.b or before.d != after.d:
        raise TransitionMatrixError(GRID_MISMATCH, "两期栅格方向或旋转不一致")
    if before.b != 0 or before.d != 0:
        raise TransitionMatrixError(
            AREA_GRID_UNSUPPORTED,
            "共享面积能力不支持带旋转或错切的分析网格",
        )
    if before.a <= 0 or before.e >= 0 or after.a <= 0 or after.e >= 0:
        raise TransitionMatrixError(GRID_MISMATCH, "两期栅格必须为北向网格")


def _integer_offset(value: float, label: str) -> int:
    nearest = round(value)
    if abs(value - nearest) > ALIGNMENT_TOLERANCE_PIXELS:
        raise TransitionMatrixError(GRID_MISMATCH, f"两期栅格{label}未对齐到像元边界")
    return nearest


def _require_corner_alignment(
    before_transform: Affine,
    after_transform: Affine,
    before_col: int,
    before_row: int,
    after_col: int,
    after_row: int,
    width: int,
    height: int,
) -> None:
    x_scale = min(abs(before_transform.a), abs(after_transform.a))
    y_scale = min(abs(before_transform.e), abs(after_transform.e))
    for col_delta, row_delta in ((0, 0), (width, 0), (0, height), (width, height)):
        before_x, before_y = before_transform * (
            before_col + col_delta,
            before_row + row_delta,
        )
        after_x, after_y = after_transform * (
            after_col + col_delta,
            after_row + row_delta,
        )
        x_error = abs(before_x - after_x) / x_scale
        y_error = abs(before_y - after_y) / y_scale
        if max(x_error, y_error) > ALIGNMENT_TOLERANCE_PIXELS:
            raise TransitionMatrixError(
                GRID_MISMATCH,
                "两期栅格在共同范围内存在累计对齐偏差",
            )


def _grid_key(dataset) -> tuple:
    return (
        CRS.from_user_input(dataset.crs).to_wkt(),
        *(float(value) for value in dataset.transform[:6]),
        dataset.width,
        dataset.height,
    )


def _canonical_grid_reference(before, after):
    period = "before" if _grid_key(before) <= _grid_key(after) else "after"
    return period, before if period == "before" else after


def _direct_intersection(before, after):
    if not _same_crs(before.crs, after.crs):
        raise TransitionMatrixError(GRID_MISMATCH, "两期栅格坐标系不一致")
    _require_supported_pair_transform(before.transform, after.transform)
    if not _close_pixel_size(before.transform.a, after.transform.a) or not _close_pixel_size(
        before.transform.e,
        after.transform.e,
    ):
        raise TransitionMatrixError(GRID_MISMATCH, "两期栅格像元大小不一致")

    reference_period, reference = _canonical_grid_reference(before, after)
    before_origin_col = _integer_offset(
        (before.transform.c - reference.transform.c) / reference.transform.a,
        "横向原点",
    )
    before_origin_row = _integer_offset(
        (before.transform.f - reference.transform.f) / reference.transform.e,
        "纵向原点",
    )
    after_origin_col = _integer_offset(
        (after.transform.c - reference.transform.c) / reference.transform.a,
        "横向原点",
    )
    after_origin_row = _integer_offset(
        (after.transform.f - reference.transform.f) / reference.transform.e,
        "纵向原点",
    )
    first_col = max(before_origin_col, after_origin_col)
    first_row = max(before_origin_row, after_origin_row)
    last_col = min(before_origin_col + before.width, after_origin_col + after.width)
    last_row = min(before_origin_row + before.height, after_origin_row + after.height)
    width = last_col - first_col
    height = last_row - first_row
    if width <= 0 or height <= 0:
        raise TransitionMatrixError(NO_COMMON_VALID_AREA, "两期影像没有共同空间范围")

    before_col = first_col - before_origin_col
    before_row = first_row - before_origin_row
    after_col = first_col - after_origin_col
    after_row = first_row - after_origin_row

    _require_corner_alignment(
        before.transform,
        after.transform,
        before_col,
        before_row,
        after_col,
        after_row,
        width,
        height,
    )
    transform = reference.transform * Affine.translation(first_col, first_row)
    grid = RasterGrid(width=width, height=height, crs=reference.crs, transform=transform)
    return grid, before_col, before_row, after_col, after_row, reference_period


def _directly_compatible(before, after) -> bool:
    before_transform = before.transform
    after_transform = after.transform
    if not _same_crs(before.crs, after.crs):
        return False
    if (
        before_transform.b != 0
        or before_transform.d != 0
        or after_transform.b != 0
        or after_transform.d != 0
        or before_transform.a <= 0
        or before_transform.e >= 0
        or after_transform.a <= 0
        or after_transform.e >= 0
    ):
        return False
    if not _close_pixel_size(before_transform.a, after_transform.a) or not _close_pixel_size(
        before_transform.e,
        after_transform.e,
    ):
        return False
    _, reference = _canonical_grid_reference(before, after)
    x_offset = (after_transform.c - before_transform.c) / reference.transform.a
    y_offset = (after_transform.f - before_transform.f) / reference.transform.e
    return (
        abs(x_offset - round(x_offset)) <= ALIGNMENT_TOLERANCE_PIXELS
        and abs(y_offset - round(y_offset)) <= ALIGNMENT_TOLERANCE_PIXELS
    )


def _ground_pixel_lengths(dataset) -> tuple[float, float]:
    center_col = dataset.width / 2
    center_row = dataset.height / 2
    center = dataset.transform * (center_col, center_row)
    column = dataset.transform * (center_col + 1, center_row)
    row = dataset.transform * (center_col, center_row + 1)
    spatial_ref = CRS.from_user_input(dataset.crs)
    geodetic_ref = spatial_ref.geodetic_crs
    if geodetic_ref is None:
        raise TransitionMatrixError(
            SPATIAL_METADATA_UNAVAILABLE,
            "源栅格坐标系缺少可用于地表面积换算的大地基准",
        )
    try:
        transformer = Transformer.from_crs(spatial_ref, geodetic_ref, always_xy=True)
        center_lon, center_lat = transformer.transform(*center)
        column_lon, column_lat = transformer.transform(*column)
        row_lon, row_lat = transformer.transform(*row)
        geod = spatial_ref.get_geod()
        _, _, width_m = geod.inv(center_lon, center_lat, column_lon, column_lat)
        _, _, height_m = geod.inv(center_lon, center_lat, row_lon, row_lat)
    except ProjError as exc:
        raise TransitionMatrixError(
            SPATIAL_METADATA_UNAVAILABLE,
            "源栅格坐标系无法用于地表分辨率换算",
        ) from exc
    values = abs(width_m), abs(height_m)
    if not all(isfinite(value) and value > 0 for value in values):
        raise TransitionMatrixError(AREA_GRID_UNSUPPORTED, "无法确定源栅格的有效地表分辨率")
    return values


def _select_reference(before, after):
    before_lengths = _ground_pixel_lengths(before)
    after_lengths = _ground_pixel_lengths(after)
    before_score = before_lengths[0] * before_lengths[1]
    after_score = after_lengths[0] * after_lengths[1]
    if np.isclose(before_score, after_score, rtol=PIXEL_SIZE_RTOL, atol=0):
        period = "before" if _grid_key(before) <= _grid_key(after) else "after"
    else:
        period = "before" if before_score > after_score else "after"
    return period, before if period == "before" else after


def _resolution_in_crs(dataset, target_crs) -> tuple[float, float]:
    if _same_crs(dataset.crs, target_crs):
        transform = dataset.transform
        return hypot(transform.a, transform.d), hypot(transform.b, transform.e)
    transform, _, _ = calculate_default_transform(
        dataset.crs,
        target_crs,
        dataset.width,
        dataset.height,
        *dataset.bounds,
    )
    return abs(transform.a), abs(transform.e)


def _snap_floor(value: float) -> int:
    nearest = round(value)
    return nearest if abs(value - nearest) <= ALIGNMENT_TOLERANCE_PIXELS else floor(value)


def _snap_ceil(value: float) -> int:
    nearest = round(value)
    return nearest if abs(value - nearest) <= ALIGNMENT_TOLERANCE_PIXELS else ceil(value)


def _aligned_grid(before, after):
    reference_period, reference = _select_reference(before, after)
    target_crs = reference.crs
    before_resolution = _resolution_in_crs(before, target_crs)
    after_resolution = _resolution_in_crs(after, target_crs)
    resolution = (
        max(before_resolution[0], after_resolution[0]),
        max(before_resolution[1], after_resolution[1]),
    )
    before_bounds = transform_bounds(before.crs, target_crs, *before.bounds, densify_pts=21)
    after_bounds = transform_bounds(after.crs, target_crs, *after.bounds, densify_pts=21)
    left = max(before_bounds[0], after_bounds[0])
    bottom = max(before_bounds[1], after_bounds[1])
    right = min(before_bounds[2], after_bounds[2])
    top = min(before_bounds[3], after_bounds[3])
    if left >= right or bottom >= top:
        raise TransitionMatrixError(NO_COMMON_VALID_AREA, "两期影像没有共同空间范围")

    anchor_x = reference.bounds.left
    anchor_y = reference.bounds.top
    first_col = _snap_floor((left - anchor_x) / resolution[0])
    last_col = _snap_ceil((right - anchor_x) / resolution[0])
    first_row = _snap_floor((anchor_y - top) / resolution[1])
    last_row = _snap_ceil((anchor_y - bottom) / resolution[1])
    width = last_col - first_col
    height = last_row - first_row
    if width <= 0 or height <= 0:
        raise TransitionMatrixError(NO_COMMON_VALID_AREA, "两期影像没有共同有效像元中心")
    transform = Affine(
        resolution[0],
        0,
        anchor_x + first_col * resolution[0],
        0,
        -resolution[1],
        anchor_y - first_row * resolution[1],
    )
    return RasterGrid(width=width, height=height, crs=target_crs, transform=transform), {
        "alignment_mode": "WARPED",
        "reference_period": reference_period,
        "resolution": [resolution[0], resolution[1]],
        "resampling": "nearest",
        "coordinate_transform_tolerance": 0.0,
        "alignment_tolerance_pixels": ALIGNMENT_TOLERANCE_PIXELS,
        "pixel_size_rtol": PIXEL_SIZE_RTOL,
        "edge_rule": EDGE_RULE,
    }


@lru_cache(maxsize=8)
def _coordinate_transformer(source_wkt: str, target_wkt: str):
    return Transformer.from_crs(source_wkt, target_wkt, always_xy=True)


def _transform_coordinates(source_crs, target_crs, x, y):
    try:
        transformer = _coordinate_transformer(
            CRS.from_user_input(source_crs).to_wkt(),
            CRS.from_user_input(target_crs).to_wkt(),
        )
        return transformer.transform(x, y)
    except ProjError as exc:
        raise TransitionMatrixError(
            SPATIAL_METADATA_UNAVAILABLE,
            "分析网格无法转换到源栅格坐标系",
        ) from exc


def _read_aligned_pair(classes, valid, grid: RasterGrid, window: Window):
    height = int(window.height)
    width = int(window.width)
    columns = np.arange(window.col_off, window.col_off + width, dtype=np.float64) + 0.5
    rows = np.arange(window.row_off, window.row_off + height, dtype=np.float64) + 0.5
    target_columns, target_rows = np.meshgrid(columns, rows)
    transform = grid.transform
    target_x = (
        transform.a * target_columns
        + transform.b * target_rows
        + transform.c
    )
    target_y = (
        transform.d * target_columns
        + transform.e * target_rows
        + transform.f
    )
    if _same_crs(grid.crs, classes.crs):
        source_x, source_y = target_x, target_y
    elif target_x.size == 1:
        source_x, source_y = _transform_coordinates(
            grid.crs,
            classes.crs,
            float(target_x[0, 0]),
            float(target_y[0, 0]),
        )
    else:
        source_x, source_y = _transform_coordinates(
            grid.crs,
            classes.crs,
            target_x.ravel(),
            target_y.ravel(),
        )
    source_x = np.asarray(source_x).reshape(height, width)
    source_y = np.asarray(source_y).reshape(height, width)
    inverse = ~classes.transform
    source_columns = inverse.a * source_x + inverse.b * source_y + inverse.c
    source_rows = inverse.d * source_x + inverse.e * source_y + inverse.f
    finite = np.isfinite(source_columns) & np.isfinite(source_rows)
    source_column_indices = np.zeros((height, width), dtype=np.int64)
    source_row_indices = np.zeros((height, width), dtype=np.int64)
    source_column_indices[finite] = np.floor(source_columns[finite]).astype(np.int64)
    source_row_indices[finite] = np.floor(source_rows[finite]).astype(np.int64)
    inside = (
        finite
        & (source_column_indices >= 0)
        & (source_column_indices < classes.width)
        & (source_row_indices >= 0)
        & (source_row_indices < classes.height)
    )
    aligned_classes = np.zeros((height, width), dtype=np.uint8)
    aligned_valid = np.zeros((height, width), dtype=np.uint8)
    if not np.any(inside):
        return aligned_classes, aligned_valid

    first_column = int(source_column_indices[inside].min())
    last_column = int(source_column_indices[inside].max())
    first_row = int(source_row_indices[inside].min())
    last_row = int(source_row_indices[inside].max())
    source_window = Window(
        first_column,
        first_row,
        last_column - first_column + 1,
        last_row - first_row + 1,
    )
    source_classes = classes.read(1, window=source_window)
    source_valid = valid.read(1, window=source_window)
    local_rows = source_row_indices[inside] - first_row
    local_columns = source_column_indices[inside] - first_column
    aligned_classes[inside] = source_classes[local_rows, local_columns]
    aligned_valid[inside] = source_valid[local_rows, local_columns]
    aligned_classes[aligned_valid == 0] = 0
    return aligned_classes, aligned_valid


def _require_raster_contract(classes, valid, label: str) -> None:
    if classes.crs is None or valid.crs is None:
        raise TransitionMatrixError(
            SPATIAL_METADATA_UNAVAILABLE,
            f"{label}类别栅格或有效掩膜缺少坐标系",
        )
    transform_values = (*classes.transform[:6], *valid.transform[:6])
    if not all(isfinite(value) for value in transform_values):
        raise TransitionMatrixError(
            SPATIAL_METADATA_UNAVAILABLE,
            f"{label}类别栅格或有效掩膜的空间变换不可用",
        )
    for transform in (classes.transform, valid.transform):
        if transform.a * transform.e - transform.b * transform.d == 0:
            raise TransitionMatrixError(
                SPATIAL_METADATA_UNAVAILABLE,
                f"{label}类别栅格或有效掩膜的空间变换不可逆",
            )
    if (
        classes.count != 1
        or valid.count != 1
        or classes.dtypes != ("uint8",)
        or valid.dtypes != ("uint8",)
        or classes.width != valid.width
        or classes.height != valid.height
        or not _same_crs(classes.crs, valid.crs)
        or classes.transform != valid.transform
    ):
        raise TransitionMatrixError(
            RESULT_CONTRACT_BROKEN,
            f"{label}类别栅格与有效掩膜契约不一致",
        )


def _accumulate_window(
    counts: np.ndarray,
    before_classes: np.ndarray,
    before_valid: np.ndarray,
    after_classes: np.ndarray,
    after_valid: np.ndarray,
    *,
    axis: str,
    row_off: int,
    col_off: int,
) -> int:
    if np.any((before_valid != 0) & (before_valid != 1)) or np.any(
        (after_valid != 0) & (after_valid != 1)
    ):
        raise TransitionMatrixError(
            RESULT_CONTRACT_BROKEN,
            "有效掩膜必须只包含 0/1",
        )
    selected = (before_valid == 1) & (after_valid == 1)
    before_values = before_classes[selected]
    after_values = after_classes[selected]
    if before_values.size and (
        int(before_values.max()) >= CLASS_COUNT or int(after_values.max()) >= CLASS_COUNT
    ):
        raise TransitionMatrixError(
            RESULT_CONTRACT_BROKEN,
            "类别编号必须在 0-5 之间",
        )

    height, width = before_classes.shape
    pairs = before_values.astype(np.int64) * CLASS_COUNT + after_values.astype(np.int64)
    if axis == "column":
        positions = np.broadcast_to(np.arange(width), (height, width))[selected]
        encoded = pairs * width + positions
        local = np.bincount(encoded, minlength=CLASS_COUNT**2 * width).reshape(
            CLASS_COUNT**2,
            width,
        )
        counts[:, col_off : col_off + width] += local
    else:
        positions = np.broadcast_to(np.arange(height)[:, None], (height, width))[selected]
        encoded = pairs * height + positions
        local = np.bincount(encoded, minlength=CLASS_COUNT**2 * height).reshape(
            CLASS_COUNT**2,
            height,
        )
        counts[:, row_off : row_off + height] += local
    return int(before_values.size)


def _locked_result_directories(paths: list[str | Path]):
    stack = ExitStack()
    for directory in sorted({str(Path(path).parent.resolve()) for path in paths}):
        stack.enter_context(classification_result_lock(directory, timeout_seconds=2))
    return stack


def compute_transition_matrix_m2(
    before_classes_path: str | Path,
    before_valid_mask_path: str | Path,
    after_classes_path: str | Path,
    after_valid_mask_path: str | Path,
    *,
    window_size: int = WINDOW_SIZE,
) -> TransitionMatrixResult:
    if window_size <= 0:
        raise ValueError("window_size 必须为正整数")
    paths = [
        before_classes_path,
        before_valid_mask_path,
        after_classes_path,
        after_valid_mask_path,
    ]
    try:
        with _locked_result_directories(paths), rasterio.Env(
            GDAL_CACHEMAX=GDAL_CACHE_BYTES
        ), rasterio.open(before_classes_path) as before_classes, rasterio.open(
            before_valid_mask_path
        ) as before_valid, rasterio.open(after_classes_path) as after_classes, rasterio.open(
            after_valid_mask_path
        ) as after_valid:
            _require_raster_contract(before_classes, before_valid, "前期")
            _require_raster_contract(after_classes, after_valid, "后期")
            direct = _directly_compatible(before_classes, after_classes)
            if direct:
                try:
                    (
                        grid,
                        before_col,
                        before_row,
                        after_col,
                        after_row,
                        reference_period,
                    ) = _direct_intersection(
                        before_classes,
                        after_classes,
                    )
                except TransitionMatrixError as exc:
                    if exc.error_code != GRID_MISMATCH:
                        raise
                    direct = False
            if direct:
                analysis = {
                    "alignment_mode": "DIRECT",
                    "reference_period": reference_period,
                    "resolution": [grid.transform.a, abs(grid.transform.e)],
                    "resampling": "none",
                    "coordinate_transform_tolerance": 0.0,
                    "alignment_tolerance_pixels": ALIGNMENT_TOLERANCE_PIXELS,
                    "pixel_size_rtol": PIXEL_SIZE_RTOL,
                    "edge_rule": EDGE_RULE,
                }
            else:
                try:
                    grid, analysis = _aligned_grid(before_classes, after_classes)
                except (ProjError, RasterioError) as exc:
                    raise TransitionMatrixError(
                        SPATIAL_METADATA_UNAVAILABLE,
                        "无法建立两期栅格的共同分析网格",
                    ) from exc
                before_col = before_row = after_col = after_row = 0
            try:
                axis, pixel_areas = pixel_area_axis_m2(
                    grid.crs,
                    grid.transform,
                    (grid.height, grid.width),
                )
            except UnsupportedAreaGridError as exc:
                raise TransitionMatrixError(AREA_GRID_UNSUPPORTED, str(exc)) from exc

            axis_length = grid.width if axis == "column" else grid.height
            counts = np.zeros((CLASS_COUNT**2, axis_length), dtype=np.int64)
            common_valid_pixels = 0
            for row_off in range(0, grid.height, window_size):
                height = min(window_size, grid.height - row_off)
                for col_off in range(0, grid.width, window_size):
                    width = min(window_size, grid.width - col_off)
                    if direct:
                        before_window = Window(
                            before_col + col_off,
                            before_row + row_off,
                            width,
                            height,
                        )
                        after_window = Window(
                            after_col + col_off,
                            after_row + row_off,
                            width,
                            height,
                        )
                        before_classes_data = before_classes.read(1, window=before_window)
                        before_valid_data = before_valid.read(1, window=before_window)
                        after_classes_data = after_classes.read(1, window=after_window)
                        after_valid_data = after_valid.read(1, window=after_window)
                    else:
                        target_window = Window(col_off, row_off, width, height)
                        before_classes_data, before_valid_data = _read_aligned_pair(
                            before_classes,
                            before_valid,
                            grid,
                            target_window,
                        )
                        after_classes_data, after_valid_data = _read_aligned_pair(
                            after_classes,
                            after_valid,
                            grid,
                            target_window,
                        )
                    common_valid_pixels += _accumulate_window(
                        counts,
                        before_classes_data,
                        before_valid_data,
                        after_classes_data,
                        after_valid_data,
                        axis=axis,
                        row_off=row_off,
                        col_off=col_off,
                    )
    except TimeoutError as exc:
        raise TransitionMatrixError(
            IDENTIFICATION_RESULT_BUSY,
            "识别结果正在使用，请稍后重试",
        ) from exc

    if common_valid_pixels == 0:
        raise TransitionMatrixError(NO_COMMON_VALID_AREA, "两期影像没有共同有效区域")
    matrix = (counts.astype(np.float64) @ pixel_areas).reshape(CLASS_COUNT, CLASS_COUNT)
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("转移矩阵必须包含有限非负面积")
    bounds = rasterio.transform.array_bounds(grid.height, grid.width, grid.transform)
    return TransitionMatrixResult(
        matrix_m2=matrix.tolist(),
        common_valid_area_m2=float(matrix.sum()),
        grid=grid,
        bounds=list(bounds),
        before_window=(
            [before_col, before_row, grid.width, grid.height] if direct else None
        ),
        after_window=(
            [after_col, after_row, grid.width, grid.height] if direct else None
        ),
        analysis=analysis,
    )
