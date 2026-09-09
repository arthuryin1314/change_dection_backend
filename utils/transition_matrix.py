from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.windows import Window

from utils.change_result_errors import (
    AREA_GRID_UNSUPPORTED,
    GRID_MISMATCH,
    IDENTIFICATION_RESULT_BUSY,
    NO_COMMON_VALID_AREA,
)
from utils.classification_storage import RasterGrid
from utils.classification_result_lock import classification_result_lock
from utils.pixel_area import UnsupportedAreaGridError, pixel_area_axis_m2


CLASS_COUNT = 6
ALIGNMENT_TOLERANCE_PIXELS = 1e-6
PIXEL_SIZE_RTOL = 1e-9
WINDOW_SIZE = 256
GDAL_CACHE_BYTES = 128 * 1024 * 1024


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
    before_window: list[int]
    after_window: list[int]


def _same_crs(before, after) -> bool:
    return CRS.from_user_input(before) == CRS.from_user_input(after)


def _close_pixel_size(before: float, after: float) -> bool:
    return bool(np.isclose(before, after, rtol=PIXEL_SIZE_RTOL, atol=0))


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
    for col_delta, row_delta in ((0, 0), (width, 0), (0, height), (width, height)):
        before_x, before_y = before_transform * (
            before_col + col_delta,
            before_row + row_delta,
        )
        after_x, after_y = after_transform * (
            after_col + col_delta,
            after_row + row_delta,
        )
        x_error = abs(before_x - after_x) / abs(before_transform.a)
        y_error = abs(before_y - after_y) / abs(before_transform.e)
        if max(x_error, y_error) > ALIGNMENT_TOLERANCE_PIXELS:
            raise TransitionMatrixError(
                GRID_MISMATCH,
                "两期栅格在共同范围内存在累计对齐偏差",
            )


def _intersection(before, after):
    if not _same_crs(before.crs, after.crs):
        raise TransitionMatrixError(GRID_MISMATCH, "两期栅格坐标系不一致")
    _require_supported_pair_transform(before.transform, after.transform)
    if not _close_pixel_size(before.transform.a, after.transform.a) or not _close_pixel_size(
        before.transform.e,
        after.transform.e,
    ):
        raise TransitionMatrixError(GRID_MISMATCH, "两期栅格像元大小不一致")

    col_offset = _integer_offset(
        (after.transform.c - before.transform.c) / before.transform.a,
        "横向原点",
    )
    row_offset = _integer_offset(
        (after.transform.f - before.transform.f) / before.transform.e,
        "纵向原点",
    )
    before_col = max(0, col_offset)
    before_row = max(0, row_offset)
    after_col = max(0, -col_offset)
    after_row = max(0, -row_offset)
    width = min(before.width - before_col, after.width - after_col)
    height = min(before.height - before_row, after.height - after_row)
    if width <= 0 or height <= 0:
        raise TransitionMatrixError(NO_COMMON_VALID_AREA, "两期影像没有共同空间范围")

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
    transform = before.transform * Affine.translation(before_col, before_row)
    grid = RasterGrid(width=width, height=height, crs=before.crs, transform=transform)
    return grid, before_col, before_row, after_col, after_row


def _require_raster_contract(classes, valid, label: str) -> None:
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
            GRID_MISMATCH,
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
        raise ValueError("有效掩膜必须只包含 0/1")
    selected = (before_valid == 1) & (after_valid == 1)
    before_values = before_classes[selected]
    after_values = after_classes[selected]
    if before_values.size and (
        int(before_values.max()) >= CLASS_COUNT or int(after_values.max()) >= CLASS_COUNT
    ):
        raise ValueError("类别编号必须在 0-5 之间")

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
) -> TransitionMatrixResult:
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
            grid, before_col, before_row, after_col, after_row = _intersection(
                before_classes,
                after_classes,
            )
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
            for row_off in range(0, grid.height, WINDOW_SIZE):
                height = min(WINDOW_SIZE, grid.height - row_off)
                for col_off in range(0, grid.width, WINDOW_SIZE):
                    width = min(WINDOW_SIZE, grid.width - col_off)
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
                    common_valid_pixels += _accumulate_window(
                        counts,
                        before_classes.read(1, window=before_window),
                        before_valid.read(1, window=before_window),
                        after_classes.read(1, window=after_window),
                        after_valid.read(1, window=after_window),
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
        before_window=[before_col, before_row, grid.width, grid.height],
        after_window=[after_col, after_row, grid.width, grid.height],
    )
