from pathlib import Path
from numbers import Real

import numpy as np
import rasterio

from utils.pixel_area import pixel_area_axis_m2
from utils.classification_result_lock import classification_result_lock


CLASS_COUNT = 6
GDAL_CACHE_BYTES = 128 * 1024 * 1024


def validate_class_area_m2(values) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != CLASS_COUNT:
        raise ValueError("六类面积必须包含六个有限非负数")
    if any(not isinstance(value, Real) or isinstance(value, bool) for value in values):
        raise ValueError("六类面积必须包含六个有限非负数")
    areas = [float(value) for value in values]
    if not np.all(np.isfinite(areas)) or any(value < 0 for value in areas):
        raise ValueError("六类面积必须包含六个有限非负数")
    return areas


def _same_grid(classes, valid_mask) -> bool:
    return (
        classes.width == valid_mask.width
        and classes.height == valid_mask.height
        and classes.crs == valid_mask.crs
        and classes.transform.almost_equals(valid_mask.transform)
    )


def _add_window_counts(
    counts: np.ndarray,
    classes: np.ndarray,
    valid_mask: np.ndarray,
    *,
    axis: str,
    row_off: int,
    col_off: int,
) -> None:
    if np.any((valid_mask != 0) & (valid_mask != 1)):
        raise ValueError("有效掩膜必须只包含 0/1")
    selected = valid_mask == 1
    selected_classes = classes[selected]
    if selected_classes.size and int(selected_classes.max()) >= CLASS_COUNT:
        raise ValueError("类别编号必须在 0-5 之间")

    height, width = classes.shape
    if axis == "column":
        positions = np.broadcast_to(
            np.arange(width, dtype=np.int64),
            (height, width),
        )[selected]
        encoded = selected_classes.astype(np.int64) * width + positions
        local = np.bincount(encoded, minlength=CLASS_COUNT * width).reshape(
            CLASS_COUNT,
            width,
        )
        counts[:, col_off : col_off + width] += local
        return

    positions = np.broadcast_to(
        np.arange(height, dtype=np.int64)[:, np.newaxis],
        (height, width),
    )[selected]
    encoded = selected_classes.astype(np.int64) * height + positions
    local = np.bincount(encoded, minlength=CLASS_COUNT * height).reshape(
        CLASS_COUNT,
        height,
    )
    counts[:, row_off : row_off + height] += local


def _summarize_classification_area_m2_unlocked(
    classes_path: str | Path,
    valid_mask_path: str | Path,
) -> list[float]:
    with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_BYTES), rasterio.open(
        classes_path
    ) as classes_ds, rasterio.open(valid_mask_path) as valid_mask_ds:
        if not _same_grid(classes_ds, valid_mask_ds):
            raise ValueError("类别栅格与有效掩膜的空间网格不一致")
        if classes_ds.count != 1 or classes_ds.dtypes != ("uint8",):
            raise ValueError("类别栅格必须是单波段 uint8")
        if valid_mask_ds.count != 1 or valid_mask_ds.dtypes != ("uint8",):
            raise ValueError("有效掩膜必须是单波段 uint8")

        grid_shape = (classes_ds.height, classes_ds.width)
        axis, pixel_areas = pixel_area_axis_m2(
            classes_ds.crs,
            classes_ds.transform,
            grid_shape,
        )
        axis_length = classes_ds.width if axis == "column" else classes_ds.height
        counts = np.zeros((CLASS_COUNT, axis_length), dtype=np.int64)

        for _, window in classes_ds.block_windows(1):
            classes = classes_ds.read(1, window=window)
            valid_mask = valid_mask_ds.read(1, window=window)
            _add_window_counts(
                counts,
                classes,
                valid_mask,
                axis=axis,
                row_off=int(window.row_off),
                col_off=int(window.col_off),
            )

    areas = counts.astype(np.float64) @ pixel_areas
    return validate_class_area_m2(areas.tolist())


def summarize_classification_area_m2(
    classes_path: str | Path,
    valid_mask_path: str | Path,
) -> list[float]:
    with classification_result_lock(Path(classes_path).parent):
        return _summarize_classification_area_m2_unlocked(
            classes_path,
            valid_mask_path,
        )
