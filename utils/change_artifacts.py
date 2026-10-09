"""Persistent artifacts for a completed change analysis.

The transition GeoTIFF is the lossless, full analysis-grid representation.  The
PNG is only a display derivative and must never be used for statistics.
"""

import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

from utils.classification_contract import (
    CLASSIFICATION_SCHEME_VERSION,
    CLASS_NAMES,
)
from utils.classification_render import CLASS_COLORS
from utils.classification_result_lock import classification_result_lock
from utils.transition_matrix import (
    CLASS_COUNT,
    _read_aligned_pair,
)


ARTIFACT_VERSION = "change-artifacts-v1"
NODATA_VALUE = 255
MAX_VISUAL_EDGE = 1600

# A stable color table for the 30 directed, non-diagonal transitions.  The
# diagonal is intentionally absent because unchanged pixels are not artifacts.
_TRANSITION_COLORS = {
    code: tuple(int(channel) for channel in color)
    for code, color in {
        1: (230, 25, 75),
        2: (60, 180, 75),
        3: (255, 225, 25),
        4: (0, 130, 200),
        5: (245, 130, 48),
        6: (145, 30, 180),
        8: (70, 240, 240),
        9: (240, 50, 230),
        10: (210, 245, 60),
        11: (250, 190, 190),
        12: (0, 128, 128),
        13: (230, 190, 255),
        15: (255, 250, 200),
        16: (128, 0, 0),
        17: (170, 255, 195),
        18: (128, 128, 0),
        19: (255, 215, 180),
        20: (0, 0, 128),
        22: (255, 255, 255),
        23: (100, 100, 100),
        24: (40, 40, 40),
        25: (180, 180, 180),
        26: (80, 80, 80),
        27: (200, 100, 100),
        29: (100, 100, 200),
        30: (200, 200, 100),
        31: (100, 200, 200),
        32: (200, 100, 200),
        33: (150, 150, 50),
        34: (50, 150, 150),
    }.items()
}


def _hex(color: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % color


def _windows(result):
    for row_off in range(0, result.grid.height, 256):
        height = min(256, result.grid.height - row_off)
        for col_off in range(0, result.grid.width, 256):
            width = min(256, result.grid.width - col_off)
            yield row_off, col_off, height, width


def _transition_data(before_classes, before_valid, after_classes, after_valid):
    if np.any((before_valid != 0) & (before_valid != 1)) or np.any(
        (after_valid != 0) & (after_valid != 1)
    ):
        raise ValueError("有效掩膜必须只包含 0/1")
    valid = (before_valid == 1) & (after_valid == 1)
    if np.any(before_classes[valid] > 5) or np.any(after_classes[valid] > 5):
        raise ValueError("类别编号必须在 0-5 之间")
    changed = valid & (before_classes != after_classes)
    result = np.full(before_classes.shape, NODATA_VALUE, dtype=np.uint8)
    result[changed] = (
        before_classes[changed].astype(np.uint16) * CLASS_COUNT
        + after_classes[changed].astype(np.uint16)
    ).astype(np.uint8)
    return result


def _read_transition_window(
    before_classes,
    before_valid,
    after_classes,
    after_valid,
    result,
    row_off: int,
    col_off: int,
    height: int,
    width: int,
):
    if result.before_window is not None:
        before_col, before_row, _, _ = result.before_window
        after_col, after_row, _, _ = result.after_window
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
            result.grid,
            target_window,
        )
        after_classes_data, after_valid_data = _read_aligned_pair(
            after_classes,
            after_valid,
            result.grid,
            target_window,
        )
    return _transition_data(
        before_classes_data,
        before_valid_data,
        after_classes_data,
        after_valid_data,
    )


def _legend(result, actual_codes: list[int]) -> dict:
    classes = []
    for class_id, name in enumerate(CLASS_NAMES):
        color = tuple(int(channel) for channel in CLASS_COLORS[class_id])
        classes.append({"id": class_id, "name": name, "color": _hex(color)})
    transitions = []
    for code in actual_codes:
        before_id, after_id = divmod(code, CLASS_COUNT)
        color = _TRANSITION_COLORS[code]
        transitions.append(
            {
                "code": code,
                "before": classes[before_id],
                "after": classes[after_id],
                "color": _hex(color),
            }
        )
    return {
        "artifact_version": ARTIFACT_VERSION,
        "classification_scheme_version": CLASSIFICATION_SCHEME_VERSION,
        "encoding": {
            "formula": "before_id * 6 + after_id",
            "class_count": CLASS_COUNT,
        },
        "nodata_value": NODATA_VALUE,
        "classes": classes,
        "transitions": transitions,
    }


def _write_visualization(transition_path: Path, output_path: Path) -> tuple[int, int]:
    with rasterio.open(transition_path) as dataset:
        scale = min(1.0, MAX_VISUAL_EDGE / max(dataset.width, dataset.height))
        width = max(1, round(dataset.width * scale))
        height = max(1, round(dataset.height * scale))
        values = dataset.read(
            1,
            out_shape=(height, width),
            resampling=rasterio.enums.Resampling.nearest,
        )
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for code, color in _TRANSITION_COLORS.items():
        mask = values == code
        if np.any(mask):
            rgba[mask, :3] = color
            rgba[mask, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(output_path, format="PNG", optimize=True)
    return width, height


def write_change_artifacts(
    *,
    before_classes_path: str | Path,
    before_valid_mask_path: str | Path,
    after_classes_path: str | Path,
    after_valid_mask_path: str | Path,
    result,
    storage_root: str | Path,
    result_id: str | int,
) -> dict:
    """Atomically publish transition.tif, visualization.png and legend.json."""
    root = Path(storage_root)
    final_directory = root / str(result_id)
    temporary_directory = root / f".{result_id}.{uuid4().hex}.tmp"
    if final_directory.exists():
        raise FileExistsError(f"变化结果 artifact 目录已存在: {result_id}")
    root.mkdir(parents=True, exist_ok=True)
    temporary_directory.mkdir()
    try:
        transition_path = temporary_directory / "transition.tif"
        profile = {
            "driver": "GTiff",
            "width": result.grid.width,
            "height": result.grid.height,
            "count": 1,
            "dtype": "uint8",
            "crs": result.grid.crs,
            "transform": result.grid.transform,
            "nodata": NODATA_VALUE,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }
        actual_codes = set()
        source_paths = (
            before_classes_path,
            before_valid_mask_path,
            after_classes_path,
            after_valid_mask_path,
        )
        with ExitStack() as locks:
            for directory in sorted({str(Path(path).parent.resolve()) for path in source_paths}):
                locks.enter_context(classification_result_lock(directory, timeout_seconds=2))
            with rasterio.open(
                before_classes_path
            ) as before_classes, rasterio.open(
                before_valid_mask_path
            ) as before_valid, rasterio.open(
                after_classes_path
            ) as after_classes, rasterio.open(
                after_valid_mask_path
            ) as after_valid, rasterio.open(transition_path, "w", **profile) as output:
                for row_off, col_off, height, width in _windows(result):
                    transition = _read_transition_window(
                        before_classes,
                        before_valid,
                        after_classes,
                        after_valid,
                        result,
                        row_off,
                        col_off,
                        height,
                        width,
                    )
                    output.write(transition, 1, window=Window(col_off, row_off, width, height))
                    actual_codes.update(int(code) for code in np.unique(transition) if code != NODATA_VALUE)

        visualization_path = temporary_directory / "visualization.png"
        visual_width, visual_height = _write_visualization(transition_path, visualization_path)
        legend_path = temporary_directory / "legend.json"
        legend_path.write_text(
            json.dumps(_legend(result, sorted(actual_codes)), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_directory.replace(final_directory)
    except BaseException:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise

    return {
        "artifact_version": ARTIFACT_VERSION,
        "transition_geotiff": {
            "filename": "transition.tif",
            "media_type": "image/tiff",
            "width": result.grid.width,
            "height": result.grid.height,
            "nodata_value": NODATA_VALUE,
        },
        "visualization_png": {
            "filename": "visualization.png",
            "media_type": "image/png",
            "width": visual_width,
            "height": visual_height,
            "max_edge": MAX_VISUAL_EDGE,
        },
        "legend": {
            "filename": "legend.json",
            "media_type": "application/json",
        },
    }


def remove_change_artifacts(storage_root: str | Path, result_id: str | int) -> None:
    root = Path(storage_root).resolve()
    target = (Path(storage_root) / str(result_id)).resolve()
    if target.parent != root:
        raise RuntimeError("变化结果 artifact 目录越出存储根")
    shutil.rmtree(target, ignore_errors=True)
