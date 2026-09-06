import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS as PyprojCRS, Transformer
from rasterio.crs import CRS
from rasterio.windows import Window


@dataclass(frozen=True)
class RasterGrid:
    width: int
    height: int
    crs: str | CRS
    transform: Affine


@dataclass(frozen=True)
class StoredClassification:
    directory: Path
    classes_path: Path
    valid_mask_path: Path


def _stored_paths(directory: Path) -> StoredClassification:
    return StoredClassification(
        directory=directory,
        classes_path=directory / "classes.tif",
        valid_mask_path=directory / "valid_mask.tif",
    )


def _dataset_matches_grid(dataset, grid: RasterGrid) -> bool:
    expected_crs = CRS.from_user_input(grid.crs)
    actual_epsg = dataset.crs.to_epsg()
    expected_epsg = expected_crs.to_epsg()
    crs_matches = (
        actual_epsg == expected_epsg
        if actual_epsg is not None and expected_epsg is not None
        else dataset.crs == expected_crs
    )
    if not crs_matches:
        transformer = Transformer.from_crs(
            PyprojCRS.from_wkt(expected_crs.to_wkt()),
            PyprojCRS.from_wkt(dataset.crs.to_wkt()),
            always_xy=True,
        )
        sample_pixels = (
            (0, 0),
            (grid.width / 2, grid.height / 2),
            (grid.width, grid.height),
        )
        tolerance = min(abs(grid.transform.a), abs(grid.transform.e)) * 1e-6
        crs_matches = True
        for column, row in sample_pixels:
            x, y = grid.transform * (column, row)
            transformed_x, transformed_y = transformer.transform(x, y)
            if abs(transformed_x - x) > tolerance or abs(transformed_y - y) > tolerance:
                crs_matches = False
                break
    return (
        dataset.count == 1
        and dataset.dtypes == ("uint8",)
        and dataset.width == grid.width
        and dataset.height == grid.height
        and crs_matches
        and dataset.transform.almost_equals(grid.transform)
    )


def stored_classification_validation_error(
    stored: StoredClassification,
    grid: RasterGrid,
    *,
    verify_pixels: bool = True,
) -> str | None:
    try:
        with rasterio.open(stored.classes_path) as classes_ds, rasterio.open(
            stored.valid_mask_path
        ) as valid_ds:
            if not _dataset_matches_grid(classes_ds, grid):
                return (
                    "classes.tif 网格或数据类型不匹配: "
                    f"count={classes_ds.count}, dtype={classes_ds.dtypes}, "
                    f"size={classes_ds.width}x{classes_ds.height}, "
                    f"crs={classes_ds.crs}, "
                    f"expected_crs={CRS.from_user_input(grid.crs).to_string()}, "
                    f"transform_equal={classes_ds.transform.almost_equals(grid.transform)}, "
                    f"transform={classes_ds.transform}, expected_transform={grid.transform}"
                )
            if not _dataset_matches_grid(valid_ds, grid):
                return (
                    "valid_mask.tif 网格或数据类型不匹配: "
                    f"count={valid_ds.count}, dtype={valid_ds.dtypes}, "
                    f"size={valid_ds.width}x{valid_ds.height}, "
                    f"crs={valid_ds.crs}, transform={valid_ds.transform}"
                )
            if classes_ds.block_shapes != [(256, 256)]:
                return f"classes.tif 分块不匹配: {classes_ds.block_shapes}"
            if valid_ds.block_shapes != [(256, 256)]:
                return f"valid_mask.tif 分块不匹配: {valid_ds.block_shapes}"
            if classes_ds.compression.name != "deflate":
                return f"classes.tif 压缩不匹配: {classes_ds.compression.name}"
            if valid_ds.compression.name != "deflate":
                return f"valid_mask.tif 压缩不匹配: {valid_ds.compression.name}"

            if verify_pixels:
                for _, window in classes_ds.block_windows(1):
                    classes = classes_ds.read(1, window=window)
                    valid = valid_ds.read(1, window=window)
                    if np.any(classes > 5):
                        return f"classes.tif 包含非法类别: {int(classes.max())}"
                    if np.any((valid != 0) & (valid != 1)):
                        return f"valid_mask.tif 包含非 0/1 值: {np.unique(valid).tolist()}"
                    if np.any((valid == 0) & (classes != 0)):
                        return "无效区的类别值未规范为 0"
    except (OSError, rasterio.errors.RasterioError) as exc:
        return f"GeoTIFF 不可读: {exc}"
    return None


def validate_stored_classification(
    stored: StoredClassification,
    grid: RasterGrid,
    *,
    verify_pixels: bool = True,
) -> bool:
    return (
        stored_classification_validation_error(
            stored,
            grid,
            verify_pixels=verify_pixels,
        )
        is None
    )


class AtomicClassificationWriter:
    def __init__(self, root: str | Path, result_id: str, grid: RasterGrid):
        self.root = Path(root)
        self.result_id = result_id
        self.grid = grid
        self.final_directory = self.root / result_id
        self.temp_directory = self.root / f".{result_id}.{uuid4().hex}.tmp"
        self._classes = None
        self._valid = None
        self._stored = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.final_directory.exists():
            raise FileExistsError(f"识别结果目录已存在: {self.result_id}")
        self.temp_directory.mkdir()
        profile = {
            "driver": "GTiff",
            "width": self.grid.width,
            "height": self.grid.height,
            "count": 1,
            "dtype": "uint8",
            "crs": self.grid.crs,
            "transform": self.grid.transform,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }
        paths = _stored_paths(self.temp_directory)
        self._classes = rasterio.open(paths.classes_path, "w", **profile)
        self._valid = rasterio.open(paths.valid_mask_path, "w", **profile)
        return self

    def write(
        self,
        window: Window,
        classes: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        if self._classes is None or self._valid is None:
            raise RuntimeError("识别结果写入器未打开")
        expected_shape = (int(window.height), int(window.width))
        if classes.shape != expected_shape or valid_mask.shape != expected_shape:
            raise ValueError("写入数组形状与窗口不一致")
        if np.any(classes < 0) or np.any(classes > 5):
            raise ValueError("类别编号必须在 0-5 之间")
        if np.any((valid_mask != 0) & (valid_mask != 1)):
            raise ValueError("有效掩膜必须只包含 0/1")

        valid = valid_mask.astype(np.uint8, copy=False)
        normalized_classes = np.where(valid == 1, classes, 0).astype(
            np.uint8,
            copy=False,
        )
        self._classes.write(normalized_classes, 1, window=window)
        self._valid.write(valid, 1, window=window)

    def _close_datasets(self) -> None:
        if self._classes is not None:
            self._classes.close()
            self._classes = None
        if self._valid is not None:
            self._valid.close()
            self._valid = None

    def finalize(
        self,
        *,
        before_publish: Callable[[], None] | None = None,
    ) -> StoredClassification:
        if self._stored is not None:
            return self._stored
        self._close_datasets()
        temporary = _stored_paths(self.temp_directory)
        validation_error = stored_classification_validation_error(
            temporary,
            self.grid,
        )
        if validation_error is not None:
            raise RuntimeError(f"识别结果文件校验失败: {validation_error}")
        if before_publish is not None:
            before_publish()
        self.temp_directory.replace(self.final_directory)
        self._stored = _stored_paths(self.final_directory)
        return self._stored

    def __exit__(self, exc_type, exc, traceback):
        self._close_datasets()
        if exc_type is None and self._stored is None:
            self.finalize()
        if self.temp_directory.exists():
            shutil.rmtree(self.temp_directory)
        return False


def write_classification_result(
    root: str | Path,
    result_id: str,
    grid: RasterGrid,
    classes: np.ndarray,
    valid_mask: np.ndarray,
) -> StoredClassification:
    with AtomicClassificationWriter(root, result_id, grid) as writer:
        writer.write(Window(0, 0, grid.width, grid.height), classes, valid_mask)
        return writer.finalize()
