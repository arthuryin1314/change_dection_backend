import numpy as np
import pytest
import rasterio
from affine import Affine

from utils.classification_area import (
    summarize_classification_area_m2,
    validate_class_area_m2,
)
from utils.classification_storage import RasterGrid, write_classification_result


GEOGRAPHIC_PIXEL_M2 = 1069626.783188343


def _write_result(tmp_path, classes, valid_mask, *, result_id="result-1"):
    grid = RasterGrid(
        width=classes.shape[1],
        height=classes.shape[0],
        crs="EPSG:4326",
        transform=Affine(0.01, 0, 110, 0, -0.01, 30),
    )
    return write_classification_result(
        tmp_path,
        result_id,
        grid,
        classes,
        valid_mask,
    )


def test_geographic_summary_keeps_valid_background_and_excludes_nodata(tmp_path):
    stored = _write_result(
        tmp_path,
        np.array([[0, 0, 1, 2, 3, 4, 5, 5, 0]], dtype=np.uint8),
        np.array([[1, 0, 1, 1, 1, 1, 1, 0, 0]], dtype=np.uint8),
    )

    areas = summarize_classification_area_m2(
        stored.classes_path,
        stored.valid_mask_path,
    )

    assert areas == pytest.approx([GEOGRAPHIC_PIXEL_M2] * 6, rel=1e-10, abs=1e-5)


def test_native_block_edge_is_counted_once(tmp_path):
    classes = np.zeros((1, 257), dtype=np.uint8)
    valid = np.ones((1, 257), dtype=np.uint8)
    valid[0, -1] = 0
    stored = _write_result(tmp_path, classes, valid)

    areas = summarize_classification_area_m2(
        stored.classes_path,
        stored.valid_mask_path,
    )

    assert areas[0] == pytest.approx(256 * GEOGRAPHIC_PIXEL_M2, rel=1e-10)
    assert areas[1:] == [0.0] * 5


def test_summary_rejects_misaligned_mask(tmp_path):
    stored = _write_result(
        tmp_path,
        np.array([[0, 1]], dtype=np.uint8),
        np.array([[1, 1]], dtype=np.uint8),
    )
    with rasterio.open(stored.valid_mask_path, "r+") as dataset:
        dataset.transform = Affine(0.01, 0, 111, 0, -0.01, 30)

    with pytest.raises(ValueError, match="网格"):
        summarize_classification_area_m2(
            stored.classes_path,
            stored.valid_mask_path,
        )


@pytest.mark.parametrize(
    "areas",
    (
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, float("nan")],
        [1, 2, 3, 4, 5, -1],
        [1, 2, 3, 4, 5, "6"],
    ),
)
def test_persisted_area_requires_six_finite_nonnegative_numbers(areas):
    with pytest.raises(ValueError, match="六类.*有限非负"):
        validate_class_area_m2(areas)


def _write_tiled_pair(directory, classes, valid_mask, block_size):
    directory.mkdir()
    profile = {
        "driver": "GTiff",
        "width": classes.shape[1],
        "height": classes.shape[0],
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": Affine(0.01, 0, 110, 0, -0.01, 30),
        "tiled": True,
        "blockxsize": block_size,
        "blockysize": block_size,
    }
    classes_path = directory / "classes.tif"
    mask_path = directory / "valid_mask.tif"
    with rasterio.open(classes_path, "w", **profile) as dataset:
        dataset.write(classes, 1)
    with rasterio.open(mask_path, "w", **profile) as dataset:
        dataset.write(valid_mask, 1)
    return classes_path, mask_path


def test_summary_is_bitwise_identical_for_different_native_block_sizes(tmp_path):
    classes = (np.arange(32 * 48, dtype=np.uint16).reshape(32, 48) % 6).astype(
        np.uint8
    )
    valid = np.ones((32, 48), dtype=np.uint8)
    valid[::5, ::7] = 0
    small_blocks = _write_tiled_pair(tmp_path / "small", classes, valid, 16)
    large_blocks = _write_tiled_pair(tmp_path / "large", classes, valid, 32)

    assert summarize_classification_area_m2(
        *small_blocks
    ) == summarize_classification_area_m2(*large_blocks)
