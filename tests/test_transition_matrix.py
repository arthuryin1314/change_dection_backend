from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from utils.transition_matrix import (
    AREA_GRID_UNSUPPORTED,
    GRID_MISMATCH,
    NO_COMMON_VALID_AREA,
    TransitionMatrixError,
    compute_transition_matrix_m2,
)


GEOGRAPHIC_PIXEL_M2 = 1069626.783188343


def _write_pair(
    directory: Path,
    classes: np.ndarray,
    valid: np.ndarray,
    *,
    transform: Affine,
    crs: str = "EPSG:4326",
):
    directory.mkdir()
    profile = {
        "driver": "GTiff",
        "width": classes.shape[1],
        "height": classes.shape[0],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 16,
        "blockysize": 16,
    }
    classes_path = directory / "classes.tif"
    valid_path = directory / "valid_mask.tif"
    with rasterio.open(classes_path, "w", **profile) as dataset:
        dataset.write(classes, 1)
    with rasterio.open(valid_path, "w", **profile) as dataset:
        dataset.write(valid, 1)
    return classes_path, valid_path


def test_transition_matrix_has_all_before_rows_and_after_columns(tmp_path):
    before_classes = np.repeat(np.arange(6, dtype=np.uint8), 6)[None, :]
    after_classes = np.tile(np.arange(6, dtype=np.uint8), 6)[None, :]
    valid = np.ones_like(before_classes, dtype=np.uint8)
    transform = Affine(0.01, 0, 110, 0, -0.01, 30)
    before = _write_pair(tmp_path / "before", before_classes, valid, transform=transform)
    after = _write_pair(tmp_path / "after", after_classes, valid, transform=transform)

    result = compute_transition_matrix_m2(*before, *after)

    expected = np.full((6, 6), GEOGRAPHIC_PIXEL_M2)
    assert np.asarray(result.matrix_m2) == pytest.approx(expected, rel=1e-10, abs=1e-5)
    assert result.common_valid_area_m2 == pytest.approx(
        36 * GEOGRAPHIC_PIXEL_M2,
        rel=1e-10,
        abs=1e-5,
    )
    assert result.grid.width == 36
    assert result.grid.height == 1


def test_transition_matrix_uses_spatial_overlap_and_both_valid_masks(tmp_path):
    before = _write_pair(
        tmp_path / "before",
        np.array([[5, 4, 0, 2]], dtype=np.uint8),
        np.array([[1, 1, 1, 1]], dtype=np.uint8),
        transform=Affine(0.01, 0, 110, 0, -0.01, 30),
    )
    after = _write_pair(
        tmp_path / "after",
        np.array([[1, 3, 4, 5]], dtype=np.uint8),
        np.array([[1, 0, 1, 1]], dtype=np.uint8),
        transform=Affine(0.01, 0, 110.02, 0, -0.01, 30),
    )

    result = compute_transition_matrix_m2(*before, *after)

    expected = np.zeros((6, 6))
    expected[0, 1] = GEOGRAPHIC_PIXEL_M2
    assert np.asarray(result.matrix_m2) == pytest.approx(expected, rel=1e-10, abs=1e-5)
    assert result.common_valid_area_m2 == pytest.approx(
        GEOGRAPHIC_PIXEL_M2,
        rel=1e-10,
        abs=1e-5,
    )
    assert result.grid.width == 2
    assert result.grid.transform.c == pytest.approx(110.02)


def test_transition_matrix_rejects_half_pixel_origin_offset(tmp_path):
    values = np.zeros((1, 2), dtype=np.uint8)
    valid = np.ones_like(values)
    before = _write_pair(
        tmp_path / "before",
        values,
        valid,
        transform=Affine(0.01, 0, 110, 0, -0.01, 30),
    )
    after = _write_pair(
        tmp_path / "after",
        values,
        valid,
        transform=Affine(0.01, 0, 110.005, 0, -0.01, 30),
    )

    with pytest.raises(TransitionMatrixError) as caught:
        compute_transition_matrix_m2(*before, *after)

    assert caught.value.error_code == GRID_MISMATCH


def test_transition_matrix_rejects_empty_common_valid_area(tmp_path):
    values = np.zeros((1, 2), dtype=np.uint8)
    transform = Affine(0.01, 0, 110, 0, -0.01, 30)
    before = _write_pair(
        tmp_path / "before",
        values,
        np.array([[1, 0]], dtype=np.uint8),
        transform=transform,
    )
    after = _write_pair(
        tmp_path / "after",
        values,
        np.array([[0, 1]], dtype=np.uint8),
        transform=transform,
    )

    with pytest.raises(TransitionMatrixError) as caught:
        compute_transition_matrix_m2(*before, *after)

    assert caught.value.error_code == NO_COMMON_VALID_AREA


def test_transition_matrix_keeps_area_grid_failure_distinct_from_grid_mismatch(
    tmp_path,
):
    values = np.zeros((1, 2), dtype=np.uint8)
    valid = np.ones_like(values)
    transform = Affine(0.01, 0.001, 110, 0, -0.01, 30)
    before = _write_pair(tmp_path / "before", values, valid, transform=transform)
    after = _write_pair(tmp_path / "after", values, valid, transform=transform)

    with pytest.raises(TransitionMatrixError) as caught:
        compute_transition_matrix_m2(*before, *after)

    assert caught.value.error_code == AREA_GRID_UNSUPPORTED


def test_transition_matrix_handles_non_window_aligned_spatial_offset(tmp_path):
    before_values = np.zeros((300, 300), dtype=np.uint8)
    after_values = np.ones((300, 300), dtype=np.uint8)
    valid = np.ones_like(before_values)
    before = _write_pair(
        tmp_path / "before-offset",
        before_values,
        valid,
        transform=Affine(10, 0, 500000, 0, -10, 3000000),
        crs="EPSG:32649",
    )
    after = _write_pair(
        tmp_path / "after-offset",
        after_values,
        valid,
        transform=Affine(10, 0, 500170, 0, -10, 2999890),
        crs="EPSG:32649",
    )

    result = compute_transition_matrix_m2(*before, *after)

    assert result.before_window == [17, 11, 283, 289]
    assert result.after_window == [0, 0, 283, 289]
    assert result.matrix_m2[0][1] == result.common_valid_area_m2
    assert result.matrix_m2[0][1] > 0
    assert np.count_nonzero(result.matrix_m2) == 1


def test_transition_matrix_accepts_semantically_equal_crs_encodings(tmp_path):
    values = np.zeros((1, 2), dtype=np.uint8)
    valid = np.ones_like(values)
    transform = Affine(0.01, 0, 110, 0, -0.01, 30)
    before = _write_pair(
        tmp_path / "before-crs",
        values,
        valid,
        transform=transform,
        crs="EPSG:4326",
    )
    wkt = rasterio.crs.CRS.from_epsg(4326).to_wkt()
    after = _write_pair(
        tmp_path / "after-crs",
        values,
        valid,
        transform=transform,
        crs=wkt,
    )

    result = compute_transition_matrix_m2(*before, *after)

    assert result.common_valid_area_m2 == pytest.approx(2 * GEOGRAPHIC_PIXEL_M2, rel=1e-10, abs=1e-5)
