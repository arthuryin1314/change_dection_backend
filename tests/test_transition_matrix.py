from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import Transformer

from utils.transition_matrix import (
    NO_COMMON_VALID_AREA,
    TransitionMatrixError,
    compute_transition_matrix_m2,
)


GEOGRAPHIC_PIXEL_M2 = 1069626.783188343
LOCAL_CRS_WKT = (
    'LOCAL_CS["arbitrary",LOCAL_DATUM["unknown",0],UNIT["metre",1],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
)


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
    assert result.analysis["alignment_mode"] == "DIRECT"


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


def test_transition_matrix_aligns_different_resolution_with_nearest_neighbor(tmp_path):
    before = _write_pair(
        tmp_path / "before",
        np.array([[0, 1], [2, 3]], dtype=np.uint8),
        np.ones((2, 2), dtype=np.uint8),
        transform=Affine(2, 0, 500000, 0, -2, 3000000),
        crs="EPSG:32649",
    )
    after = _write_pair(
        tmp_path / "after",
        np.array([[5, 5, 4, 4], [5, 5, 4, 4], [3, 3, 2, 2], [3, 3, 2, 2]], dtype=np.uint8),
        np.ones((4, 4), dtype=np.uint8),
        transform=Affine(1, 0, 500000, 0, -1, 3000000),
        crs="EPSG:32649",
    )

    result = compute_transition_matrix_m2(*before, *after)

    assert result.analysis == {
        "alignment_mode": "WARPED",
        "reference_period": "before",
        "resolution": [2.0, 2.0],
        "resampling": "nearest",
        "coordinate_transform_tolerance": 0.0,
        "alignment_tolerance_pixels": 1e-6,
        "pixel_size_rtol": 1e-9,
        "edge_rule": "target_pixel_center_full_cell",
    }
    assert result.grid.width == 2
    assert result.grid.height == 2
    assert np.flatnonzero(np.asarray(result.matrix_m2)).tolist() == [5, 10, 15, 20]


def test_transition_matrix_aligns_half_pixel_origin_offset(tmp_path):
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

    result = compute_transition_matrix_m2(*before, *after)

    assert result.analysis["alignment_mode"] == "WARPED"


def test_nearly_equal_resolution_with_cumulative_drift_uses_alignment(tmp_path):
    values = np.zeros((1, 2000), dtype=np.uint8)
    valid = np.ones_like(values)
    before = _write_pair(
        tmp_path / "before-near-resolution",
        values,
        valid,
        transform=Affine(1, 0, 500000, 0, -1, 3000000),
        crs="EPSG:32649",
    )
    after = _write_pair(
        tmp_path / "after-near-resolution",
        values,
        valid,
        transform=Affine(1.0000000009, 0, 500000, 0, -1, 3000000),
        crs="EPSG:32649",
    )

    result = compute_transition_matrix_m2(*before, *after)

    assert result.analysis["alignment_mode"] == "WARPED"
    assert result.common_valid_area_m2 > 0


def test_direct_tolerance_uses_an_order_independent_analysis_grid(tmp_path):
    values = np.zeros((1, 1000), dtype=np.uint8)
    valid = np.ones_like(values)
    exact = _write_pair(
        tmp_path / "exact-resolution",
        values,
        valid,
        transform=Affine(1, 0, 500000, 0, -1, 3000000),
        crs="EPSG:32649",
    )
    noisy = _write_pair(
        tmp_path / "noisy-resolution",
        values,
        valid,
        transform=Affine(1.0000000009, 0, 500000, 0, -1, 3000000),
        crs="EPSG:32649",
    )

    forward = compute_transition_matrix_m2(*exact, *noisy)
    reverse = compute_transition_matrix_m2(*noisy, *exact)

    assert forward.analysis["alignment_mode"] == "DIRECT"
    assert reverse.analysis["alignment_mode"] == "DIRECT"
    assert forward.grid == reverse.grid
    assert np.array_equal(np.asarray(forward.matrix_m2), np.asarray(reverse.matrix_m2).T)


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


def test_cross_crs_uses_a_north_up_grid_and_fixed_area_reference(tmp_path):
    before = _write_pair(
        tmp_path / "before-geographic",
        np.array([[0, 1]], dtype=np.uint8),
        np.ones((1, 2), dtype=np.uint8),
        transform=Affine(0.01, 0, 110, 0, -0.01, 30),
    )
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    left, top = to_utm.transform(109.995, 30.005)
    after = _write_pair(
        tmp_path / "after-utm",
        np.full((5, 9), 2, dtype=np.uint8),
        np.ones((5, 9), dtype=np.uint8),
        transform=Affine(300, 0, left, 0, -300, top),
        crs="EPSG:32649",
    )

    result = compute_transition_matrix_m2(*before, *after)

    assert str(result.grid.crs) == "EPSG:4326"
    assert result.grid.transform.b == 0
    assert result.grid.transform.d == 0
    assert result.matrix_m2[0][2] == pytest.approx(GEOGRAPHIC_PIXEL_M2, rel=1e-10, abs=1e-5)
    assert result.matrix_m2[1][2] == pytest.approx(GEOGRAPHIC_PIXEL_M2, rel=1e-10, abs=1e-5)
    assert np.count_nonzero(result.matrix_m2) == 2


def test_swapping_periods_keeps_grid_and_transposes_matrix(tmp_path):
    coarse = _write_pair(
        tmp_path / "coarse",
        np.array([[0, 1], [2, 3]], dtype=np.uint8),
        np.ones((2, 2), dtype=np.uint8),
        transform=Affine(2, 0, 500000, 0, -2, 3000000),
        crs="EPSG:32649",
    )
    fine = _write_pair(
        tmp_path / "fine",
        np.array([[5, 5, 4, 4], [5, 5, 4, 4], [3, 3, 2, 2], [3, 3, 2, 2]], dtype=np.uint8),
        np.ones((4, 4), dtype=np.uint8),
        transform=Affine(1, 0, 500000, 0, -1, 3000000),
        crs="EPSG:32649",
    )

    forward = compute_transition_matrix_m2(*coarse, *fine)
    reverse = compute_transition_matrix_m2(*fine, *coarse)

    assert forward.grid == reverse.grid
    assert np.array_equal(np.asarray(forward.matrix_m2), np.asarray(reverse.matrix_m2).T)
    assert forward.common_valid_area_m2 == pytest.approx(reverse.common_valid_area_m2, rel=1e-12)


def test_equal_area_anisotropic_inputs_use_per_axis_coarser_resolution(tmp_path):
    horizontal = _write_pair(
        tmp_path / "horizontal",
        np.zeros((2, 4), dtype=np.uint8),
        np.ones((2, 4), dtype=np.uint8),
        transform=Affine(1, 0, 500000, 0, -4, 3000000),
        crs="EPSG:32649",
    )
    square = _write_pair(
        tmp_path / "square",
        np.ones((4, 2), dtype=np.uint8),
        np.ones((4, 2), dtype=np.uint8),
        transform=Affine(2, 0, 500000, 0, -2, 3000000),
        crs="EPSG:32649",
    )

    forward = compute_transition_matrix_m2(*horizontal, *square)
    reverse = compute_transition_matrix_m2(*square, *horizontal)

    assert forward.analysis["resolution"] == [2.0, 4.0]
    assert forward.grid == reverse.grid
    assert np.array_equal(np.asarray(forward.matrix_m2), np.asarray(reverse.matrix_m2).T)


def test_aligned_result_is_independent_of_target_window_size(tmp_path):
    coarse_values = np.indices((105, 105)).sum(axis=0).astype(np.uint8) % 6
    fine_values = np.repeat(np.repeat(coarse_values, 2, axis=0), 2, axis=1)
    valid = np.ones_like(coarse_values, dtype=np.uint8)
    valid[40:70, 30:80] = 0
    coarse = _write_pair(
        tmp_path / "coarse-window",
        coarse_values,
        valid,
        transform=Affine(2, 0, 500000, 0, -2, 3000000),
        crs="EPSG:32649",
    )
    fine = _write_pair(
        tmp_path / "fine-window",
        fine_values,
        np.repeat(np.repeat(valid, 2, axis=0), 2, axis=1),
        transform=Affine(1, 0, 500000, 0, -1, 3000000),
        crs="EPSG:32649",
    )

    results = [
        compute_transition_matrix_m2(*coarse, *fine, window_size=size)
        for size in (256, 100, 1000)
    ]

    assert results[0].matrix_m2 == results[1].matrix_m2 == results[2].matrix_m2
    assert results[0].common_valid_area_m2 == results[1].common_valid_area_m2 == results[2].common_valid_area_m2


@pytest.mark.parametrize(("after_left", "succeeds"), [(1.9, False), (0.9, True)])
def test_subpixel_overlap_depends_on_the_presence_of_a_common_valid_center(
    tmp_path,
    after_left,
    succeeds,
):
    before = _write_pair(
        tmp_path / "before-narrow",
        np.zeros((1, 1), dtype=np.uint8),
        np.ones((1, 1), dtype=np.uint8),
        transform=Affine(2, 0, 0, 0, -2, 2),
        crs="EPSG:32649",
    )
    after = _write_pair(
        tmp_path / "after-narrow",
        np.ones((1, 1), dtype=np.uint8),
        np.ones((1, 1), dtype=np.uint8),
        transform=Affine(2, 0, after_left, 0, -2, 2),
        crs="EPSG:32649",
    )

    if succeeds:
        assert compute_transition_matrix_m2(*before, *after).matrix_m2[0][1] > 0
    else:
        with pytest.raises(TransitionMatrixError) as caught:
            compute_transition_matrix_m2(*before, *after)
        assert caught.value.error_code == NO_COMMON_VALID_AREA


def test_invalid_selected_classification_values_are_a_stable_contract_error(tmp_path):
    values = np.array([[6]], dtype=np.uint8)
    valid = np.ones_like(values)
    transform = Affine(1, 0, 500000, 0, -1, 3000000)
    before = _write_pair(tmp_path / "before-invalid", values, valid, transform=transform, crs="EPSG:32649")
    after = _write_pair(tmp_path / "after-invalid", values, valid, transform=transform, crs="EPSG:32649")

    with pytest.raises(TransitionMatrixError) as caught:
        compute_transition_matrix_m2(*before, *after)

    assert caught.value.error_code == "RESULT_CONTRACT_BROKEN"


def test_classes_and_mask_grid_mismatch_is_a_stable_contract_error(tmp_path):
    values = np.zeros((1, 2), dtype=np.uint8)
    valid = np.ones_like(values)
    before = _write_pair(
        tmp_path / "before-broken",
        values,
        valid,
        transform=Affine(1, 0, 500000, 0, -1, 3000000),
        crs="EPSG:32649",
    )
    after = _write_pair(
        tmp_path / "after-broken",
        values,
        valid,
        transform=Affine(1, 0, 500000, 0, -1, 3000000),
        crs="EPSG:32649",
    )
    with rasterio.open(before[1], "r+") as dataset:
        dataset.transform = Affine(1, 0, 500001, 0, -1, 3000000)

    with pytest.raises(TransitionMatrixError) as caught:
        compute_transition_matrix_m2(*before, *after)

    assert caught.value.error_code == "RESULT_CONTRACT_BROKEN"


def test_missing_spatial_reference_has_a_stable_error(tmp_path):
    values = np.zeros((1, 2), dtype=np.uint8)
    valid = np.ones_like(values)
    before = _write_pair(
        tmp_path / "before-no-crs",
        values,
        valid,
        transform=Affine(1, 0, 0, 0, -1, 1),
        crs=None,
    )
    after = _write_pair(
        tmp_path / "after-with-crs",
        values,
        valid,
        transform=Affine(1, 0, 0, 0, -1, 1),
        crs="EPSG:32649",
    )

    with pytest.raises(TransitionMatrixError) as caught:
        compute_transition_matrix_m2(*before, *after)

    assert caught.value.error_code == "SPATIAL_METADATA_UNAVAILABLE"


def test_engineering_crs_without_a_geodetic_basis_has_a_stable_error(tmp_path):
    values = np.zeros((1, 2), dtype=np.uint8)
    valid = np.ones_like(values)
    before = _write_pair(
        tmp_path / "before-local-crs",
        values,
        valid,
        transform=Affine(2, 0, 0, 0, -2, 2),
        crs=LOCAL_CRS_WKT,
    )
    after = _write_pair(
        tmp_path / "after-local-crs",
        values,
        valid,
        transform=Affine(1, 0, 0, 0, -1, 1),
        crs=LOCAL_CRS_WKT,
    )

    with pytest.raises(TransitionMatrixError) as caught:
        compute_transition_matrix_m2(*before, *after)

    assert caught.value.error_code == "SPATIAL_METADATA_UNAVAILABLE"


def test_singular_spatial_transform_has_a_stable_error(tmp_path):
    values = np.zeros((1, 2), dtype=np.uint8)
    valid = np.ones_like(values)
    singular = Affine(1, 2, 500000, 0.5, 1, 3000000)
    before = _write_pair(
        tmp_path / "before-singular",
        values,
        valid,
        transform=singular,
        crs="EPSG:32649",
    )
    after = _write_pair(
        tmp_path / "after-valid",
        values,
        valid,
        transform=Affine(1, 0, 500000, 0, -1, 3000000),
        crs="EPSG:32649",
    )

    with pytest.raises(TransitionMatrixError) as caught:
        compute_transition_matrix_m2(*before, *after)

    assert caught.value.error_code == "SPATIAL_METADATA_UNAVAILABLE"
