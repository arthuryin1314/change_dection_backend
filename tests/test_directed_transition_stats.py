from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from utils.transition_matrix import compute_transition_matrix_m2


def _write_pair(directory: Path, classes: np.ndarray, valid: np.ndarray):
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


def test_directed_transition_stats_use_36_slots_and_keep_nodata_separate(tmp_path):
    before = _write_pair(
        tmp_path / "before",
        np.array([[0, 1, 2, 3]], dtype=np.uint8),
        np.array([[1, 1, 1, 0]], dtype=np.uint8),
    )
    after = _write_pair(
        tmp_path / "after",
        np.array([[1, 1, 2, 4]], dtype=np.uint8),
        np.array([[1, 1, 1, 1]], dtype=np.uint8),
    )

    result = compute_transition_matrix_m2(*before, *after)

    assert len(result.transition_pixel_counts) == 36
    assert len(result.transition_area_m2) == 36
    assert result.transition_pixel_counts[0 * 6 + 1] == 1
    assert result.transition_pixel_counts[1 * 6 + 1] == 0
    assert result.transition_pixel_counts[2 * 6 + 2] == 0
    assert result.transition_pixel_counts[3 * 6 + 4] == 0
    assert result.unchanged_pixel_count == 2
    assert result.nodata_pixel_count == 1
    assert sum(result.transition_pixel_counts) == 1
    assert result.nodata_value == 255
    assert result.classification_scheme_version == "land-cover-6/v1"
