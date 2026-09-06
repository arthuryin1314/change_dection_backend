from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.windows import Window

from utils.classification_storage import (
    AtomicClassificationWriter,
    RasterGrid,
    StoredClassification,
    validate_stored_classification,
    write_classification_result,
)


GRID = RasterGrid(
    width=3,
    height=2,
    crs="EPSG:4528",
    transform=Affine(0.8, 0, 500000, 0, -0.8, 3200000),
)
CLASSES = np.array([[0, 1, 5], [3, 4, 2]], dtype=np.uint8)
VALID = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.uint8)
EXPECTED_CLASSES = np.array([[0, 1, 0], [3, 0, 2]], dtype=np.uint8)


def test_atomic_write_publishes_two_lossless_matching_geotiffs(tmp_path):
    stored = write_classification_result(
        tmp_path,
        "result-1",
        GRID,
        CLASSES,
        VALID,
    )

    assert stored == StoredClassification(
        directory=tmp_path / "result-1",
        classes_path=tmp_path / "result-1" / "classes.tif",
        valid_mask_path=tmp_path / "result-1" / "valid_mask.tif",
    )
    assert validate_stored_classification(stored, GRID) is True
    with rasterio.open(stored.classes_path) as classes_ds:
        assert np.array_equal(classes_ds.read(1), EXPECTED_CLASSES)
        assert classes_ds.dtypes == ("uint8",)
        assert classes_ds.compression.name == "deflate"
        assert classes_ds.is_tiled
        assert classes_ds.block_shapes == [(256, 256)]
    with rasterio.open(stored.valid_mask_path) as valid_ds:
        assert np.array_equal(valid_ds.read(1), VALID)
        assert valid_ds.crs == rasterio.crs.CRS.from_epsg(4528)
        assert valid_ds.transform == GRID.transform


def test_writer_failure_never_exposes_half_a_result(tmp_path):
    with pytest.raises(RuntimeError, match="inference failed"):
        with AtomicClassificationWriter(tmp_path, "result-2", GRID) as writer:
            writer.write(Window(0, 0, 3, 1), CLASSES[:1], VALID[:1])
            raise RuntimeError("inference failed")

    assert not (tmp_path / "result-2").exists()
    assert list(tmp_path.glob(".result-2.*.tmp")) == []


def test_writer_rejects_invalid_class_values_before_publish(tmp_path):
    invalid = CLASSES.copy()
    invalid[0, 0] = 6

    with pytest.raises(ValueError, match="0-5"):
        write_classification_result(tmp_path, "result-3", GRID, invalid, VALID)

    assert not (tmp_path / "result-3").exists()


def test_writer_rejects_non_binary_valid_mask_before_publish(tmp_path):
    invalid = VALID.copy()
    invalid[0, 0] = 2

    with pytest.raises(ValueError, match="0/1"):
        write_classification_result(tmp_path, "result-4", GRID, CLASSES, invalid)

    assert not (tmp_path / "result-4").exists()


def test_only_finalize_makes_result_visible(tmp_path):
    with AtomicClassificationWriter(tmp_path, "result-5", GRID) as writer:
        writer.write(Window(0, 0, 3, 2), CLASSES, VALID)
        assert not (tmp_path / "result-5").exists()
        stored = writer.finalize()
        assert stored.directory.exists()

    assert validate_stored_classification(stored, GRID)


def test_publish_guard_runs_after_validation_and_before_atomic_rename(tmp_path):
    def reject_publish():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        with AtomicClassificationWriter(tmp_path, "result-6", GRID) as writer:
            writer.write(Window(0, 0, 3, 2), CLASSES, VALID)
            writer.finalize(before_publish=reject_publish)

    assert not (tmp_path / "result-6").exists()
    assert list(tmp_path.glob(".result-6.*.tmp")) == []
