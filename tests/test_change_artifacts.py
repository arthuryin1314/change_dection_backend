from pathlib import Path
import os
from types import SimpleNamespace

import numpy as np
import rasterio
from affine import Affine
from PIL import Image

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://user:pass@localhost/test",
)

from utils.classification_storage import RasterGrid
from utils.transition_matrix import compute_transition_matrix_m2
from utils.change_artifacts import _TRANSITION_COLORS, write_change_artifacts
from router.change_results import _serialize_artifacts


def _write(directory: Path, values: np.ndarray, valid: np.ndarray):
    directory.mkdir()
    grid = RasterGrid(
        width=values.shape[1],
        height=values.shape[0],
        crs="EPSG:3857",
        transform=Affine(10, 0, 0, 0, -10, 40),
    )
    paths = (directory / "classes.tif", directory / "valid.tif")
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": "uint8",
        "crs": grid.crs,
        "transform": grid.transform,
        "tiled": True,
        "blockxsize": 16,
        "blockysize": 16,
    }
    with rasterio.open(paths[0], "w", **profile) as ds:
        ds.write(values, 1)
    with rasterio.open(paths[1], "w", **profile) as ds:
        ds.write(valid, 1)
    return paths


def test_change_artifacts_publish_transition_raster_png_and_actual_legend(tmp_path):
    before = _write(
        tmp_path / "before",
        np.array([[0, 1, 2, 3]], dtype=np.uint8),
        np.array([[1, 1, 1, 0]], dtype=np.uint8),
    )
    after = _write(
        tmp_path / "after",
        np.array([[1, 1, 4, 5]], dtype=np.uint8),
        np.array([[1, 1, 1, 1]], dtype=np.uint8),
    )
    matrix = compute_transition_matrix_m2(*before, *after)

    artifacts = write_change_artifacts(
        before_classes_path=before[0],
        before_valid_mask_path=before[1],
        after_classes_path=after[0],
        after_valid_mask_path=after[1],
        result=matrix,
        storage_root=tmp_path / "artifacts",
        result_id="41",
    )

    assert set(artifacts) == {
        "artifact_version",
        "transition_geotiff",
        "visualization_png",
        "legend",
    }
    transition_path = tmp_path / "artifacts" / "41" / artifacts["transition_geotiff"]["filename"]
    with rasterio.open(transition_path) as dataset:
        assert (dataset.width, dataset.height) == (4, 1)
        assert dataset.dtypes == ("uint8",)
        assert dataset.nodata == 255
        np.testing.assert_array_equal(dataset.read(1), [[1, 255, 16, 255]])

    png_path = tmp_path / "artifacts" / "41" / artifacts["visualization_png"]["filename"]
    with Image.open(png_path) as image:
        assert max(image.size) <= 1600
        assert image.size == (4, 1)

    legend = (tmp_path / "artifacts" / "41" / artifacts["legend"]["filename"]).read_text(encoding="utf-8")
    assert '"classification_scheme_version": "land-cover-6/v1"' in legend
    assert '"nodata_value": 255' in legend
    assert '"code": 1' in legend
    assert '"code": 16' in legend
    assert '"code": 7' not in legend
    assert not list((tmp_path / "artifacts").glob(".41.*.tmp"))


def test_transition_color_table_contains_only_directed_non_diagonal_codes():
    assert len(_TRANSITION_COLORS) == 30
    assert not ({0, 7, 14, 21, 28, 35} & _TRANSITION_COLORS.keys())


def test_change_artifacts_failure_removes_temporary_directory(tmp_path):
    before = _write(
        tmp_path / "before",
        np.array([[0, 1]], dtype=np.uint8),
        np.array([[1, 1]], dtype=np.uint8),
    )
    after = _write(
        tmp_path / "after",
        np.array([[1, 1]], dtype=np.uint8),
        np.array([[1, 1]], dtype=np.uint8),
    )
    matrix = compute_transition_matrix_m2(*before, *after)
    after[1].unlink()
    with np.testing.assert_raises(rasterio.errors.RasterioIOError):
        write_change_artifacts(
            before_classes_path=before[0],
            before_valid_mask_path=before[1],
            after_classes_path=after[0],
            after_valid_mask_path=after[1],
            result=matrix,
            storage_root=tmp_path / "artifacts",
            result_id="42",
        )
    assert not (tmp_path / "artifacts" / "42").exists()
    assert not list((tmp_path / "artifacts").glob(".42.*.tmp"))


def test_change_artifacts_failure_does_not_remove_a_competing_published_directory(tmp_path, monkeypatch):
    before = _write(
        tmp_path / "before",
        np.array([[0, 1]], dtype=np.uint8),
        np.array([[1, 1]], dtype=np.uint8),
    )
    after = _write(
        tmp_path / "after",
        np.array([[1, 1]], dtype=np.uint8),
        np.array([[1, 1]], dtype=np.uint8),
    )
    matrix = compute_transition_matrix_m2(*before, *after)
    final_directory = tmp_path / "artifacts" / "43"
    def competing_replace(path, target):
        final_directory.mkdir(parents=True)
        (final_directory / "winner").write_text("published", encoding="utf-8")
        raise FileExistsError("competing writer won")

    monkeypatch.setattr(Path, "replace", competing_replace)
    with np.testing.assert_raises(FileExistsError):
        write_change_artifacts(
            before_classes_path=before[0],
            before_valid_mask_path=before[1],
            after_classes_path=after[0],
            after_valid_mask_path=after[1],
            result=matrix,
            storage_root=tmp_path / "artifacts",
            result_id="43",
        )
    assert (final_directory / "winner").read_text(encoding="utf-8") == "published"
    assert not list((tmp_path / "artifacts").glob(".43.*.tmp"))


def test_result_details_expose_authenticated_artifact_urls():
    row = SimpleNamespace(
        id=41,
        request_id="90f0bdba-6b7e-4a45-a078-dce12340f6d2",
        artifact_metadata={
            "artifact_version": "change-artifacts-v1",
            "transition_geotiff": {
                "filename": "transition.tif",
                "media_type": "image/tiff",
            },
            "visualization_png": {
                "filename": "visualization.png",
                "media_type": "image/png",
            },
            "legend": {
                "filename": "legend.json",
                "media_type": "application/json",
            },
        },
    )
    artifacts = _serialize_artifacts(row)
    assert artifacts["artifact_version"] == "change-artifacts-v1"
    assert artifacts["legend"]["url"].endswith("/artifacts/legend.json")
