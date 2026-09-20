import os
from pathlib import Path

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://user:pass@localhost/test",
)

from services.change_input_freeze import FrozenFile, freeze_file
from services.change_orchestration import _identity, _resolved_period, _source_snapshot


def test_freeze_file_keeps_submitted_bytes_after_source_is_deleted(tmp_path):
    source = tmp_path / "source.tif"
    source.write_bytes(b"submitted-image-version")

    frozen = freeze_file(source, tmp_path / "task-inputs", "before.tif")
    source.unlink()

    assert frozen == FrozenFile(
        path=tmp_path / "task-inputs" / "before.tif",
        sha256="3db8afed3721f6669e60ad706d390570bc44091cba2af745b1cb045ddb65ef57",
        size=23,
    )
    assert frozen.path.read_bytes() == b"submitted-image-version"


def test_result_identity_uses_the_submitted_contract():
    contract = {
        "inference_parameters": {"tile_size": 256, "overlap": 32},
        "classification_scheme_version": "land-cover-test",
        "pipeline_version": "pipeline-test",
        "grid_policy_version": "grid-test",
    }

    identity = _identity(7, "1" * 64, "2" * 64, contract)

    assert identity.inference_parameters == contract["inference_parameters"]
    assert identity.classification_scheme_version == "land-cover-test"
    assert identity.pipeline_version == "pipeline-test"
    assert identity.grid_policy_version == "grid-test"


def test_classification_source_snapshot_keeps_narrow_shared_contract():
    submitted = {
        "before_image_id": 11,
        "model_id": 22,
        "before": {"name": "变化提交时影像名"},
        "model": {"name": "变化提交时模型名"},
        "snapshot_metadata": {
            "before": {
                "name": "变化提交时影像名",
                "capture_date": "2024-05-01",
                "satellite": "Sentinel-2",
                "resolution": "10.0000",
            },
            "model": {"name": "变化提交时模型名"},
        },
    }

    assert _source_snapshot(submitted, "before") == {
        "image": {"id": 11, "name": "变化提交时影像名"},
        "model": {"id": 22, "name": "变化提交时模型名"},
    }


def test_reused_classification_keeps_names_from_change_submission():
    from datetime import datetime, timezone
    from types import SimpleNamespace

    completed_at = datetime.now(timezone.utc)
    row = SimpleNamespace(
        id="classification-result",
        identity_sha256="3" * 64,
        source_image_id=11,
        source_model_id=22,
        resolution=[10.0, 10.0],
        crs="EPSG:4528",
        raster_width=100,
        raster_height=80,
        source_snapshot={
            "image": {"id": 11, "name": "识别时旧影像名"},
            "model": {"id": 22, "name": "识别时旧模型名"},
        },
        image_content_sha256="1" * 64,
        weight_content_sha256="2" * 64,
        inference_parameters={"tile_size": 256, "overlap": 32},
        classification_scheme_version="land-cover-test",
        pipeline_version="pipeline-test",
        grid_policy_version="grid-test",
        completed_at=completed_at,
        classes_path="classes.tif",
        valid_mask_path="valid-mask.tif",
        area_status="SUCCEEDED",
        class_area_m2=[1, 2, 3, 4, 5, 6],
        area_completed_at=completed_at,
        area_failure_detail=None,
    )
    submitted = {
        "before_image_id": 11,
        "model_id": 22,
        "before": {"name": "变化提交时影像名"},
        "model": {"name": "变化提交时模型名"},
        "snapshot_metadata": {
            "before": {
                "name": "变化提交时影像名",
                "capture_date": "2024-05-01",
                "satellite": "Sentinel-2",
                "resolution": "10.0000",
            },
            "model": {"name": "变化提交时模型名"},
        },
    }

    period = _resolved_period(row, submitted, "before")

    assert period.snapshot["source"] == {
        "image": {
            "id": 11,
            "name": "变化提交时影像名",
            "capture_date": "2024-05-01",
            "satellite": "Sentinel-2",
            "resolution": "10.0000",
            "raster_resolution": [10.0, 10.0],
            "crs": "EPSG:4528",
            "width": 100,
            "height": 80,
        },
        "model": {"id": 22, "name": "变化提交时模型名"},
    }


def test_legacy_submitted_inputs_without_snapshot_metadata_are_safe():
    from datetime import datetime, timezone
    from types import SimpleNamespace

    completed_at = datetime.now(timezone.utc)
    row = SimpleNamespace(
        id="classification-result",
        identity_sha256="3" * 64,
        source_image_id=11,
        source_model_id=22,
        source_snapshot=None,
        image_content_sha256="1" * 64,
        weight_content_sha256="2" * 64,
        inference_parameters={},
        classification_scheme_version="land-cover-test",
        pipeline_version="pipeline-test",
        grid_policy_version="grid-test",
        completed_at=completed_at,
        classes_path="classes.tif",
        valid_mask_path="valid-mask.tif",
        area_status="SUCCEEDED",
        class_area_m2=[1, 2, 3, 4, 5, 6],
        area_completed_at=completed_at,
        area_failure_detail=None,
        resolution=None,
        crs="EPSG:4528",
        raster_width=100,
        raster_height=80,
    )

    period = _resolved_period(
        row,
        {
            "before_image_id": 11,
            "model_id": 22,
            "before": {"name": "legacy"},
            "model": {"name": "legacy"},
        },
        "before",
    )

    image = period.snapshot["source"]["image"]
    assert image["name"] is None
    assert image["capture_date"] is None
    assert image["satellite"] is None
    assert image["resolution"] is None
    assert image["raster_resolution"] is None
    assert image["crs"] == "EPSG:4528"
    assert image["width"] == 100
    assert image["height"] == 80
