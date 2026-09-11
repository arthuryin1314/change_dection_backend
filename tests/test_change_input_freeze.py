import os
from pathlib import Path

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://user:pass@localhost/test",
)

from services.change_input_freeze import FrozenFile, freeze_file
from services.change_orchestration import _identity


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
