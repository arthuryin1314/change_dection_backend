import io
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from affine import Affine

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from config.db_config import get_db
from router import identification_results
from services.identification_results import (
    PROCESSING,
    SUCCEEDED,
    IdentificationResultRecord,
    ResultClaim,
)
from utils.classification_storage import RasterGrid, write_classification_result
from utils.get_user_by_token import get_current_user


class FakeDatabase:
    def __init__(self):
        self.committed = False
        self.flushed = False

    async def flush(self):
        self.flushed = True

    async def commit(self):
        self.committed = True


def _make_client(db=None):
    database = db or FakeDatabase()

    async def override_db():
        yield database

    app = FastAPI()
    app.include_router(identification_results.router)
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)
    return TestClient(app), database


def _sources():
    weight_path = Path(__file__).parents[1] / "weights" / "best_epoch_weights.pth"
    image = SimpleNamespace(
        id=1,
        user_id=7,
        img_path=__file__,
        content_sha256="a" * 64,
        content_sha256_size=os.path.getsize(__file__),
        content_sha256_mtime_ns=os.stat(__file__).st_mtime_ns,
    )
    model = SimpleNamespace(
        id=11,
        user_id=7,
        model_type="semantic_segmentation",
        framework="PyTorch",
        weight_file_path=str(weight_path),
        weight_content_sha256="b" * 64,
        weight_content_sha256_size=weight_path.stat().st_size,
        weight_content_sha256_mtime_ns=weight_path.stat().st_mtime_ns,
    )
    return image, model


def _claim(status=PROCESSING, should_start=True):
    now = datetime(2026, 9, 6, tzinfo=timezone.utc)
    return ResultClaim(
        IdentificationResultRecord(
            result_id="result-1",
            user_id=7,
            identity_sha256="c" * 64,
            status=status,
            lease_owner="worker",
            started_at=now,
            heartbeat_at=now,
            lease_expires_at=now,
        ),
        should_start=should_start,
    )


def test_post_returns_202_and_schedules_only_after_database_commit():
    client, db = _make_client()
    image, model = _sources()
    scheduled = Mock(side_effect=lambda request, owner: setattr(db, "scheduled_after_commit", db.committed))

    with patch.object(
        identification_results.crud_images,
        "get_image_by_id",
        new=AsyncMock(return_value=image),
    ), patch.object(
        identification_results.crud_ml_models,
        "get_ml_model_by_id",
        new=AsyncMock(return_value=model),
    ), patch.object(
        identification_results,
        "claim_identification_result",
        new=AsyncMock(return_value=_claim()),
    ), patch.object(identification_results, "_schedule_generation", new=scheduled):
        response = client.post(
            "/api/identification-results",
            json={"image_id": 1, "model_id": 11},
        )

    assert response.status_code == 202
    assert response.json()["data"] == {
        "result_id": "result-1",
        "status": "PROCESSING",
    }
    assert db.scheduled_after_commit is True
    scheduled.assert_called_once()


def test_old_generation_callback_does_not_unregister_replacement_task():
    old_task = Mock()
    old_task.cancelled.return_value = False
    old_task.exception.return_value = None
    replacement_task = Mock()
    identification_results._generation_tasks["result-1"] = {
        old_task,
        replacement_task,
    }
    try:
        identification_results._generation_done("result-1", old_task)

        assert identification_results._generation_tasks["result-1"] == {
            replacement_task
        }
    finally:
        identification_results._generation_tasks.pop("result-1", None)


def test_post_reuses_succeeded_identity_without_scheduling():
    client, _ = _make_client()
    image, model = _sources()

    with patch.object(
        identification_results.crud_images,
        "get_image_by_id",
        new=AsyncMock(return_value=image),
    ), patch.object(
        identification_results.crud_ml_models,
        "get_ml_model_by_id",
        new=AsyncMock(return_value=model),
    ), patch.object(
        identification_results,
        "claim_identification_result",
        new=AsyncMock(return_value=_claim(SUCCEEDED, False)),
    ), patch.object(
        identification_results,
        "is_result_reusable",
        new=AsyncMock(return_value=True),
    ), patch.object(identification_results, "_schedule_generation") as scheduled:
        response = client.post(
            "/api/identification-results",
            json={"image_id": 1, "model_id": 11},
        )

    assert response.status_code == 200
    assert response.json()["data"]["result_id"] == "result-1"
    scheduled.assert_not_called()


def test_post_scopes_both_sources_to_current_user():
    client, _ = _make_client()
    image_lookup = AsyncMock(return_value=None)

    with patch.object(
        identification_results.crud_images,
        "get_image_by_id",
        new=image_lookup,
    ):
        response = client.post(
            "/api/identification-results",
            json={"image_id": 999, "model_id": 11},
        )

    assert response.status_code == 404
    assert image_lookup.await_args.args[1:] == (999, 7)


def _stored_result(tmp_path, *, status=SUCCEEDED):
    grid = RasterGrid(
        width=2,
        height=1,
        crs="EPSG:4326",
        transform=Affine(0.01, 0, 110, 0, -0.01, 30),
    )
    stored = write_classification_result(
        tmp_path,
        "result-1",
        grid,
        np.array([[0, 1]], dtype=np.uint8),
        np.array([[1, 1]], dtype=np.uint8),
    )
    return SimpleNamespace(
        id="result-1",
        user_id=7,
        source_image_id=1,
        source_model_id=11,
        identity_sha256="c" * 64,
        image_content_sha256="a" * 64,
        weight_content_sha256="b" * 64,
        inference_parameters={"tile_size": 512, "overlap": 128},
        classification_scheme_version="land-cover-6/v1",
        pipeline_version="deeplab-native/v1",
        grid_policy_version="native-v1",
        status=status,
        started_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
        completed_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
        failure_detail=None,
        classes_path=str(stored.classes_path),
        valid_mask_path=str(stored.valid_mask_path),
        crs="EPSG:4326",
        transform=[0.01, 0, 110, 0, -0.01, 30],
        raster_width=2,
        raster_height=1,
        resolution=[0.01, 0.01],
        bounds=[110, 29.99, 110.02, 30],
    )


def test_get_returns_persisted_spatial_contract_and_hides_other_users(tmp_path):
    client, _ = _make_client()
    row = _stored_result(tmp_path)
    lookup = AsyncMock(side_effect=[row, None])

    with patch.object(
        identification_results.crud_results,
        "get_result_by_id",
        new=lookup,
    ):
        response = client.get("/api/identification-results/result-1")
        missing = client.get("/api/identification-results/other-user-result")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["grid"] == {
        "crs": "EPSG:4326",
        "transform": [0.01, 0, 110, 0, -0.01, 30],
        "width": 2,
        "height": 1,
        "resolution": [0.01, 0.01],
        "bounds": [110, 29.99, 110.02, 30],
    }
    assert data["classes"] == [
        {"id": 0, "name": "其他／背景"},
        {"id": 1, "name": "水系"},
        {"id": 2, "name": "林地"},
        {"id": 3, "name": "道路"},
        {"id": 4, "name": "种植土地"},
        {"id": 5, "name": "房屋建筑"},
    ]
    assert missing.status_code == 404


def test_render_defaults_to_1_through_5_but_explicit_zero_renders_background(tmp_path):
    client, _ = _make_client()
    row = _stored_result(tmp_path)

    with patch.object(
        identification_results.crud_results,
        "get_result_by_id",
        new=AsyncMock(return_value=row),
    ):
        default = client.get(
            "/api/identification-results/result-1/render",
            params={"bbox": "110,29.99,110.02,30", "width": 2, "height": 1, "srs": "EPSG:4326"},
        )
        background = client.get(
            "/api/identification-results/result-1/render",
            params=[
                ("bbox", "110,29.99,110.02,30"),
                ("width", "2"),
                ("height", "1"),
                ("srs", "EPSG:4326"),
                ("classes", "0"),
            ],
        )

    assert default.status_code == 200
    assert background.status_code == 200
    default_alpha = [pixel[3] for pixel in Image.open(io.BytesIO(default.content)).convert("RGBA").getdata()]
    background_alpha = [pixel[3] for pixel in Image.open(io.BytesIO(background.content)).convert("RGBA").getdata()]
    assert default_alpha == [0, 180]
    assert background_alpha == [180, 0]


def test_render_rejects_failed_or_unready_result(tmp_path):
    client, _ = _make_client()
    row = _stored_result(tmp_path, status=PROCESSING)

    with patch.object(
        identification_results.crud_results,
        "get_result_by_id",
        new=AsyncMock(return_value=row),
    ):
        response = client.get(
            "/api/identification-results/result-1/render",
            params={"bbox": "110,29.99,110.02,30", "width": 2, "height": 1},
        )

    assert response.status_code == 409
