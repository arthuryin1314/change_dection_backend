import asyncio
import hashlib
import io
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from rasterio.crs import CRS as RasterioCRS
from rasterio.transform import array_bounds
from affine import Affine

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from config.db_config import get_db
from router import identification_results
from tests.fake_inference_runtime import FakeInferenceRuntime
from services.identification_results import (
    PROCESSING,
    SUCCEEDED,
    IdentificationResultRecord,
    ResultIdentity,
    ResultClaim,
)
from utils.classification_contract import inference_parameters
from utils.classification_storage import RasterGrid, write_classification_result
from utils.get_user_by_token import get_current_user
from utils import classification_area


class FakeDatabase:
    def __init__(self):
        self.committed = False
        self.flushed = False
        self.rolled_back = False

    async def flush(self):
        self.flushed = True

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.rolled_back = True


def _make_client(db=None, current_user=None):
    database = db or FakeDatabase()
    user = current_user if current_user is not None else SimpleNamespace(id=7)

    async def override_db():
        yield database

    app = FastAPI()
    app.include_router(identification_results.router)
    app.dependency_overrides[identification_results.get_inference_runtime] = FakeInferenceRuntime
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: user
    return TestClient(app), database


class ExpiringCurrentUser:
    def __init__(self, db):
        self.db = db

    @property
    def id(self):
        if self.db.rolled_back:
            raise RuntimeError("current_user was accessed after rollback")
        return 7


def _sources(tmp_path):
    weight_path = tmp_path / "fixture.pth"
    weight_path.write_bytes(b"fake weight for source identity")
    image = SimpleNamespace(
        id=1,
        user_id=7,
        image_name="计算时影像",
        img_path=__file__,
        content_sha256="a" * 64,
        content_sha256_size=os.path.getsize(__file__),
        content_sha256_mtime_ns=os.stat(__file__).st_mtime_ns,
    )
    model = SimpleNamespace(
        id=11,
        user_id=7,
        model_name="计算时模型",
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


def test_post_returns_202_and_schedules_only_after_database_commit(tmp_path):
    client, db = _make_client()
    image, model = _sources(tmp_path)
    scheduled = Mock(side_effect=lambda request, owner, runtime: setattr(db, "scheduled_after_commit", db.committed))

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


def test_post_captures_source_names_outside_result_identity(tmp_path):
    client, _ = _make_client()
    image, model = _sources(tmp_path)
    store = Mock()

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
        "SqlAlchemyClaimStore",
        return_value=store,
    ) as store_type, patch.object(
        identification_results,
        "claim_identification_result",
        new=AsyncMock(return_value=_claim()),
    ), patch.object(identification_results, "_schedule_generation"):
        response = client.post(
            "/api/identification-results",
            json={"image_id": 1, "model_id": 11},
        )

    assert response.status_code == 202
    assert store_type.call_args.kwargs["source_snapshot"] == {
        "image": {"id": 1, "name": "计算时影像"},
        "model": {"id": 11, "name": "计算时模型"},
    }


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


def test_post_reuses_succeeded_identity_without_scheduling(tmp_path):
    client, _ = _make_client()
    image, model = _sources(tmp_path)

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


def _stored_result(tmp_path, *, status=SUCCEEDED, grid=None):
    grid = grid if grid is not None else RasterGrid(
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
        source_snapshot={
            "image": {"id": 1, "name": "计算时影像"},
            "model": {"id": 11, "name": "计算时模型"},
        },
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
        lease_owner="worker",
        failure_detail=None,
        classes_path=str(stored.classes_path),
        valid_mask_path=str(stored.valid_mask_path),
        crs=grid.crs,
        transform=list(grid.transform)[:6],
        raster_width=grid.width,
        raster_height=grid.height,
        resolution=[abs(grid.transform.a), abs(grid.transform.e)],
        bounds=list(array_bounds(grid.height, grid.width, grid.transform)),
        class_area_m2=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        area_status="SUCCEEDED",
        area_completed_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
        area_failure_detail=None,
    )


def test_resolve_returns_complete_identity_match_without_writing_database(tmp_path):
    client, db = _make_client()
    image, model = _sources(tmp_path)
    row = _stored_result(tmp_path)
    matched = _claim(SUCCEEDED, False).record
    lookup = AsyncMock(return_value=matched)

    with patch.object(
        identification_results.crud_images,
        "get_image_by_id",
        new=AsyncMock(return_value=image),
    ), patch.object(
        identification_results.crud_ml_models,
        "get_ml_model_by_id",
        new=AsyncMock(return_value=model),
    ), patch.object(
        identification_results.SqlAlchemyClaimStore,
        "get_by_identity",
        new=lookup,
    ), patch.object(
        identification_results.crud_results,
        "get_result_by_id",
        new=AsyncMock(return_value=row),
    ):
        response = client.get(
            "/api/identification-results/resolve",
            params={"image_id": 1, "model_id": 11},
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "SUCCEEDED"
    assert data["result"]["result_id"] == "result-1"
    assert data["result"]["class_area_m2"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert db.flushed is False
    assert db.committed is False


def test_history_lists_saved_results_and_reads_detail_without_writing(tmp_path):
    client, db = _make_client()
    del client.app.dependency_overrides[identification_results.get_inference_runtime]
    row = _stored_result(tmp_path)

    with patch.object(
        identification_results.crud_results,
        "list_succeeded_history",
        new=AsyncMock(return_value=([row], 1)),
        create=True,
    ), patch.object(
        identification_results.crud_results,
        "get_succeeded_history_by_id",
        new=AsyncMock(return_value=row),
        create=True,
    ):
        listing = client.get("/api/identification-results/history")
        detail = client.get("/api/identification-results/history/result-1")

    assert listing.status_code == 200
    assert listing.json()["data"] == {
        "items": [{
            "result_id": "result-1",
            "source": row.source_snapshot,
            "completed_at": "2026-09-06T00:00:00+00:00",
            "area_status": "SUCCEEDED",
        }],
        "total": 1,
        "page": 1,
        "page_size": 20,
    }
    assert detail.status_code == 200
    assert detail.json()["data"]["source"] == row.source_snapshot
    assert detail.json()["data"]["render_status"] == "AVAILABLE"
    assert db.committed is False
    assert db.flushed is False


def test_history_detail_reports_missing_saved_files_without_invalidating_result(tmp_path):
    client, db = _make_client()
    row = _stored_result(tmp_path)
    Path(row.classes_path).unlink()

    with patch.object(
        identification_results.crud_results,
        "get_succeeded_history_by_id",
        new=AsyncMock(return_value=row),
        create=True,
    ), patch.object(
        identification_results.crud_results,
        "invalidate_succeeded_result",
        new=AsyncMock(),
    ) as invalidate:
        response = client.get("/api/identification-results/history/result-1")

    assert response.status_code == 200
    assert response.json()["data"]["render_status"] == "UNAVAILABLE"
    assert db.committed is False
    invalidate.assert_not_awaited()


def test_resolve_missing_identity_does_not_create_or_schedule_result(tmp_path):
    client, db = _make_client()
    image, model = _sources(tmp_path)

    with patch.object(
        identification_results.crud_images,
        "get_image_by_id",
        new=AsyncMock(return_value=image),
    ), patch.object(
        identification_results.crud_ml_models,
        "get_ml_model_by_id",
        new=AsyncMock(return_value=model),
    ), patch.object(
        identification_results.SqlAlchemyClaimStore,
        "get_by_identity",
        new=AsyncMock(return_value=None),
    ), patch.object(
        identification_results.crud_results,
        "get_latest_by_sources",
        new=AsyncMock(return_value=None),
    ), patch.object(identification_results, "_schedule_generation") as scheduled:
        response = client.get(
            "/api/identification-results/resolve",
            params={"image_id": 1, "model_id": 11},
        )

    assert response.status_code == 200
    assert response.json()["data"] == {"status": "MISSING"}
    assert db.flushed is False
    assert db.committed is False
    scheduled.assert_not_called()


def test_resolve_snapshots_current_user_before_releasing_read_transaction(tmp_path):
    db = FakeDatabase()
    client, _ = _make_client(db, ExpiringCurrentUser(db))
    image, model = _sources(tmp_path)

    with patch.object(
        identification_results.crud_images,
        "get_image_by_id",
        new=AsyncMock(return_value=image),
    ), patch.object(
        identification_results.crud_ml_models,
        "get_ml_model_by_id",
        new=AsyncMock(return_value=model),
    ), patch.object(
        identification_results.SqlAlchemyClaimStore,
        "get_by_identity",
        new=AsyncMock(return_value=None),
    ), patch.object(
        identification_results.crud_results,
        "get_latest_by_sources",
        new=AsyncMock(return_value=None),
    ):
        response = client.get(
            "/api/identification-results/resolve",
            params={"image_id": 1, "model_id": 11},
        )

    assert response.status_code == 200
    assert response.json()["data"] == {"status": "MISSING"}


def test_area_request_computes_saved_grid_and_commits_complete_summary(tmp_path):
    db = FakeDatabase()
    client, _ = _make_client(db, ExpiringCurrentUser(db))
    row = _stored_result(tmp_path)
    row.area_status = "NOT_COMPUTED"
    row.class_area_m2 = None
    saved = AsyncMock(return_value=True)

    with patch.object(
        identification_results.crud_results,
        "get_result_by_id",
        new=AsyncMock(return_value=row),
    ), patch.object(
        identification_results.crud_results,
        "mark_area_succeeded",
        new=saved,
    ):
        response = client.post(
            "/api/identification-results/result-1/areas",
            params={"identity_sha256": "c" * 64},
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["result_id"] == "result-1"
    assert data["identity_sha256"] == "c" * 64
    assert data["area_status"] == "SUCCEEDED"
    assert data["class_area_m2"] == pytest.approx(
        [1069626.783188343, 1069626.783188343, 0, 0, 0, 0]
    )
    assert db.rolled_back is True
    assert db.committed is True
    assert saved.await_args.kwargs["lease_owner"] == "worker"
    assert saved.await_args.kwargs["completed_at"] == row.completed_at


def test_area_requests_serialize_large_raster_scans(tmp_path):
    rows = {
        "before": _stored_result(tmp_path / "before"),
        "after": _stored_result(tmp_path / "after"),
    }
    for row in rows.values():
        row.id = row.id + row.id
        row.area_status = "NOT_COMPUTED"
        row.class_area_m2 = None

    active = 0
    peak = 0

    async def lookup(db, result_id, user_id):
        return rows[result_id]

    async def save(*args, **kwargs):
        return True

    def scan(*args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        import time

        time.sleep(0.02)
        active -= 1
        return [1, 2, 3, 4, 5, 6]

    async def run():
        original_semaphore = identification_results._classification_raster_semaphore
        identification_results._classification_raster_semaphore = asyncio.Semaphore(1)
        with patch.object(
            identification_results.crud_results,
            "get_result_by_id",
            new=lookup,
        ), patch.object(
            identification_results.crud_results,
            "mark_area_succeeded",
            new=save,
        ), patch.object(
            identification_results,
            "summarize_classification_area_m2",
            new=scan,
        ):
            try:
                await asyncio.gather(
                    identification_results.calculate_identification_result_areas(
                        "before", "c" * 64, FakeDatabase(), SimpleNamespace(id=7)
                    ),
                    identification_results.calculate_identification_result_areas(
                        "after", "c" * 64, FakeDatabase(), SimpleNamespace(id=7)
                    ),
                )
            finally:
                identification_results._classification_raster_semaphore = original_semaphore

    asyncio.run(run())
    assert peak == 1


def test_area_request_rejects_result_replaced_during_scan(tmp_path):
    client, db = _make_client()
    row = _stored_result(tmp_path)
    row.area_status = "NOT_COMPUTED"
    row.class_area_m2 = None

    with patch.object(
        identification_results.crud_results,
        "get_result_by_id",
        new=AsyncMock(return_value=row),
    ), patch.object(
        identification_results.crud_results,
        "mark_area_succeeded",
        new=AsyncMock(return_value=False),
    ):
        response = client.post(
            "/api/identification-results/result-1/areas",
            params={"identity_sha256": "c" * 64},
        )

    assert response.status_code == 409
    assert db.committed is False


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

def test_first_render_waits_for_area_scan_then_history_and_reuse_render(tmp_path):
    rows = {period: _stored_result(tmp_path / period) for period in ("before", "after")}
    for period, row in rows.items():
        row.id = period
        row.area_status = "NOT_COMPUTED"
        row.class_area_m2 = None
    acquired = threading.Event()
    release = threading.Event()
    original_lock = classification_area.classification_result_lock

    @contextmanager
    def hold_area_lock(*args, **kwargs):
        with original_lock(*args, **kwargs):
            acquired.set()
            assert release.wait(10), "area scan was not released"
            yield

    async def lookup(db, result_id, user_id):
        return rows[result_id]

    async def save(db, **kwargs):
        row = rows[kwargs["result_id"]]
        row.area_status = SUCCEEDED
        row.class_area_m2 = kwargs["class_area_m2"]
        row.area_completed_at = kwargs["area_completed_at"]
        return True

    async def run():
        client, _ = _make_client()
        with patch.object(identification_results, "_classification_raster_semaphore", asyncio.Semaphore(1)), patch.object(
            identification_results.crud_results, "get_result_by_id", new=lookup,
        ), patch.object(
            identification_results.crud_results, "mark_area_succeeded", new=save,
        ), patch.object(classification_area, "classification_result_lock", new=hold_area_lock):
            async with AsyncClient(transport=ASGITransport(app=client.app, raise_app_exceptions=False), base_url="http://test") as http:
                params = {"bbox": "110,29.99,110.02,30", "width": 2, "height": 1, "classes": [0, 1]}
                area = asyncio.create_task(http.post("/api/identification-results/before/areas", params={"identity_sha256": "c" * 64}))
                try:
                    assert await asyncio.to_thread(acquired.wait, 5)

                    async def finish_scan():
                        # Exceed the real validation lock timeout, as a first large scan does.
                        await asyncio.sleep(2.2)
                        release.set()

                    before, after, _ = await asyncio.gather(
                        http.get("/api/identification-results/before/render", params=params),
                        http.get("/api/identification-results/after/render", params=params),
                        finish_scan(),
                    )
                    assert (await area).status_code == 200
                    assert [before.status_code, after.status_code] == [200, 200]
                    assert np.asarray(Image.open(io.BytesIO(before.content)))[:, :, 3].any()
                    history = await http.get("/api/identification-results/before/render", params=params)
                    assert history.status_code == 200
                    cached_area = await http.post("/api/identification-results/before/areas", params={"identity_sha256": "c" * 64})
                    assert cached_area.status_code == 200
                    reused = await http.get("/api/identification-results/before/render", params=params)
                    assert reused.content == history.content == before.content
                finally:
                    release.set()
                    await area

    asyncio.run(run())


@pytest.mark.parametrize("busy_step", ["_row_is_reusable", "render_classification_png"])
def test_render_maps_busy_file_lock_to_retryable_409(tmp_path, busy_step):
    client, _ = _make_client()
    row = _stored_result(tmp_path)
    with patch.object(identification_results.crud_results, "get_result_by_id", new=AsyncMock(return_value=row)), patch.object(
        identification_results, busy_step, side_effect=TimeoutError("等待识别结果文件锁超时"),
    ):
        response = client.get("/api/identification-results/result-1/render", params={"bbox": "110,29.99,110.02,30", "width": 2, "height": 1})
    assert response.status_code == 409
    assert response.json()["detail"] == "等待识别结果文件锁超时"


def test_resolve_unavailable_result_does_not_invalidate_or_write(tmp_path):
    client, db = _make_client()
    image, model = _sources(tmp_path)
    row = _stored_result(tmp_path)

    with patch.object(
        identification_results.crud_images,
        "get_image_by_id",
        new=AsyncMock(return_value=image),
    ), patch.object(
        identification_results.crud_ml_models,
        "get_ml_model_by_id",
        new=AsyncMock(return_value=model),
    ), patch.object(
        identification_results.SqlAlchemyClaimStore,
        "get_by_identity",
        new=AsyncMock(return_value=_claim(SUCCEEDED, False).record),
    ), patch.object(
        identification_results.crud_results,
        "get_result_by_id",
        new=AsyncMock(return_value=row),
    ), patch.object(
        identification_results,
        "_row_is_reusable",
        return_value=False,
    ), patch.object(
        identification_results.crud_results,
        "invalidate_succeeded_result",
        new=AsyncMock(),
    ) as invalidate:
        response = client.get(
            "/api/identification-results/resolve",
            params={"image_id": 1, "model_id": 11},
        )

    assert response.status_code == 200
    assert response.json()["data"] == {"status": "UNAVAILABLE"}
    assert db.flushed is False
    assert db.committed is False
    invalidate.assert_not_awaited()


def test_area_failure_is_persisted_without_invalidating_classification(tmp_path):
    client, db = _make_client()
    row = _stored_result(tmp_path)
    row.area_status = "NOT_COMPUTED"
    row.class_area_m2 = None
    save_failure = AsyncMock(return_value=True)

    with patch.object(
        identification_results.crud_results,
        "get_result_by_id",
        new=AsyncMock(return_value=row),
    ), patch.object(
        identification_results,
        "summarize_classification_area_m2",
        side_effect=ValueError("有效掩膜损坏"),
    ), patch.object(
        identification_results.crud_results,
        "mark_area_failed",
        new=save_failure,
    ), patch.object(
        identification_results.crud_results,
        "invalidate_succeeded_result",
        new=AsyncMock(),
    ) as invalidate:
        response = client.post(
            "/api/identification-results/result-1/areas",
            params={"identity_sha256": "c" * 64},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "有效掩膜损坏"
    assert db.rolled_back is True
    assert db.committed is True
    assert save_failure.await_args.kwargs["lease_owner"] == "worker"
    assert save_failure.await_args.kwargs["completed_at"] == row.completed_at
    invalidate.assert_not_awaited()
    assert row.status == SUCCEEDED


def test_area_request_rejects_stale_identity_before_scanning(tmp_path):
    client, db = _make_client()
    row = _stored_result(tmp_path)
    row.area_status = "NOT_COMPUTED"
    row.class_area_m2 = None

    with patch.object(
        identification_results.crud_results,
        "get_result_by_id",
        new=AsyncMock(return_value=row),
    ), patch.object(
        identification_results,
        "summarize_classification_area_m2",
    ) as summarize:
        response = client.post(
            "/api/identification-results/result-1/areas",
            params={"identity_sha256": "d" * 64},
        )

    assert response.status_code == 409
    assert db.rolled_back is False
    summarize.assert_not_called()


def test_reusability_validation_is_offloaded_from_async_request_loop():
    row = SimpleNamespace(status=SUCCEEDED)
    lookup = AsyncMock(return_value=row)
    offload = AsyncMock(return_value=True)

    with patch.object(
        identification_results.crud_results,
        "get_result_by_id",
        new=lookup,
    ), patch.object(
        identification_results.asyncio,
        "to_thread",
        new=offload,
    ):
        reusable = asyncio.run(
            identification_results.is_result_reusable(object(), "result-1", 7)
        )

    assert reusable is True
    offload.assert_awaited_once_with(identification_results._row_is_reusable, row)

def test_resolve_recomputes_changed_source_hash_without_persisting_cache(tmp_path):
    client, db = _make_client()
    image, model = _sources(tmp_path)
    changed_source = tmp_path / "changed-source.tif"
    changed_source.write_bytes(b"changed-image-content")
    image.img_path = str(changed_source)
    image.content_sha256_size = 0
    lookup = AsyncMock(return_value=None)

    expected_identity = ResultIdentity(
        user_id=7,
        image_content_sha256=hashlib.sha256(b"changed-image-content").hexdigest(),
        weight_content_sha256="b" * 64,
        inference_parameters=inference_parameters(),
    ).sha256()

    with patch.object(
        identification_results.crud_images,
        "get_image_by_id",
        new=AsyncMock(return_value=image),
    ), patch.object(
        identification_results.crud_ml_models,
        "get_ml_model_by_id",
        new=AsyncMock(return_value=model),
    ), patch.object(
        identification_results.SqlAlchemyClaimStore,
        "get_by_identity",
        new=lookup,
    ), patch.object(
        identification_results.crud_results,
        "get_latest_by_sources",
        new=AsyncMock(return_value=None),
    ):
        response = client.get(
            "/api/identification-results/resolve",
            params={"image_id": 1, "model_id": 11},
        )

    assert response.status_code == 200
    assert response.json()["data"] == {"status": "MISSING"}
    assert lookup.await_args.args == (7, expected_identity)
    assert db.flushed is False
    assert db.committed is False


@pytest.mark.parametrize(
    ("status", "failure_detail", "expected"),
    (
        (PROCESSING, None, {"status": PROCESSING}),
        ("FAILED", "模型推理失败", {"status": "FAILED", "failure_detail": "模型推理失败"}),
    ),
)
def test_resolve_exposes_matching_unready_status_without_writing(
    tmp_path,
    status,
    failure_detail,
    expected,
):
    client, db = _make_client()
    image, model = _sources(tmp_path)
    row = _stored_result(tmp_path, status=status)
    row.failure_detail = failure_detail

    with patch.object(
        identification_results.crud_images,
        "get_image_by_id",
        new=AsyncMock(return_value=image),
    ), patch.object(
        identification_results.crud_ml_models,
        "get_ml_model_by_id",
        new=AsyncMock(return_value=model),
    ), patch.object(
        identification_results.SqlAlchemyClaimStore,
        "get_by_identity",
        new=AsyncMock(return_value=_claim(status, False).record),
    ), patch.object(
        identification_results.crud_results,
        "get_result_by_id",
        new=AsyncMock(return_value=row),
    ):
        response = client.get(
            "/api/identification-results/resolve",
            params={"image_id": 1, "model_id": 11},
        )

    assert response.status_code == 200
    assert response.json()["data"] == expected
    assert db.flushed is False
    assert db.committed is False


def test_change_resolver_distinguishes_previous_source_version(tmp_path):
    image, model = _sources(tmp_path)
    db = FakeDatabase()
    old = SimpleNamespace(id="old-version")

    with patch.object(
        identification_results.crud_images,
        "get_image_by_id",
        new=AsyncMock(return_value=image),
    ), patch.object(
        identification_results.crud_ml_models,
        "get_ml_model_by_id",
        new=AsyncMock(return_value=model),
    ), patch.object(
        identification_results.SqlAlchemyClaimStore,
        "get_by_identity",
        new=AsyncMock(return_value=None),
    ), patch.object(
        identification_results.crud_results,
        "get_latest_by_sources",
        new=AsyncMock(return_value=old),
    ):
        resolution = asyncio.run(
            identification_results.resolve_identification_for_change(
                db,
                user_id=7,
                image_id=1,
                model_id=11,
            )
        )

    assert resolution.status == "MISSING"
    assert resolution.reason == "VERSION_MISMATCH"


def test_resolve_maps_busy_result_lock_to_retryable_409():
    client, _ = _make_client()
    with patch.object(
        identification_results,
        "resolve_identification_for_change",
        new=AsyncMock(side_effect=TimeoutError("等待识别结果文件锁超时")),
    ):
        response = client.get(
            "/api/identification-results/resolve",
            params={"image_id": 1, "model_id": 11},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "等待识别结果文件锁超时"


@pytest.mark.parametrize("crs,transform", [
    ("EPSG:4528", Affine(0.8, 0, 40558753.6, 0, -0.8, 3571463.2)),
    (RasterioCRS.from_epsg(4528).to_wkt(), Affine(0.8, 0, 40558753.6, 0, -0.8, 3571463.2)),
    ("EPSG:4326", Affine(0.01, 0, 110, 0, -0.01, 30)),
])
def test_history_render_grid_produces_reprojected_classification_png(tmp_path, crs, transform):
    client, db = _make_client()
    row = _stored_result(tmp_path, grid=RasterGrid(width=2, height=1, crs=crs, transform=transform))
    with patch.object(identification_results.crud_results, "get_succeeded_history_by_id", new=AsyncMock(return_value=row)), patch.object(
        identification_results.crud_results, "get_result_by_id", new=AsyncMock(return_value=row),
    ):
        detail = client.get("/api/identification-results/history/result-1")
        assert detail.status_code == 200
        data = detail.json()["data"]
        assert data["grid"]["crs"] == crs
        grid = data["render_grid"]
        assert grid["crs"] == "EPSG:3857"
        left, bottom, right, top = grid["bounds"]
        assert 12_000_000 < left < right < 14_000_000
        assert 3_000_000 < bottom < top < 4_500_000
        image = client.get("/api/identification-results/result-1/render", params={
            "bbox": ",".join(map(str, grid["bounds"])), "srs": grid["crs"],
            "width": 200, "height": 100, "classes": [0, 1],
        })
    assert image.status_code == 200
    pixels = np.array(Image.open(io.BytesIO(image.content)).convert("RGBA"))
    opaque = pixels[pixels[:, :, 3] > 0]
    assert {tuple(pixel) for pixel in opaque} == {(0, 0, 0, 180), (0, 0, 255, 180)}
    assert not db.committed


@pytest.mark.parametrize("crs,bounds", [
    ("EPSG:NOT_REAL", [110, 29, 111, 30]),
    (None, [110, 29, 111, 30]),
    ("EPSG:4326", [110, 95, 111, 96]),
    ("EPSG:4326", None),
])
def test_history_projection_failure_keeps_other_detail_readable(tmp_path, crs, bounds):
    client, db = _make_client()
    row = _stored_result(tmp_path)
    row.crs = crs
    row.bounds = bounds
    with patch.object(identification_results.crud_results, "get_succeeded_history_by_id", new=AsyncMock(return_value=row)):
        response = client.get("/api/identification-results/history/result-1")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == SUCCEEDED
    assert data["class_area_m2"] == row.class_area_m2
    assert data["render_grid"] is None
    assert data["render_error"] == "分类图坐标系或范围无法转换为地图坐标，请检查空间网格信息"
    assert not db.committed


@pytest.mark.parametrize("crs,expected", [
    ("EPSG:4326", "EPSG:4326 · WGS 84"),
    (RasterioCRS.from_epsg(4528).to_wkt(), "EPSG:4528 · CGCS2000 / 3-degree Gauss-Kruger zone 40"),
    ('LOCAL_CS["Local survey grid",UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]', "Local survey grid"),
    ("invalid projection", "无法识别的坐标系"),
    (None, "未记录"),
])
def test_history_shows_crs_label_and_preserves_original_definition(tmp_path, crs, expected):
    client, _ = _make_client()
    row = _stored_result(tmp_path)
    row.crs = crs
    with patch.object(identification_results.crud_results, "get_succeeded_history_by_id", new=AsyncMock(return_value=row)):
        response = client.get("/api/identification-results/history/result-1")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["crs_label"] == expected
    assert data["grid"]["crs"] == crs
