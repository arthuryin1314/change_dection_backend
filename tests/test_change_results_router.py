import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.db_config import get_db
from router import change_results
from utils.classification_storage import RasterGrid
from utils.get_user_by_token import get_current_user
from utils.transition_matrix import TransitionMatrixError, TransitionMatrixResult


class FakeDatabase:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


def _client(db=None):
    database = db or FakeDatabase()

    async def override_db():
        yield database

    app = FastAPI()
    app.include_router(change_results.router)
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)
    return TestClient(app), database


def _payload():
    return {
        "request_id": "90f0bdba-6b7e-4a45-a078-dce12340f6d2",
        "before_image_id": 1,
        "after_image_id": 2,
        "model_id": 11,
        "before_result_id": "before-result",
        "before_identity_sha256": "a" * 64,
        "after_result_id": "after-result",
        "after_identity_sha256": "b" * 64,
    }


def _processing(**overrides):
    now = datetime(2026, 9, 9, tzinfo=timezone.utc)
    values = {
        "request_id": _payload()["request_id"],
        "user_id": 7,
        "status": "PROCESSING",
        "lease_owner": "worker",
        "started_at": now,
        "heartbeat_at": now,
        "completed_at": None,
        "error_http_status": None,
        "error_code": None,
        "error_message": None,
        "error_data": None,
        "matrix_m2": None,
        "before_image_id": 1,
        "after_image_id": 2,
        "source_model_id": 11,
        "before_result_id": "before-result",
        "before_identity_sha256": "a" * 64,
        "after_result_id": "after-result",
        "after_identity_sha256": "b" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolved_pair():
    before = SimpleNamespace(
        id="before-result",
        identity_sha256="a" * 64,
        classes_path="before/classes.tif",
        valid_mask_path="before/valid_mask.tif",
        source_image_id=1,
        source_model_id=11,
        image_content_sha256="1" * 64,
        weight_content_sha256="2" * 64,
        inference_parameters={"tile_size": 512},
        classification_scheme_version="land-cover-v1",
        pipeline_version="pipeline-v1",
        grid_policy_version="native-grid-v1",
        completed_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
        lease_owner="before-generation",
    )
    after = SimpleNamespace(**{**before.__dict__, "id": "after-result", "identity_sha256": "b" * 64, "source_image_id": 2})
    return SimpleNamespace(before=before, after=after)


def _matrix():
    return TransitionMatrixResult(
        matrix_m2=[[1.0 if row == column else 0.0 for column in range(6)] for row in range(6)],
        common_valid_area_m2=6.0,
        grid=RasterGrid(width=6, height=1, crs="EPSG:4326", transform=(0.01, 0, 110, 0, -0.01, 30)),
        bounds=[110.0, 29.99, 110.06, 30.0],
        before_window=[0, 0, 6, 1],
        after_window=[0, 0, 6, 1],
    )


def test_post_computes_after_releasing_request_connection_and_returns_saved_result():
    client, db = _client()
    processing = _processing()
    succeeded = _processing(
        status="SUCCEEDED",
        completed_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
        matrix_m2=_matrix().matrix_m2,
        common_valid_area_m2=6.0,
        crs="EPSG:4326",
        transform=[0.01, 0, 110, 0, -0.01, 30],
        raster_width=6,
        raster_height=1,
        bounds=_matrix().bounds,
        before_snapshot={"result_id": "before-result"},
        after_snapshot={"result_id": "after-result"},
        before_window=[0, 0, 6, 1],
        after_window=[0, 0, 6, 1],
        calculated_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
    )
    compute = Mock(side_effect=lambda *_: (_matrix() if db.rollbacks else (_ for _ in ()).throw(AssertionError("connection held"))))

    with patch.object(change_results, "resolve_change_inputs", new=AsyncMock(return_value=_resolved_pair())), patch.object(
        change_results.crud_results,
        "claim_request",
        new=AsyncMock(return_value=SimpleNamespace(row=processing, should_start=True)),
    ), patch.object(change_results, "compute_transition_matrix_m2", new=compute), patch.object(
        change_results.crud_results,
        "mark_succeeded",
        new=AsyncMock(return_value=succeeded),
    ), patch.object(change_results, "_heartbeat", new=AsyncMock()):
        response = client.post("/api/change-results", json=_payload())

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "SUCCEEDED"
    assert response.json()["data"]["matrix_m2"] == _matrix().matrix_m2
    assert db.rollbacks >= 1


def test_duplicate_processing_request_returns_202_without_computing():
    client, _ = _client()
    with patch.object(
        change_results.crud_results,
        "claim_request",
        new=AsyncMock(return_value=SimpleNamespace(row=_processing(), should_start=False)),
    ), patch.object(change_results, "resolve_change_inputs") as resolve, patch.object(
        change_results,
        "compute_transition_matrix_m2",
    ) as compute:
        response = client.post("/api/change-results", json=_payload())

    assert response.status_code == 202
    assert response.json()["data"]["status"] == "PROCESSING"
    resolve.assert_not_called()
    compute.assert_not_called()


def test_duplicate_request_id_with_different_inputs_returns_409():
    client, _ = _client()
    existing = _processing(after_image_id=99)
    with patch.object(
        change_results.crud_results,
        "claim_request",
        new=AsyncMock(return_value=SimpleNamespace(row=existing, should_start=False)),
    ), patch.object(change_results, "resolve_change_inputs") as resolve:
        response = client.post("/api/change-results", json=_payload())

    assert response.status_code == 409
    assert response.json()["data"]["error_code"] == "REQUEST_ID_CONFLICT"
    resolve.assert_not_called()


def test_domain_failure_returns_structured_422_and_persists_failure():
    client, _ = _client()
    failure = _processing(
        status="FAILED",
        error_http_status=422,
        error_code="GRID_MISMATCH",
        error_message="两期栅格像元大小不一致",
        error_data={"request_id": _payload()["request_id"]},
    )
    mark_failed = AsyncMock(return_value=failure)
    with patch.object(change_results, "resolve_change_inputs", new=AsyncMock(return_value=_resolved_pair())), patch.object(
        change_results.crud_results,
        "claim_request",
        new=AsyncMock(return_value=SimpleNamespace(row=_processing(), should_start=True)),
    ), patch.object(
        change_results,
        "compute_transition_matrix_m2",
        side_effect=TransitionMatrixError("GRID_MISMATCH", "两期栅格像元大小不一致"),
    ), patch.object(change_results.crud_results, "mark_failed", new=mark_failed), patch.object(
        change_results,
        "_heartbeat",
        new=AsyncMock(),
    ):
        response = client.post("/api/change-results", json=_payload())

    assert response.status_code == 422
    assert response.json()["data"]["error_code"] == "GRID_MISMATCH"
    assert response.json()["data"]["request_id"] == _payload()["request_id"]
    mark_failed.assert_awaited_once()



def test_change_input_failures_report_both_periods_with_stable_reasons():
    payload = change_results.CreateChangeResultRequest(**_payload())
    resolutions = [
        SimpleNamespace(status="MISSING", reason="VERSION_MISMATCH", row=None),
        SimpleNamespace(status="PROCESSING", reason="PROCESSING", row=None),
    ]
    with patch.object(
        change_results,
        "resolve_identification_for_change",
        new=AsyncMock(side_effect=resolutions),
    ):
        try:
            import asyncio
            asyncio.run(change_results.resolve_change_inputs(object(), 7, payload))
        except change_results.ChangeInputError as error:
            assert error.periods == {
                "before": {"reason": "VERSION_MISMATCH", "status": "MISSING"},
                "after": {"reason": "INCOMPLETE", "status": "PROCESSING"},
            }
        else:
            raise AssertionError("expected ChangeInputError")


@pytest.mark.parametrize(
    ("period_statuses", "expected_periods"),
    [
        (
            ("MISSING", "SUCCEEDED"),
            {"before": {"reason": "MISSING", "status": "MISSING"}},
        ),
        (
            ("SUCCEEDED", "MISSING"),
            {"after": {"reason": "MISSING", "status": "MISSING"}},
        ),
        (
            ("MISSING", "MISSING"),
            {
                "before": {"reason": "MISSING", "status": "MISSING"},
                "after": {"reason": "MISSING", "status": "MISSING"},
            },
        ),
        (
            ("PROCESSING", "FAILED"),
            {
                "before": {"reason": "INCOMPLETE", "status": "PROCESSING"},
                "after": {"reason": "INCOMPLETE", "status": "FAILED"},
            },
        ),
    ],
)
def test_api_reports_unavailable_periods_without_computing(period_statuses, expected_periods):
    client, _ = _client()
    resolved = _resolved_pair()
    rows = (resolved.before, resolved.after)
    resolutions = [
        SimpleNamespace(
            status=status,
            reason="MISSING" if status == "MISSING" else status,
            row=row if status == "SUCCEEDED" else None,
        )
        for status, row in zip(period_statuses, rows)
    ]

    async def persist_failure(_db, **values):
        return _processing(
            status="FAILED",
            completed_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
            error_http_status=values["http_status"],
            error_code=values["error_code"],
            error_message=values["message"],
            error_data=values["error_data"],
        )

    compute = Mock()
    resolve = AsyncMock(side_effect=resolutions)
    with patch.object(
        change_results.crud_results,
        "claim_request",
        new=AsyncMock(return_value=SimpleNamespace(row=_processing(), should_start=True)),
    ), patch.object(
        change_results,
        "resolve_identification_for_change",
        new=resolve,
    ), patch.object(
        change_results,
        "compute_transition_matrix_m2",
        new=compute,
    ), patch.object(
        change_results.crud_results,
        "mark_failed",
        new=AsyncMock(side_effect=persist_failure),
    ):
        response = client.post("/api/change-results", json=_payload())

    assert response.status_code == 422
    assert response.json()["data"]["error_code"] == "IDENTIFICATION_RESULT_UNAVAILABLE"
    assert response.json()["data"]["periods"] == expected_periods
    assert resolve.await_count == 2
    compute.assert_not_called()
