import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://user:pass@localhost/test",
)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.db_config import get_db
from router import change_results
from utils.get_user_by_token import get_current_user


REQUEST_ID = "90f0bdba-6b7e-4a45-a078-dce12340f6d2"
RETRY_ID = "16328ca2-acde-4f57-a8f5-201cbeec84f1"


class FakeDatabase:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


def _client(user_id=7):
    database = FakeDatabase()

    async def override_db():
        yield database

    app = FastAPI()
    app.include_router(change_results.router)
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    return TestClient(app), database


def _payload(request_id=REQUEST_ID):
    return {
        "request_id": request_id,
        "before_image_id": 1,
        "after_image_id": 2,
        "model_id": 11,
    }


def _submitted():
    return {
        "before_image_id": 1,
        "after_image_id": 2,
        "model_id": 11,
        "before": {
            "path": "uploads/images/before.tif",
            "cached_sha256": "1" * 64,
            "cached_size": 10,
            "cached_mtime_ns": 100,
        },
        "after": {
            "path": "uploads/images/after.tif",
            "cached_sha256": "2" * 64,
            "cached_size": 11,
            "cached_mtime_ns": 101,
        },
        "model": {
            "weight_path": "uploads/model_assets/weights.pth",
            "cached_sha256": "3" * 64,
            "cached_size": 12,
            "cached_mtime_ns": 102,
        },
        "identity_contract": {
            "inference_parameters": {"tile_size": 512, "overlap": 128},
            "classification_scheme_version": "land-cover-6/v1",
            "pipeline_version": "deeplab-native/v1",
            "grid_policy_version": "native-v1",
        },
    }


def _row(**overrides):
    now = datetime(2026, 9, 10, tzinfo=timezone.utc)
    values = {
        "id": 41,
        "request_id": REQUEST_ID,
        "user_id": 7,
        "status": "PROCESSING",
        "phase": "PREPARING",
        "lease_owner": "owner",
        "started_at": now,
        "heartbeat_at": now,
        "completed_at": None,
        "before_image_id": 1,
        "after_image_id": 2,
        "source_model_id": 11,
        "before_result_id": None,
        "before_identity_sha256": None,
        "after_result_id": None,
        "after_identity_sha256": None,
        "submitted_inputs": _submitted(),
        "error_http_status": None,
        "error_code": None,
        "error_message": None,
        "error_data": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _succeeded():
    row = _row(
        status="SUCCEEDED",
        phase="SUCCEEDED",
        completed_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        before_result_id="before-result",
        before_identity_sha256="a" * 64,
        after_result_id="after-result",
        after_identity_sha256="b" * 64,
        before_snapshot={"result_id": "before-result", "identity_sha256": "a" * 64},
        after_snapshot={"result_id": "after-result", "identity_sha256": "b" * 64},
        matrix_m2=[[0.0] * 6 for _ in range(6)],
        common_valid_area_m2=1.0,
        crs="EPSG:4528",
        transform=[1, 0, 0, 0, -1, 1],
        raster_width=1,
        raster_height=1,
        bounds=[0, 0, 1, 1],
        before_window=None,
        after_window=None,
        calculation_version="transition-matrix-v2",
        grid_policy_version="aligned-grid-v1",
        analysis_identity_sha256="c" * 64,
        analysis_metadata={"alignment_mode": "DIRECT"},
        calculated_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
    )
    return row


def test_post_always_returns_202_and_schedules_new_task():
    client, _ = _client()
    row = _row()
    claim = SimpleNamespace(
        row=row,
        should_start=True,
        request_conflict=False,
    )
    schedule = Mock()
    with patch.object(
        change_results.crud_results,
        "get_request_binding",
        new=AsyncMock(return_value=None),
    ), patch.object(
        change_results,
        "_submitted_inputs",
        new=AsyncMock(return_value=_submitted()),
    ), patch.object(
        change_results.crud_results,
        "claim_submission",
        new=AsyncMock(return_value=claim),
    ), patch.object(
        change_results,
        "schedule_change_orchestration",
        new=schedule,
    ):
        response = client.post("/api/change-results", json=_payload())

    assert response.status_code == 202
    assert response.json()["data"]["phase"] == "PREPARING"
    schedule.assert_called_once_with(41, 7, "owner")


def test_post_replay_of_completed_request_still_returns_202():
    client, _ = _client()
    binding = SimpleNamespace(
        change_result_id=41,
        request_fingerprint=change_results._request_fingerprint(
            change_results.CreateChangeResultRequest(**_payload())
        ),
    )
    with patch.object(
        change_results.crud_results,
        "get_request_binding",
        new=AsyncMock(return_value=binding),
    ), patch.object(
        change_results.crud_results,
        "get_by_id",
        new=AsyncMock(return_value=_succeeded()),
    ):
        response = client.post("/api/change-results", json=_payload())

    assert response.status_code == 202
    assert response.json()["data"]["status"] == "SUCCEEDED"


def test_post_replay_of_failed_request_returns_202_with_retry_details():
    client, _ = _client()
    binding = SimpleNamespace(
        change_result_id=41,
        request_fingerprint=change_results._request_fingerprint(
            change_results.CreateChangeResultRequest(**_payload())
        ),
    )
    failed = _row(
        status="FAILED",
        phase="FAILED",
        completed_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        error_code="IDENTIFICATION_EXECUTION_FAILED",
        error_message="识别执行失败",
        error_data={"retryable": True, "periods": {"after": {"reason": "推理失败"}}},
    )
    with patch.object(
        change_results.crud_results,
        "get_request_binding",
        new=AsyncMock(return_value=binding),
    ), patch.object(
        change_results.crud_results,
        "get_by_id",
        new=AsyncMock(return_value=failed),
    ):
        response = client.post("/api/change-results", json=_payload())

    assert response.status_code == 202
    assert response.json()["data"]["error_code"] == "IDENTIFICATION_EXECUTION_FAILED"
    assert response.json()["data"]["error_data"]["retryable"] is True


def test_post_replay_upgrades_matching_legacy_request_fingerprint():
    client, database = _client()
    binding = SimpleNamespace(
        change_result_id=41,
        request_fingerprint="legacy:0123456789abcdef0123456789abcdef",
    )
    with patch.object(
        change_results.crud_results,
        "get_request_binding",
        new=AsyncMock(return_value=binding),
    ), patch.object(
        change_results.crud_results,
        "get_by_id",
        new=AsyncMock(return_value=_succeeded()),
    ):
        response = client.post("/api/change-results", json=_payload())

    assert response.status_code == 202
    assert binding.request_fingerprint == change_results._request_fingerprint(
        change_results.CreateChangeResultRequest(**_payload())
    )
    assert database.commits == 1


def test_post_rejects_legacy_request_id_for_different_selection():
    client, database = _client()
    binding = SimpleNamespace(
        change_result_id=41,
        request_fingerprint="legacy:0123456789abcdef0123456789abcdef",
    )
    row = _succeeded()
    row.after_image_id = 99
    with patch.object(
        change_results.crud_results,
        "get_request_binding",
        new=AsyncMock(return_value=binding),
    ), patch.object(
        change_results.crud_results,
        "get_by_id",
        new=AsyncMock(return_value=row),
    ):
        response = client.post("/api/change-results", json=_payload())

    assert response.status_code == 409
    assert response.json()["data"]["error_code"] == "REQUEST_ID_CONFLICT"
    assert binding.request_fingerprint.startswith("legacy:")
    assert database.commits == 0


def test_post_rejects_same_request_id_for_different_selection():
    client, _ = _client()
    binding = SimpleNamespace(
        change_result_id=41,
        request_fingerprint="f" * 64,
    )
    with patch.object(
        change_results.crud_results,
        "get_request_binding",
        new=AsyncMock(return_value=binding),
    ):
        response = client.post("/api/change-results", json=_payload())

    assert response.status_code == 409
    assert response.json()["data"]["error_code"] == "REQUEST_ID_CONFLICT"


def test_get_processing_includes_selection_and_period_progress():
    client, _ = _client()
    row = _row(
        phase="ENSURING_AFTER",
        before_result_id="before-result",
        before_identity_sha256="a" * 64,
    )
    with patch.object(
        change_results.crud_results,
        "expire_request",
        new=AsyncMock(return_value=0),
    ), patch.object(
        change_results.crud_results,
        "get_by_request",
        new=AsyncMock(return_value=row),
    ):
        response = client.get(f"/api/change-results/{REQUEST_ID}")

    assert response.status_code == 202
    data = response.json()["data"]
    assert data["inputs"] == {
        "before_image_id": 1,
        "after_image_id": 2,
        "model_id": 11,
    }
    assert data["periods"]["before"]["status"] == "SUCCEEDED"
    assert data["periods"]["after"]["status"] == "PROCESSING"


def test_retry_expires_stale_request_before_checking_failed_state():
    client, _ = _client()
    stale = _row(
        heartbeat_at=datetime.now(timezone.utc) - timedelta(minutes=3),
    )
    failed = _row(
        status="FAILED",
        phase="FAILED",
        error_http_status=409,
        error_code="CHANGE_RESULT_INTERRUPTED",
        error_message="变化分析执行已中断",
        error_data={"retryable": True},
    )
    order = []

    async def expire(*_args, **_kwargs):
        order.append("expire")
        return 1

    async def reload(*_args, **_kwargs):
        order.append("reload")
        return failed if order and order[0] == "expire" else stale

    claim = SimpleNamespace(
        row=_row(id=42, request_id=RETRY_ID),
        should_start=True,
        request_conflict=False,
    )
    with patch.object(
        change_results.crud_results,
        "expire_request",
        new=AsyncMock(side_effect=expire),
    ), patch.object(
        change_results.crud_results,
        "get_by_request",
        new=AsyncMock(side_effect=reload),
    ), patch.object(
        change_results.crud_results,
        "claim_submission",
        new=AsyncMock(return_value=claim),
    ), patch.object(
        change_results,
        "schedule_change_orchestration",
    ):
        response = client.post(
            f"/api/change-results/{REQUEST_ID}/retry",
            json={"request_id": RETRY_ID},
        )

    assert response.status_code == 202
    assert order[:2] == ["expire", "reload"]


def test_get_other_users_request_returns_404():
    client, _ = _client(user_id=8)
    with patch.object(
        change_results.crud_results,
        "expire_request",
        new=AsyncMock(return_value=0),
    ), patch.object(
        change_results.crud_results,
        "get_by_request",
        new=AsyncMock(return_value=None),
    ):
        response = client.get(f"/api/change-results/{REQUEST_ID}")

    assert response.status_code == 404
