"""Acceptance checks that must run in the Web backend's Torch-free environment."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("path", "payload", "error_code"),
    [
        (
            "/api/segment",
            {"image_id": 1, "model_id": 1, "bbox": "0,0,1,1", "width": 10, "height": 10},
            "INFERENCE_UNAVAILABLE",
        ),
        (
            "/api/identification-results",
            {"image_id": 1, "model_id": 1},
            "INFERENCE_UNAVAILABLE",
        ),
        (
            "/api/change-results",
            {
                "request_id": "90f0bdba-6b7e-4a45-a078-dce12340f6d2",
                "before_image_id": 1, "after_image_id": 2, "model_id": 1,
            },
            "CHANGE_DETECTION_DISABLED",
        ),
        (
            "/api/change-results/90f0bdba-6b7e-4a45-a078-dce12340f6d2/retry",
            {"request_id": "16328ca2-acde-4f57-a8f5-201cbeec84f1"},
            "CHANGE_DETECTION_DISABLED",
        ),
    ],
)
def test_inference_requests_fail_before_database_or_file_work(
    path, payload, error_code, monkeypatch
):
    from fastapi.testclient import TestClient
    from main import app
    from config.db_config import get_db
    from utils.get_user_by_token import get_current_user

    async def database_without_operations():
        yield SimpleNamespace()

    monkeypatch.setitem(app.dependency_overrides, get_db, database_without_operations)
    monkeypatch.setitem(app.dependency_overrides, get_current_user, lambda: SimpleNamespace(id=7))
    response = TestClient(app).post(path, json=payload)
    assert response.status_code == 503
    assert response.json()["code"] == 503
    assert response.json()["data"]["error_code"] == error_code
    assert "暂不" in response.json()["message"]


def test_application_imports_without_torch():
    environment = dict(os.environ)
    environment.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; "
            "assert importlib.util.find_spec('torch') is None; "
            "assert importlib.util.find_spec('torchvision') is None; "
            "import main; assert main.app.openapi()['paths']",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_login_works_with_the_clean_environment_dependencies(monkeypatch):
    import bcrypt
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock
    from main import app
    from config.db_config import get_db
    from router.users import routes

    user = SimpleNamespace(
        id=7, username="environment-test", phone="13800000000",
        password=bcrypt.hashpw(b"ticket54-password", bcrypt.gensalt()).decode("ascii"),
    )

    async def database():
        yield SimpleNamespace(commit=AsyncMock())

    monkeypatch.setitem(app.dependency_overrides, get_db, database)
    monkeypatch.setattr(routes, "get_user_by_telNum", AsyncMock(return_value=user))
    monkeypatch.setattr(routes, "create_token", AsyncMock(return_value="test-token"))
    client = TestClient(app)
    response = client.post(
        "/api/users/login", json={"telNum": user.phone, "password": "ticket54-password"}
    )
    assert response.status_code == 200
    denied = client.post(
        "/api/users/login", json={"telNum": user.phone, "password": "wrong-password"}
    )
    assert denied.status_code == 400
