import os
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from config.db_config import get_db
from router import ml_models
from utils.get_user_by_token import get_current_user


@pytest.mark.parametrize("method,path", [
    ("post", "/api/models/upload"),
    ("put", "/api/models/1"),
    ("delete", "/api/models/1"),
])
def test_regular_user_cannot_mutate_catalog_even_with_invalid_payload(method, path):
    app = FastAPI()
    app.include_router(ml_models.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7, is_admin=False)

    async def database():
        yield SimpleNamespace()

    app.dependency_overrides[get_db] = database
    with TestClient(app) as client:
        response = getattr(client, method)(path)
    assert response.status_code == 403


@pytest.mark.parametrize("method,path,status", [
    ("post", "/api/models/upload", 503),
    ("delete", "/api/models/1", 405),
])
def test_legacy_asset_mutations_are_disabled_for_admins_too(method, path, status):
    app = FastAPI()
    app.include_router(ml_models.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7, is_admin=True)
    with TestClient(app) as client:
        response = getattr(client, method)(path)
    assert response.status_code == status
