import asyncio
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from config.db_config import get_db
from crud import ml_models as crud_ml_models
from router import ml_models
from utils.get_user_by_token import get_current_user


async def _override_db():
    yield SimpleNamespace()


def _override_user():
    return SimpleNamespace(id=7)


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(ml_models.router)
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_user
    return TestClient(app)


def _model():
    return SimpleNamespace(
        id=11,
        model_name="土地识别",
        model_type="semantic_segmentation",
        framework="PyTorch",
        weight_file_path="weights/model.pth",
        model_file_path="models/model.py",
        description="test",
        updated_time=datetime(2026, 8, 30),
    )


def test_model_list_accepts_optional_type_and_framework_filters():
    fetch = AsyncMock(return_value=([_model()], 1))
    with patch.object(ml_models, "get_ml_models", new=fetch):
        response = _make_client().get(
            "/api/models/list",
            params={
                "modelType": "semantic_segmentation",
                "framework": "PyTorch",
            },
        )

    assert response.status_code == 200
    assert response.json()["data"]["total"] == 1
    assert fetch.await_args.args[1:] == (
        7,
        1,
        10,
        "",
        "semantic_segmentation",
        "PyTorch",
    )


def test_get_ml_models_applies_type_and_framework_filters():
    db = SimpleNamespace(
        scalar=AsyncMock(return_value=1),
        execute=AsyncMock(
            return_value=SimpleNamespace(
                scalars=lambda: SimpleNamespace(all=lambda: [_model()]),
            ),
        ),
    )

    items, total = asyncio.run(
        crud_ml_models.get_ml_models(
            db,
            7,
            1,
            10,
            model_type="semantic_segmentation",
            framework="PyTorch",
        )
    )

    assert items == [_model()]
    assert total == 1
    sql = str(
        db.execute.await_args.args[0].compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )
    assert "model_library.model_type = 'semantic_segmentation'" in sql
    assert "model_library.framework = 'PyTorch'" in sql
