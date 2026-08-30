import asyncio
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.testclient import TestClient
import pytest

from config.db_config import get_db
from router import ml_models
from utils.model_asset_storage import ModelAssetTooLargeError, save_upload_file
from utils.get_user_by_token import get_current_user


class FakeDB:
    async def commit(self):
        return None

    async def rollback(self):
        return None


class FailingCommitDB:
    def __init__(self, record, old_weight, old_model):
        self.record = record
        self.old_weight = old_weight
        self.old_model = old_model
        self.rollback_called = False

    async def commit(self):
        raise RuntimeError("commit failed")

    async def rollback(self):
        self.rollback_called = True
        self.record.weight_file_path = str(self.old_weight)
        self.record.model_file_path = str(self.old_model)


async def _override_db():
    yield FakeDB()


def _override_user():
    return SimpleNamespace(id=7)


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(ml_models.router)
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_user
    return TestClient(app)


def _make_client_with_db(db) -> TestClient:
    app = FastAPI()
    app.include_router(ml_models.router)

    async def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = _override_user
    return TestClient(app)


def test_creating_same_original_names_keeps_two_real_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_models, "BASE_DIR", tmp_path)
    records = []

    async def create_record(**kwargs):
        record = SimpleNamespace(
            id=len(records) + 1,
            user_id=kwargs["user_id"],
            model_name=kwargs["model_name"],
            model_type=kwargs["model_type"],
            framework=kwargs["framework"],
            weight_file_path=kwargs["weight_file_path"],
            model_file_path=kwargs["model_file_path"],
            description=kwargs["description"],
            upload_time=datetime.now(),
            updated_time=datetime.now(),
        )
        records.append(record)
        return record

    client = _make_client()
    original_name = "weights.pth"
    with monkeypatch.context() as patch:
        patch.setattr(ml_models, "create_ml_model", create_record)
        first = client.post(
            "/api/models/upload",
            data={
                "model_name": "模型一",
                "model_type": "semantic_segmentation",
                "framework": "PyTorch",
                "description": "第一份模型描述",
            },
            files={
                "weight": (original_name, b"first-weight", "application/octet-stream"),
                "model_file": ("model.py", b"print('first')", "text/plain"),
            },
        )
        second = client.post(
            "/api/models/upload",
            data={
                "model_name": "模型二",
                "model_type": "change_detection",
                "framework": "ONNX",
                "description": "第二份模型描述",
            },
            files={
                "weight": (original_name, b"second-weight", "application/octet-stream"),
                "model_file": ("model.py", b"print('second')", "text/plain"),
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    first_path = Path(first.json()["data"]["weight_file_path"])
    second_path = Path(second.json()["data"]["weight_file_path"])
    assert first_path != second_path
    assert first_path.name == original_name
    assert second_path.name == original_name
    assert first_path.read_bytes() == b"first-weight"
    assert second_path.read_bytes() == b"second-weight"


def test_replacing_same_weight_name_keeps_new_file_and_removes_old(tmp_path, monkeypatch):
    old_weight = tmp_path / "old" / "weights.pth"
    old_model = tmp_path / "old" / "model.py"
    old_weight.parent.mkdir()
    old_weight.write_bytes(b"old-weight")
    old_model.write_bytes(b"old-model")
    record = SimpleNamespace(
        id=11,
        user_id=7,
        model_name="原模型",
        model_type="semantic_segmentation",
        framework="PyTorch",
        weight_file_path=str(old_weight),
        model_file_path=str(old_model),
        description="原模型描述",
    )

    monkeypatch.setattr(ml_models, "BASE_DIR", tmp_path / "root")
    monkeypatch.setattr(
        ml_models,
        "get_ml_model_by_id",
        AsyncMock(return_value=record),
    )

    async def update_record(db, model_id, user_id, **kwargs):
        for key, value in kwargs.items():
            setattr(record, key, value)
        return record

    monkeypatch.setattr(ml_models, "update_ml_model", update_record)

    response = _make_client().put(
        "/api/models/11",
        files={"weight": ("weights.pth", b"new-weight", "application/octet-stream")},
    )

    assert response.status_code == 200
    new_weight = Path(record.weight_file_path)
    assert new_weight != old_weight
    assert new_weight.name == old_weight.name
    assert new_weight.read_bytes() == b"new-weight"
    assert not old_weight.exists()
    assert Path(record.model_file_path) == old_model
    assert old_model.read_bytes() == b"old-model"


def test_metadata_update_without_replacement_does_not_write_files(tmp_path, monkeypatch):
    old_weight = tmp_path / "old" / "weights.pth"
    old_model = tmp_path / "old" / "model.py"
    old_weight.parent.mkdir()
    old_weight.write_bytes(b"old-weight")
    old_model.write_bytes(b"old-model")
    record = SimpleNamespace(
        id=11,
        user_id=7,
        model_name="原模型",
        model_type="semantic_segmentation",
        framework="PyTorch",
        weight_file_path=str(old_weight),
        model_file_path=str(old_model),
        description="原模型描述",
    )

    monkeypatch.setattr(ml_models, "BASE_DIR", tmp_path / "root")
    monkeypatch.setattr(
        ml_models,
        "get_ml_model_by_id",
        AsyncMock(return_value=record),
    )

    async def update_record(db, model_id, user_id, **kwargs):
        for key, value in kwargs.items():
            setattr(record, key, value)
        return record

    async def unexpected_write(*args, **kwargs):
        raise AssertionError("metadata-only update must not write an asset")

    monkeypatch.setattr(ml_models, "update_ml_model", update_record)
    monkeypatch.setattr(ml_models, "save_upload_file", unexpected_write)

    response = _make_client().put(
        "/api/models/11",
        data={"model_name": "新模型"},
    )

    assert response.status_code == 200
    assert record.model_name == "新模型"
    assert Path(record.weight_file_path) == old_weight
    assert Path(record.model_file_path) == old_model
    assert old_weight.read_bytes() == b"old-weight"
    assert old_model.read_bytes() == b"old-model"


def test_commit_failure_keeps_old_assets_and_cleans_new_assets(tmp_path, monkeypatch):
    old_weight = tmp_path / "old" / "weights.pth"
    old_model = tmp_path / "old" / "model.py"
    old_weight.parent.mkdir()
    old_weight.write_bytes(b"old-weight")
    old_model.write_bytes(b"old-model")
    record = SimpleNamespace(
        id=11,
        user_id=7,
        model_name="原模型",
        model_type="semantic_segmentation",
        framework="PyTorch",
        weight_file_path=str(old_weight),
        model_file_path=str(old_model),
        description="原模型描述",
    )
    db = FailingCommitDB(record, old_weight, old_model)
    new_paths = {}

    monkeypatch.setattr(ml_models, "BASE_DIR", tmp_path / "root")
    monkeypatch.setattr(
        ml_models,
        "get_ml_model_by_id",
        AsyncMock(return_value=record),
    )

    async def update_record(db, model_id, user_id, **kwargs):
        new_paths.update(
            weight=Path(kwargs["weight_file_path"]),
            model=Path(kwargs["model_file_path"]),
        )
        for key, value in kwargs.items():
            setattr(record, key, value)
        return record

    monkeypatch.setattr(ml_models, "update_ml_model", update_record)

    response = _make_client_with_db(db).put(
        "/api/models/11",
        files={
            "weight": ("weights.pth", b"new-weight", "application/octet-stream"),
            "model_file": ("model.py", b"new-model", "text/plain"),
        },
    )

    assert response.status_code == 500
    assert db.rollback_called
    assert record.weight_file_path == str(old_weight)
    assert record.model_file_path == str(old_model)
    assert old_weight.read_bytes() == b"old-weight"
    assert old_model.read_bytes() == b"old-model"
    assert not new_paths["weight"].exists()
    assert not new_paths["model"].exists()
    asset_user_dir = tmp_path / "root" / "uploads" / "model_assets" / "7"
    assert list(asset_user_dir.iterdir()) == []


def test_streamed_size_limit_cleans_partial_asset(tmp_path):
    upload = UploadFile(
        file=BytesIO(b"12345"),
        filename="weights.pth",
        size=None,
    )

    with pytest.raises(ModelAssetTooLargeError):
        asyncio.run(save_upload_file(upload, tmp_path / "assets", 7, 4))

    assert list((tmp_path / "assets" / "7").iterdir()) == []


def test_invalid_extension_is_rejected_for_create_and_edit(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_models, "BASE_DIR", tmp_path / "root")
    create = _make_client().post(
        "/api/models/upload",
        data={
            "model_name": "模型一",
            "model_type": "semantic_segmentation",
            "framework": "PyTorch",
            "description": "创建模型描述",
        },
        files={
            "weight": ("weights.txt", b"not-a-weight", "text/plain"),
            "model_file": ("model.py", b"print(1)", "text/plain"),
        },
    )

    record = SimpleNamespace(
        id=11,
        user_id=7,
        model_name="原模型",
        model_type="semantic_segmentation",
        framework="PyTorch",
        weight_file_path=str(tmp_path / "old" / "weights.pth"),
        model_file_path=str(tmp_path / "old" / "model.py"),
        description="原模型描述",
    )
    monkeypatch.setattr(
        ml_models,
        "get_ml_model_by_id",
        AsyncMock(return_value=record),
    )
    edit = _make_client().put(
        "/api/models/11",
        files={"model_file": ("model.txt", b"not-a-model", "text/plain")},
    )

    assert create.status_code == 400
    assert edit.status_code == 400
    assert not (tmp_path / "root").exists()


def test_advertised_size_limit_rejects_before_any_asset_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_models, "BASE_DIR", tmp_path / "root")
    monkeypatch.setattr(ml_models, "MAX_WEIGHT_SIZE", 4)

    response = _make_client().post(
        "/api/models/upload",
        data={
            "model_name": "模型一",
            "model_type": "semantic_segmentation",
            "framework": "PyTorch",
            "description": "创建模型描述",
        },
        files={
            "weight": ("weights.pth", b"12345", "application/octet-stream"),
            "model_file": ("model.py", b"print(1)", "text/plain"),
        },
    )

    assert response.status_code == 400
    assert not (tmp_path / "root").exists()


def test_create_rejects_invalid_metadata_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_models, "BASE_DIR", tmp_path / "root")
    base_data = {
        "model_name": "模型一",
        "model_type": "semantic_segmentation",
        "framework": "PyTorch",
        "description": "创建模型描述",
    }
    invalid_values = (
        ("model_name", "x"),
        ("model_type", "unknown"),
        ("framework", "Unknown"),
        ("description", "短"),
    )

    for field, value in invalid_values:
        data = {**base_data, field: value}
        response = _make_client().post(
            "/api/models/upload",
            data=data,
            files={
                "weight": ("weights.pth", b"weight", "application/octet-stream"),
                "model_file": ("model.py", b"print(1)", "text/plain"),
            },
        )
        assert response.status_code == 400

    assert not (tmp_path / "root").exists()


def test_edit_allows_clearing_description(tmp_path, monkeypatch):
    record = SimpleNamespace(
        id=11,
        user_id=7,
        model_name="原模型",
        model_type="semantic_segmentation",
        framework="PyTorch",
        weight_file_path=str(tmp_path / "old" / "weights.pth"),
        model_file_path=str(tmp_path / "old" / "model.py"),
        description="原模型描述",
    )
    monkeypatch.setattr(
        ml_models,
        "get_ml_model_by_id",
        AsyncMock(return_value=record),
    )

    async def update_record(db, model_id, user_id, **kwargs):
        for key, value in kwargs.items():
            setattr(record, key, value)
        return record

    monkeypatch.setattr(ml_models, "update_ml_model", update_record)

    response = _make_client().put(
        "/api/models/11",
        data={"description": ""},
    )

    assert response.status_code == 200
    assert record.description == ""
