"""Exercise the migration and HTTP contract in a disposable PostgreSQL schema."""
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient
import psycopg2
from psycopg2 import sql
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

load_dotenv()
from config.db_config import get_db
from router import ml_models
from utils.get_user_by_token import get_current_user

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_POSTGRES_INTEGRATION") != "1",
    reason="需要显式启用隔离 PostgreSQL 集成测试",
)
MIGRATION = Path("migrations/010_system_model_catalog.sql")


@pytest.fixture
def catalog():
    schema = "catalog_55_" + uuid4().hex
    connection = psycopg2.connect(os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://", 1))
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            cursor.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
            cursor.execute("""
                CREATE TABLE user_info (id BIGINT PRIMARY KEY);
                INSERT INTO user_info VALUES (1), (2);
                CREATE TABLE model_library (id INTEGER PRIMARY KEY, model_name TEXT);
                INSERT INTO model_library VALUES (10, '旧模型');
                CREATE TABLE classification_results (
                    id TEXT PRIMARY KEY,
                    source_model_id INTEGER REFERENCES model_library(id) ON DELETE SET NULL,
                    source_snapshot JSON, classes_path TEXT
                );
                CREATE TABLE change_results (
                    id INTEGER PRIMARY KEY,
                    source_model_id INTEGER REFERENCES model_library(id) ON DELETE SET NULL,
                    before_snapshot JSON, after_snapshot JSON
                );
                INSERT INTO classification_results VALUES
                    ('history', 10, '{"model":{"id":10,"name":"历史模型名称"}}', 'retained.tif');
                INSERT INTO change_results VALUES
                    (1, 10, '{"identity_sha256":"before"}', '{"identity_sha256":"after"}');
            """)
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        engine = create_async_engine(
            os.environ["DATABASE_URL"], poolclass=NullPool,
            connect_args={"server_settings": {"search_path": schema}},
        )
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        app = FastAPI()
        app.include_router(ml_models.router)

        async def database():
            async with sessions() as session:
                yield session

        app.dependency_overrides[get_db] = database
        user = SimpleNamespace(id=1, is_admin=False)
        app.dependency_overrides[get_current_user] = lambda: user
        with TestClient(app) as client:
            yield connection, client, user
    finally:
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        connection.close()


def test_migration_clears_only_legacy_catalog_and_preserves_history(catalog):
    connection, _, _ = catalog
    with connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM model_library")
        assert cursor.fetchone() == (0,)
        cursor.execute("SELECT source_model_id, source_snapshot, classes_path FROM classification_results")
        assert cursor.fetchone() == (None, {"model": {"id": 10, "name": "历史模型名称"}}, "retained.tif")
        cursor.execute("SELECT source_model_id, before_snapshot, after_snapshot FROM change_results")
        assert cursor.fetchone() == (None, {"identity_sha256": "before"}, {"identity_sha256": "after"})
        cursor.execute("SELECT is_admin FROM user_info ORDER BY id")
        assert cursor.fetchall() == [(False,), (False,)]
        cursor.execute("SELECT count(*) FROM system_models")
        assert cursor.fetchone() == (0,)


def test_catalog_is_global_and_filters_before_pagination(catalog):
    connection, client, user = catalog
    with connection.cursor() as cursor:
        cursor.execute("""
            INSERT INTO system_models (model_name, model_type, description, tags)
            VALUES ('土地 A', 'land-cover', '土地模型', '["测试"]'),
                   ('变化 B', 'change-detection', '双期模型', '[]'),
                   ('土地 C', 'land-cover', NULL, '[]');
        """)
    response = client.get("/api/models/list", params={"modelType": "land-cover", "pageSize": 1})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == 2
    assert len(data["items"]) == 1
    assert data["items"][0]["model_type"] == "land-cover"
    assert data["can_manage"] is False
    user.id = 2
    assert client.get("/api/models/list", params={"modelType": "land-cover", "pageSize": 1}).json()["data"] == data
    second = client.get("/api/models/list", params={"modelType": "land-cover", "pageSize": 1, "page": 2}).json()["data"]
    assert second["items"][0]["id"] != data["items"][0]["id"]
    filtered = client.get("/api/models/list", params={"keyword": "变化"}).json()["data"]
    assert filtered["total"] == 1
    assert filtered["items"][0]["model_name"] == "变化 B"


def seed_version(connection):
    with connection.cursor() as cursor:
        cursor.execute("INSERT INTO system_models (model_name, model_type) VALUES ('生产模型', 'land-cover') RETURNING id")
        model_id = cursor.fetchone()[0]
        cursor.execute("""
            INSERT INTO system_model_versions (
                model_id, version, status, framework, input_description, input_bands,
                resource_requirements, code_config_sha256, weight_sha256, image_digest
            ) VALUES (%s, 'v1', 'AVAILABLE', 'PyTorch', '单幅 RGB GeoTIFF', '[1,2,3]',
                      '{"cpu_cores":2,"memory_mb":4096,"gpu_count":1,"internal_path":"private"}',
                      %s, %s, %s) RETURNING id
        """, (model_id, "a" * 64, "b" * 64, "sha256:" + "c" * 64))
        version_id = cursor.fetchone()[0]
        cursor.execute("UPDATE system_models SET production_version_id=%s WHERE id=%s", (version_id, model_id))
    return model_id, version_id


def test_admin_edits_metadata_without_changing_version_or_exposing_assets(catalog):
    connection, client, user = catalog
    model_id, version_id = seed_version(connection)
    public = client.get("/api/models/list").json()["data"]["items"][0]
    assert public["version"] == {
        "id": version_id, "version": "v1", "status": "AVAILABLE", "framework": "PyTorch",
        "input_description": "单幅 RGB GeoTIFF", "input_bands": [1, 2, 3],
        "resource_requirements": {"cpu_cores": 2, "memory_mb": 4096, "gpu_count": 1},
    }
    assert public["business_capabilities"] == []
    assert "sha256" not in str(public)
    assert "private" not in str(public)
    payload = {"model_name": "新名称", "description": "", "tags": ["生产", "RGB"]}
    assert client.put(f"/api/models/{model_id}", json=payload).status_code == 403
    user.is_admin = True
    assert client.put(f"/api/models/{model_id}", json=payload).status_code == 200
    result = client.get("/api/models/list").json()["data"]
    assert result["can_manage"] is True
    updated = result["items"][0]
    assert updated["model_name"] == "新名称"
    assert updated["description"] == ""
    assert updated["tags"] == ["生产", "RGB"]
    assert updated["version"] == public["version"]
    assert updated["production_version_id"] == version_id
    with connection.cursor() as cursor:
        cursor.execute("SELECT code_config_sha256, weight_sha256, image_digest FROM system_model_versions WHERE id=%s", (version_id,))
        assert cursor.fetchone() == ("a" * 64, "b" * 64, "sha256:" + "c" * 64)


@pytest.mark.parametrize("field,value", [
    ("model_type", "change-detection"), ("production_version_id", 123),
    ("weight_sha256", "d" * 64), ("weight_file_path", "replacement.pth"),
    ("version", "v2"), ("user_id", 2),
])
def test_admin_cannot_mutate_identity_through_metadata_endpoint(catalog, field, value):
    connection, client, user = catalog
    model_id, _ = seed_version(connection)
    user.is_admin = True
    response = client.put(f"/api/models/{model_id}", json={
        "model_name": "新名称", "description": None, "tags": [], field: value,
    })
    assert response.status_code == 422
    assert client.get("/api/models/list").json()["data"]["items"][0]["model_name"] == "生产模型"


def test_migration_can_repeat_without_erasing_new_catalog_and_enforces_version_owner(catalog):
    connection, client, _ = catalog
    model_id, version_id = seed_version(connection)
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        cursor.execute("INSERT INTO system_models (model_name, model_type) VALUES ('另一模型', 'change-detection') RETURNING id")
        other_id = cursor.fetchone()[0]
        with pytest.raises(psycopg2.errors.ForeignKeyViolation):
            cursor.execute("UPDATE system_models SET production_version_id=%s WHERE id=%s", (version_id, other_id))
        with pytest.raises(psycopg2.errors.CheckViolation):
            cursor.execute("INSERT INTO system_models (model_name, model_type) VALUES ('非法类型', 'semantic_segmentation')")
        with pytest.raises(psycopg2.errors.UniqueViolation):
            cursor.execute("""
                INSERT INTO system_model_versions (model_id, version, status, framework,
                    input_description, code_config_sha256, weight_sha256)
                VALUES (%s, 'v1', 'BUILDING', 'PyTorch', 'RGB', %s, %s)
            """, (model_id, "a" * 64, "b" * 64))
    data = client.get("/api/models/list").json()["data"]
    assert data["total"] == 2
    assert any(item["id"] == model_id and item["production_version_id"] == version_id for item in data["items"])
