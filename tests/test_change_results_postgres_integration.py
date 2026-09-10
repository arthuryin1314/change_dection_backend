import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import psycopg2
import pytest
from affine import Affine
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg2 import sql

load_dotenv()

from config.db_config import get_db
from router import change_results
from services.identification_results import ResultIdentity
from utils.classification_contract import inference_parameters
from utils.classification_storage import RasterGrid, write_classification_result
from utils.get_user_by_token import get_current_user


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_POSTGRES_INTEGRATION") != "1",
    reason="需要显式启用本机 PostgreSQL 集成测试",
)


def _database_url() -> str:
    return os.environ["DATABASE_URL"].replace(
        "postgresql+asyncpg://",
        "postgresql://",
        1,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _apply_migration() -> None:
    connection = psycopg2.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute(Path("migrations/003_change_results.sql").read_text(encoding="utf-8"))
            cursor.execute(Path("migrations/004_change_result_analysis.sql").read_text(encoding="utf-8"))
        connection.commit()
    finally:
        connection.close()


def _seed_result(
    connection,
    *,
    result_id: str,
    user_id: int,
    image_id: int,
    model_id: int,
    image_path: Path,
    weight_path: Path,
    stored,
    grid: RasterGrid,
) -> str:
    image_sha256 = _sha256(image_path)
    weight_sha256 = _sha256(weight_path)
    identity = ResultIdentity(
        user_id=user_id,
        image_content_sha256=image_sha256,
        weight_content_sha256=weight_sha256,
        inference_parameters=inference_parameters(),
    )
    now = datetime.now(timezone.utc)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE images
            SET content_sha256 = %s,
                content_sha256_size = %s,
                content_sha256_mtime_ns = %s
            WHERE id = %s
            """,
            (image_sha256, image_path.stat().st_size, image_path.stat().st_mtime_ns, image_id),
        )
        cursor.execute(
            """
            INSERT INTO classification_results (
                id, user_id, source_image_id, source_model_id,
                identity_sha256, image_content_sha256, weight_content_sha256,
                inference_parameters, classification_scheme_version,
                pipeline_version, grid_policy_version, status,
                started_at, heartbeat_at, lease_expires_at, lease_owner,
                completed_at, classes_path, valid_mask_path, crs, transform,
                raster_width, raster_height, resolution, bounds, area_status
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s::json,
                %s, %s, %s, 'SUCCEEDED', %s, %s, %s, %s, %s,
                %s, %s, %s, %s::json, %s, %s, %s::json, %s::json,
                'NOT_COMPUTED'
            )
            """,
            (
                result_id,
                user_id,
                image_id,
                model_id,
                identity.sha256(),
                image_sha256,
                weight_sha256,
                json.dumps(dict(identity.inference_parameters)),
                identity.classification_scheme_version,
                identity.pipeline_version,
                identity.grid_policy_version,
                now,
                now,
                now,
                f"integration-{result_id}",
                now,
                str(stored.classes_path),
                str(stored.valid_mask_path),
                str(grid.crs),
                json.dumps(list(grid.transform)[:6]),
                grid.width,
                grid.height,
                json.dumps([abs(grid.transform.a), abs(grid.transform.e)]),
                json.dumps(list(stored_bounds(grid))),
            ),
        )
    return identity.sha256()


def stored_bounds(grid: RasterGrid):
    left = grid.transform.c
    top = grid.transform.f
    right = left + grid.width * grid.transform.a
    bottom = top + grid.height * grid.transform.e
    return left, bottom, right, top


def _seed(tmp_path: Path):
    _apply_migration()
    suffix = uuid4().hex
    weight_path = tmp_path / "weights.pth"
    weight_path.write_bytes(b"change-integration-weight")
    weight_path.with_suffix(".py").write_text("# fixture", encoding="utf-8")
    grid = RasterGrid(
        width=4,
        height=2,
        crs="EPSG:4326",
        transform=Affine(0.01, 0, 110, 0, -0.01, 30),
    )
    before_source = tmp_path / "before-source.tif"
    after_source = tmp_path / "after-source.tif"
    before_source.write_bytes(b"before-source")
    after_source.write_bytes(b"after-source")
    before_stored = write_classification_result(
        tmp_path / "classification",
        "before-result",
        grid,
        np.array([[0, 1, 2, 3], [4, 5, 0, 1]], dtype=np.uint8),
        np.ones((2, 4), dtype=np.uint8),
    )
    after_stored = write_classification_result(
        tmp_path / "classification",
        "after-result",
        grid,
        np.array([[1, 1, 2, 0], [4, 0, 5, 1]], dtype=np.uint8),
        np.ones((2, 4), dtype=np.uint8),
    )

    connection = psycopg2.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_info (username, phone, password)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (f"change-{suffix}", f"8{suffix[:10]}", "not-a-login-secret"),
            )
            user_id = cursor.fetchone()[0]
            cursor.execute(
                """
                INSERT INTO model_library (
                    user_id, model_name, model_type, framework,
                    weight_file_path, model_file_path, description
                )
                VALUES (%s, %s, 'semantic_segmentation', 'PyTorch', %s, %s, %s)
                RETURNING id
                """,
                (
                    user_id,
                    f"change-{suffix}",
                    str(weight_path),
                    str(weight_path.with_suffix(".py")),
                    "PostgreSQL fixture",
                ),
            )
            model_id = cursor.fetchone()[0]
            cursor.execute(
                """
                UPDATE model_library
                SET weight_content_sha256 = %s,
                    weight_content_sha256_size = %s,
                    weight_content_sha256_mtime_ns = %s
                WHERE id = %s
                """,
                (_sha256(weight_path), weight_path.stat().st_size, weight_path.stat().st_mtime_ns, model_id),
            )
            image_ids = []
            for name, source in (("before", before_source), ("after", after_source)):
                cursor.execute(
                    """
                    INSERT INTO images (user_id, image_name, img_path)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (user_id, f"{name}-{suffix}", str(source)),
                )
                image_ids.append(cursor.fetchone()[0])

        before_identity = _seed_result(
            connection,
            result_id=f"before{suffix[:20]}",
            user_id=user_id,
            image_id=image_ids[0],
            model_id=model_id,
            image_path=before_source,
            weight_path=weight_path,
            stored=before_stored,
            grid=grid,
        )
        after_identity = _seed_result(
            connection,
            result_id=f"after{suffix[:21]}",
            user_id=user_id,
            image_id=image_ids[1],
            model_id=model_id,
            image_path=after_source,
            weight_path=weight_path,
            stored=after_stored,
            grid=grid,
        )
        connection.commit()
        return {
            "user_id": user_id,
            "model_id": model_id,
            "before_image_id": image_ids[0],
            "after_image_id": image_ids[1],
            "before_result_id": f"before{suffix[:20]}",
            "after_result_id": f"after{suffix[:21]}",
            "before_identity_sha256": before_identity,
            "after_identity_sha256": after_identity,
        }
    finally:
        connection.close()


def _cleanup(user_id: int) -> None:
    connection = psycopg2.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM user_info WHERE id = %s", (user_id,))
        connection.commit()
    finally:
        connection.close()


def test_migration_004_backfills_legacy_rows_and_is_idempotent():
    schema_name = f"change_migration_{uuid4().hex}"
    connection = psycopg2.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name))
            )
            cursor.execute(
                sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name))
            )
            cursor.execute("CREATE TABLE user_info (id BIGINT PRIMARY KEY)")
            cursor.execute("CREATE TABLE images (id INTEGER PRIMARY KEY)")
            cursor.execute("CREATE TABLE model_library (id INTEGER PRIMARY KEY)")
            cursor.execute("CREATE TABLE classification_results (id VARCHAR(32) PRIMARY KEY)")
            cursor.execute(Path("migrations/003_change_results.sql").read_text(encoding="utf-8"))
            cursor.execute("INSERT INTO user_info (id) VALUES (1)")
            cursor.execute("INSERT INTO images (id) VALUES (1), (2)")
            cursor.execute("INSERT INTO model_library (id) VALUES (1)")
            cursor.execute(
                "INSERT INTO classification_results (id) VALUES ('before'), ('after')"
            )
            cursor.execute(
                """
                INSERT INTO change_results (
                    request_id, user_id, before_image_id, after_image_id,
                    source_model_id, before_result_id, after_result_id,
                    before_identity_sha256, after_identity_sha256,
                    status, lease_owner, started_at, heartbeat_at
                )
                VALUES (
                    %s, 1, 1, 2, 1, 'before', 'after', %s, %s,
                    'SUCCEEDED', 'legacy-worker', NOW(), NOW()
                )
                """,
                (str(uuid4()), "a" * 64, "b" * 64),
            )

            migration = Path("migrations/004_change_result_analysis.sql").read_text(
                encoding="utf-8"
            )
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(
                """
                SELECT calculation_version, grid_policy_version,
                       analysis_identity_sha256, analysis_metadata
                FROM change_results
                """
            )
            assert cursor.fetchone() == (
                "transition-matrix-v1",
                "same-grid-v0",
                None,
                None,
            )
            cursor.execute(
                """
                SELECT column_name, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = 'change_results'
                  AND column_name IN ('calculation_version', 'grid_policy_version')
                """,
                (schema_name,),
            )
            assert dict(cursor.fetchall()) == {
                "calculation_version": "NO",
                "grid_policy_version": "NO",
            }
    finally:
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("SET search_path TO public")
            cursor.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema_name)
                )
            )
        connection.commit()
        connection.close()


def test_postgres_change_api_persists_reloads_deduplicates_and_authorizes(tmp_path):
    seeded = _seed(tmp_path)
    request_id = str(uuid4())
    payload = {
        "request_id": request_id,
        "before_image_id": seeded["before_image_id"],
        "after_image_id": seeded["after_image_id"],
        "model_id": seeded["model_id"],
        "before_result_id": seeded["before_result_id"],
        "before_identity_sha256": seeded["before_identity_sha256"],
        "after_result_id": seeded["after_result_id"],
        "after_identity_sha256": seeded["after_identity_sha256"],
    }
    app = FastAPI()
    app.include_router(change_results.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=seeded["user_id"])

    try:
        with TestClient(app) as client:
            created = client.post("/api/change-results", json=payload)
            assert created.status_code == 200
            data = created.json()["data"]
            assert data["status"] == "SUCCEEDED"
            assert len(data["matrix_m2"]) == 6
            assert data["matrix_m2"][0][1] > 0
            assert data["matrix_m2"][1][1] > 0
            assert data["common_valid_area_m2"] == pytest.approx(
                sum(sum(row) for row in data["matrix_m2"]),
                rel=1e-12,
            )
            assert data["analysis"]["calculation_version"] == "transition-matrix-v2"
            assert data["analysis"]["grid_policy_version"] == "aligned-grid-v1"
            assert len(data["analysis"]["identity_sha256"]) == 64
            assert data["analysis"]["alignment_mode"] == "DIRECT"

            reloaded_client = client
            fetched = reloaded_client.get(f"/api/change-results/{request_id}")
            assert fetched.status_code == 200
            assert fetched.json()["data"] == data

            duplicate = reloaded_client.post("/api/change-results", json=payload)
            assert duplicate.status_code == 200
            assert duplicate.json()["data"] == data

            conflicting = reloaded_client.post(
                "/api/change-results",
                json={**payload, "after_image_id": seeded["before_image_id"]},
            )
            assert conflicting.status_code == 409
            assert conflicting.json()["data"]["error_code"] == "REQUEST_ID_CONFLICT"

            app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
                id=seeded["user_id"] + 1_000_000
            )
            denied = reloaded_client.get(f"/api/change-results/{request_id}")
            assert denied.status_code == 404
    finally:
        _cleanup(seeded["user_id"])
