import asyncio
import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import psycopg2
import pytest
import rasterio
from affine import Affine
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg2 import sql

load_dotenv()

from config.db_config import AsyncSessionLocal, async_engine, get_db
from crud import change_results as change_result_crud
from router import change_results
from services import classification_generation
from services.identification_results import ResultIdentity
from utils.classification_contract import inference_parameters
from utils.classification_storage import RasterGrid, write_classification_result
from utils.get_user_by_token import get_current_user


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_POSTGRES_INTEGRATION") != "1",
    reason="需要显式启用本机 PostgreSQL 集成测试",
)


@pytest.fixture(autouse=True)
def _isolate_async_connection_pool():
    asyncio.run(async_engine.dispose(close=False))
    yield
    asyncio.run(async_engine.dispose(close=False))


def _database_url() -> str:
    return os.environ["DATABASE_URL"].replace(
        "postgresql+asyncpg://",
        "postgresql://",
        1,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(path: Path, value: int, grid: RasterGrid) -> None:
    data = np.full((3, grid.height, grid.width), value, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=grid.width,
        height=grid.height,
        count=3,
        dtype="uint8",
        crs=grid.crs,
        transform=grid.transform,
        nodata=0,
    ) as dataset:
        dataset.write(data)


@contextmanager
def _fake_predictor(weight_file_path, *, weight_sha256):
    def predict(rgb):
        return np.where(rgb[:, :, 0] % 2 == 0, 2, 1).astype(np.uint8)

    yield predict


def _apply_migration() -> None:
    connection = psycopg2.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute(Path("migrations/003_change_results.sql").read_text(encoding="utf-8"))
            cursor.execute(Path("migrations/004_change_result_analysis.sql").read_text(encoding="utf-8"))
            cursor.execute(Path("migrations/005_change_orchestration.sql").read_text(encoding="utf-8"))
            cursor.execute(Path("migrations/006_result_history.sql").read_text(encoding="utf-8"))
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
                completed_at, created_at, classes_path, valid_mask_path, crs, transform,
                raster_width, raster_height, resolution, bounds, area_status
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s::json,
                %s, %s, %s, 'SUCCEEDED', %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s::json, %s, %s, %s::json, %s::json,
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
    _write_source(before_source, 7, grid)
    _write_source(after_source, 8, grid)
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
            "before_source": before_source,
            "after_source": after_source,
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


def _delete_cached_periods(seeded: dict, periods: tuple[str, ...]) -> None:
    connection = psycopg2.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            for period in periods:
                cursor.execute(
                    "DELETE FROM classification_results WHERE id = %s",
                    (seeded[f"{period}_result_id"],),
                )
        connection.commit()
    finally:
        connection.close()


def _mark_period_processing(seeded: dict, period: str) -> None:
    connection = psycopg2.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE classification_results
                SET status = 'PROCESSING',
                    completed_at = NULL,
                    heartbeat_at = NOW(),
                    lease_expires_at = NOW() + INTERVAL '5 minutes'
                WHERE id = %s
                """,
                (seeded[f"{period}_result_id"],),
            )
        connection.commit()
    finally:
        connection.close()


def _wait_for_terminal(client: TestClient, request_id: str, timeout: float = 10):
    deadline = time.monotonic() + timeout
    while True:
        response = client.get(f"/api/change-results/{request_id}")
        if response.status_code != 202:
            return response
        assert time.monotonic() < deadline
        time.sleep(0.05)


def test_migrations_004_and_005_backfill_legacy_rows_and_are_idempotent():
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
            orchestration_migration = Path(
                "migrations/005_change_orchestration.sql"
            ).read_text(encoding="utf-8")
            cursor.execute(orchestration_migration)
            cursor.execute(orchestration_migration)
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
            cursor.execute(
                """
                SELECT phase, submitted_inputs->>'before_image_id',
                       submitted_inputs->>'after_image_id'
                FROM change_results
                """
            )
            assert cursor.fetchone() == ("SUCCEEDED", "1", "2")
            cursor.execute("SELECT request_fingerprint FROM change_requests")
            assert cursor.fetchone()[0].startswith("legacy:")
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
    }
    app = FastAPI()
    app.include_router(change_results.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=seeded["user_id"])

    try:
        with TestClient(app) as client:
            created = client.post("/api/change-results", json=payload)
            assert created.status_code == 202
            deadline = time.monotonic() + 10
            while True:
                fetched = client.get(f"/api/change-results/{request_id}")
                if fetched.status_code == 200:
                    break
                assert fetched.status_code == 202
                assert time.monotonic() < deadline
                time.sleep(0.05)
            data = fetched.json()["data"]
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
            assert data["before"]["source"]["image"]["name"].startswith("before-")
            assert data["after"]["source"]["image"]["name"].startswith("after-")
            assert data["before"]["source"]["model"]["name"].startswith("change-")

            reloaded_client = client
            fetched = reloaded_client.get(f"/api/change-results/{request_id}")
            assert fetched.status_code == 200
            assert fetched.json()["data"] == data

            duplicate = reloaded_client.post("/api/change-results", json=payload)
            assert duplicate.status_code == 202
            assert duplicate.json()["data"] == data

            history = reloaded_client.get("/api/change-results/history")
            assert history.status_code == 200
            assert history.json()["data"]["total"] == 1
            assert history.json()["data"]["items"][0]["result_id"] == data["result_id"]
            history_detail = reloaded_client.get(
                f"/api/change-results/history/{data['result_id']}"
            )
            assert history_detail.status_code == 200
            assert history_detail.json()["data"] == data

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
            denied_history = reloaded_client.get(
                f"/api/change-results/history/{data['result_id']}"
            )
            assert denied_history.status_code == 404
    finally:
        _cleanup(seeded["user_id"])


@pytest.mark.parametrize(
    ("missing_periods", "expected_generations"),
    [
        ((), 0),
        (("after",), 1),
        (("before", "after"), 2),
    ],
)


def test_postgres_orchestration_reuses_or_generates_each_missing_period(
    tmp_path,
    monkeypatch,
    missing_periods,
    expected_generations,
):
    from services import change_orchestration

    seeded = _seed(tmp_path)
    _delete_cached_periods(seeded, missing_periods)
    model_loads = []

    @contextmanager
    def recording_predictor(weight_file_path, *, weight_sha256):
        model_loads.append((weight_file_path, weight_sha256))

        def predict(rgb):
            return np.where(rgb[:, :, 0] % 2 == 0, 2, 1).astype(np.uint8)

        yield predict

    monkeypatch.setattr(
        classification_generation,
        "model_tile_predictor",
        recording_predictor,
    )
    monkeypatch.setenv(
        "CLASSIFICATION_RESULT_DIR",
        str(tmp_path / "generated-classifications"),
    )
    monkeypatch.setattr(
        change_orchestration,
        "SNAPSHOT_ROOT",
        tmp_path / "snapshots",
    )
    request_id = str(uuid4())
    payload = {
        "request_id": request_id,
        "before_image_id": seeded["before_image_id"],
        "after_image_id": seeded["after_image_id"],
        "model_id": seeded["model_id"],
    }
    app = FastAPI()
    app.include_router(change_results.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=seeded["user_id"]
    )

    try:
        with TestClient(app) as client:
            assert client.post("/api/change-results", json=payload).status_code == 202
            completed = _wait_for_terminal(client, request_id)

        assert completed.status_code == 200
        assert completed.json()["data"]["periods"]["before"]["status"] == "SUCCEEDED"
        assert completed.json()["data"]["periods"]["after"]["status"] == "SUCCEEDED"
        assert len(model_loads) == expected_generations
    finally:
        _cleanup(seeded["user_id"])


def test_postgres_concurrent_submission_claims_once_at_application_boundary(
    tmp_path,
    monkeypatch,
):
    seeded = _seed(tmp_path)
    barrier = asyncio.Barrier(2)
    original_claim = change_result_crud.claim_submission
    scheduled = []

    async def synchronized_claim(*args, **kwargs):
        await barrier.wait()
        return await original_claim(*args, **kwargs)

    monkeypatch.setattr(
        change_results.crud_results,
        "claim_submission",
        synchronized_claim,
    )
    monkeypatch.setattr(
        change_results,
        "schedule_change_orchestration",
        lambda result_id, user_id, owner: scheduled.append(
            (result_id, user_id, owner)
        ),
    )

    async def submit(request_id):
        async with AsyncSessionLocal() as db:
            return await change_results.create_change_result(
                change_results.CreateChangeResultRequest(
                    request_id=request_id,
                    before_image_id=seeded["before_image_id"],
                    after_image_id=seeded["after_image_id"],
                    model_id=seeded["model_id"],
                ),
                db=db,
                current_user=SimpleNamespace(id=seeded["user_id"]),
            )

    request_ids = (uuid4(), uuid4())
    try:
        async def submit_both():
            return await asyncio.gather(
                submit(request_ids[0]),
                submit(request_ids[1]),
            )

        responses = asyncio.run(submit_both())
        assert [response.status_code for response in responses] == [202, 202]
        assert len(scheduled) == 1

        connection = psycopg2.connect(_database_url())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT count(DISTINCT cr.change_result_id), count(*)
                    FROM change_requests cr
                    WHERE cr.user_id = %s AND cr.request_id IN (%s, %s)
                    """,
                    (
                        seeded["user_id"],
                        str(request_ids[0]),
                        str(request_ids[1]),
                    ),
                )
                assert cursor.fetchone() == (1, 2)
        finally:
            connection.close()
    finally:
        _cleanup(seeded["user_id"])


def test_postgres_full_identity_collision_rebinds_requests_to_one_winner(tmp_path):
    seeded = _seed(tmp_path)
    request_ids = (str(uuid4()), str(uuid4()))
    owners = (uuid4().hex, uuid4().hex)
    submitted = {
        "before_image_id": seeded["before_image_id"],
        "after_image_id": seeded["after_image_id"],
        "model_id": seeded["model_id"],
    }

    async def scenario():
        claims = []
        for index in range(2):
            async with AsyncSessionLocal() as db:
                claim = await change_result_crud.claim_submission(
                    db,
                    request_id=request_ids[index],
                    request_fingerprint=hashlib.sha256(
                        request_ids[index].encode()
                    ).hexdigest(),
                    preparation_key=hashlib.sha256(
                        f"prep-{request_ids[index]}".encode()
                    ).hexdigest(),
                    submitted_inputs=submitted,
                    user_id=seeded["user_id"],
                    owner=owners[index],
                    now=datetime.now(timezone.utc),
                )
                await db.commit()
                claims.append(claim)

        barrier = asyncio.Barrier(2)

        async def freeze(index):
            await barrier.wait()
            async with AsyncSessionLocal() as db:
                return await change_result_crud.set_frozen_identity(
                    db,
                    result_id=claims[index].row.id,
                    user_id=seeded["user_id"],
                    owner=owners[index],
                    orchestration_identity_sha256="f" * 64,
                    before_identity_sha256="b" * 64,
                    after_identity_sha256="a" * 64,
                    frozen_inputs={"test": index},
                    now=datetime.now(timezone.utc),
                )

        return claims, await asyncio.gather(freeze(0), freeze(1))

    try:
        claims, winners = asyncio.run(scenario())
        assert winners[0].id == winners[1].id

        connection = psycopg2.connect(_database_url())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT status, phase, count(*)
                    FROM change_results
                    WHERE id IN (%s, %s)
                    GROUP BY status, phase
                    ORDER BY status, phase
                    """,
                    (claims[0].row.id, claims[1].row.id),
                )
                assert cursor.fetchall() == [
                    ("FAILED", "MERGED", 1),
                    ("PROCESSING", "WAITING_BEFORE", 1),
                ]
                cursor.execute(
                    """
                    SELECT count(DISTINCT change_result_id), count(*)
                    FROM change_requests
                    WHERE user_id = %s AND request_id IN (%s, %s)
                    """,
                    (seeded["user_id"], request_ids[0], request_ids[1]),
                )
                assert cursor.fetchone() == (1, 2)
        finally:
            connection.close()
    finally:
        _cleanup(seeded["user_id"])


def test_postgres_changed_image_bytes_do_not_reuse_old_classification(
    tmp_path,
    monkeypatch,
):
    from services import change_orchestration

    seeded = _seed(tmp_path)
    _write_source(
        seeded["after_source"],
        9,
        RasterGrid(
            width=4,
            height=2,
            crs="EPSG:4326",
            transform=Affine(0.01, 0, 110, 0, -0.01, 30),
        ),
    )
    monkeypatch.setattr(
        classification_generation,
        "model_tile_predictor",
        _fake_predictor,
    )
    monkeypatch.setenv(
        "CLASSIFICATION_RESULT_DIR",
        str(tmp_path / "generated-classifications"),
    )
    monkeypatch.setattr(
        change_orchestration,
        "SNAPSHOT_ROOT",
        tmp_path / "snapshots",
    )
    request_id = str(uuid4())
    app = FastAPI()
    app.include_router(change_results.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=seeded["user_id"]
    )

    try:
        with TestClient(app) as client:
            submitted = client.post(
                "/api/change-results",
                json={
                    "request_id": request_id,
                    "before_image_id": seeded["before_image_id"],
                    "after_image_id": seeded["after_image_id"],
                    "model_id": seeded["model_id"],
                },
            )
            assert submitted.status_code == 202
            completed = _wait_for_terminal(client, request_id)

        assert completed.status_code == 200
        assert (
            completed.json()["data"]["after"]["identity_sha256"]
            != seeded["after_identity_sha256"]
        )
    finally:
        _cleanup(seeded["user_id"])


def test_postgres_identification_failure_can_retry_with_new_request_id(
    tmp_path,
    monkeypatch,
):
    from services import change_orchestration

    seeded = _seed(tmp_path)
    _delete_cached_periods(seeded, ("after",))

    @contextmanager
    def failing_predictor(weight_file_path, *, weight_sha256):
        def predict(rgb):
            raise RuntimeError("integration inference failure")

        yield predict

    monkeypatch.setattr(
        classification_generation,
        "model_tile_predictor",
        failing_predictor,
    )
    monkeypatch.setenv(
        "CLASSIFICATION_RESULT_DIR",
        str(tmp_path / "generated-classifications"),
    )
    monkeypatch.setattr(
        change_orchestration,
        "SNAPSHOT_ROOT",
        tmp_path / "snapshots",
    )
    request_id = str(uuid4())
    retry_id = str(uuid4())
    app = FastAPI()
    app.include_router(change_results.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=seeded["user_id"]
    )

    try:
        with TestClient(app) as client:
            assert client.post(
                "/api/change-results",
                json={
                    "request_id": request_id,
                    "before_image_id": seeded["before_image_id"],
                    "after_image_id": seeded["after_image_id"],
                    "model_id": seeded["model_id"],
                },
            ).status_code == 202
            failed = _wait_for_terminal(client, request_id)
            assert failed.status_code == 422
            assert (
                failed.json()["data"]["error_code"]
                == "IDENTIFICATION_EXECUTION_FAILED"
            )
            assert failed.json()["data"]["periods"]["after"]["status"] == "FAILED"

            seeded["before_source"].unlink()
            monkeypatch.setattr(
                classification_generation,
                "model_tile_predictor",
                _fake_predictor,
            )
            retried = client.post(
                f"/api/change-results/{request_id}/retry",
                json={"request_id": retry_id},
            )
            assert retried.status_code == 202
            completed = _wait_for_terminal(client, retry_id)

        assert completed.status_code == 200
        assert completed.json()["data"]["request_id"] == retry_id
        assert (
            completed.json()["data"]["before"]["identity_sha256"]
            == seeded["before_identity_sha256"]
        )
    finally:
        _cleanup(seeded["user_id"])


def test_postgres_targeted_expiry_marks_the_request_retryable(tmp_path):
    seeded = _seed(tmp_path)
    request_id = str(uuid4())
    now = datetime.now(timezone.utc)

    async def scenario():
        async with AsyncSessionLocal() as db:
            await change_result_crud.claim_submission(
                db,
                request_id=request_id,
                request_fingerprint="f" * 64,
                preparation_key="p" * 64,
                submitted_inputs={
                    "before_image_id": seeded["before_image_id"],
                    "after_image_id": seeded["after_image_id"],
                    "model_id": seeded["model_id"],
                },
                user_id=seeded["user_id"],
                owner=uuid4().hex,
                now=now - timedelta(minutes=3),
            )
            await db.commit()
        async with AsyncSessionLocal() as db:
            expired = await change_result_crud.expire_request(
                db,
                request_id=request_id,
                user_id=seeded["user_id"],
                now=now,
            )
            await db.commit()
            row = await change_result_crud.get_by_request(
                db,
                request_id,
                seeded["user_id"],
            )
            return expired, row

    try:
        expired, row = asyncio.run(scenario())

        assert expired == 1
        assert row.error_code == "CHANGE_RESULT_INTERRUPTED"
        assert row.error_data["retryable"] is True
    finally:
        _cleanup(seeded["user_id"])


def test_postgres_matrix_retry_reuses_both_periods_after_pipeline_upgrade(
    tmp_path,
    monkeypatch,
):
    from services import change_orchestration

    seeded = _seed(tmp_path)
    original_compute = change_orchestration.compute_transition_matrix_m2

    def fail_matrix(*_args):
        raise RuntimeError("integration matrix failure")

    monkeypatch.setattr(
        change_orchestration,
        "compute_transition_matrix_m2",
        fail_matrix,
    )
    monkeypatch.setattr(
        change_orchestration,
        "SNAPSHOT_ROOT",
        tmp_path / "snapshots",
    )
    request_id = str(uuid4())
    retry_id = str(uuid4())
    app = FastAPI()
    app.include_router(change_results.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=seeded["user_id"]
    )

    try:
        with TestClient(app) as client:
            assert client.post(
                "/api/change-results",
                json={
                    "request_id": request_id,
                    "before_image_id": seeded["before_image_id"],
                    "after_image_id": seeded["after_image_id"],
                    "model_id": seeded["model_id"],
                },
            ).status_code == 202
            failed = _wait_for_terminal(client, request_id)
            assert failed.status_code == 500

            monkeypatch.setattr(
                change_orchestration,
                "compute_transition_matrix_m2",
                original_compute,
            )
            monkeypatch.setattr(
                change_orchestration,
                "CLASSIFICATION_SCHEME_VERSION",
                "land-cover-6/v2",
            )
            assert client.post(
                f"/api/change-results/{request_id}/retry",
                json={"request_id": retry_id},
            ).status_code == 202
            completed = _wait_for_terminal(client, retry_id)

        assert completed.status_code == 200
        assert (
            completed.json()["data"]["before"]["identity_sha256"]
            == seeded["before_identity_sha256"]
        )
        assert (
            completed.json()["data"]["after"]["identity_sha256"]
            == seeded["after_identity_sha256"]
        )
    finally:
        _cleanup(seeded["user_id"])


def test_postgres_waiting_for_owned_identification_has_a_busy_budget(
    tmp_path,
    monkeypatch,
):
    from services import change_orchestration

    seeded = _seed(tmp_path)
    _mark_period_processing(seeded, "after")
    monkeypatch.setattr(change_orchestration, "IDENTIFICATION_WAIT_SECONDS", 0)
    monkeypatch.setattr(
        change_orchestration,
        "SNAPSHOT_ROOT",
        tmp_path / "snapshots",
    )
    request_id = str(uuid4())
    app = FastAPI()
    app.include_router(change_results.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=seeded["user_id"]
    )

    try:
        with TestClient(app) as client:
            assert client.post(
                "/api/change-results",
                json={
                    "request_id": request_id,
                    "before_image_id": seeded["before_image_id"],
                    "after_image_id": seeded["after_image_id"],
                    "model_id": seeded["model_id"],
                },
            ).status_code == 202
            failed = _wait_for_terminal(client, request_id)

        assert failed.status_code == 409
        assert failed.json()["data"]["error_code"] == "IDENTIFICATION_RESULT_BUSY"
        assert failed.json()["data"]["retryable"] is True
    finally:
        _cleanup(seeded["user_id"])
