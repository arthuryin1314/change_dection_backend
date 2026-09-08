import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import psycopg2
import pytest
import rasterio
from affine import Affine
from fastapi import FastAPI
from fastapi.testclient import TestClient
from dotenv import load_dotenv

load_dotenv()
from config.db_config import get_db
from router import identification_results
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


def _write_source(path: Path) -> None:
    data = np.full((3, 5, 6), 7, dtype=np.uint8)
    data[:, 0, 0] = 0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=6,
        height=5,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=Affine(0.01, 0, 110, 0, -0.01, 30),
        nodata=0,
    ) as dataset:
        dataset.write(data)


@contextmanager
def _fake_predictor(weight_file_path, *, weight_sha256):
    def predict(rgb):
        result = np.full(rgb.shape[:2], 5, dtype=np.uint8)
        result[0, 1] = 0
        return result

    yield predict


@contextmanager
def _failing_predictor(weight_file_path, *, weight_sha256):
    def predict(rgb):
        raise RuntimeError("integration inference failure")

    yield predict


def _insert_model(connection, user_id: int, weight_path: Path, suffix: str) -> int:
    with connection.cursor() as cursor:
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
                f"integration-{suffix}",
                str(weight_path),
                str(weight_path.with_suffix(".py")),
                "PostgreSQL integration fixture",
            ),
        )
        return cursor.fetchone()[0]


def _seed_sources(source_path: Path, weight_path: Path):
    suffix = uuid4().hex
    connection = psycopg2.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_info (username, phone, password)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (f"integration-{suffix}", f"9{suffix[:10]}", "not-a-login-secret"),
            )
            user_id = cursor.fetchone()[0]
            cursor.execute(
                """
                INSERT INTO images (user_id, image_name, img_path)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (user_id, f"integration-{suffix}", str(source_path)),
            )
            image_id = cursor.fetchone()[0]
        model_id = _insert_model(connection, user_id, weight_path, suffix)
        connection.commit()
        return user_id, image_id, model_id
    finally:
        connection.close()


def _cleanup_sources(user_id: int) -> None:
    connection = psycopg2.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM user_info WHERE id = %s", (user_id,))
        connection.commit()
    finally:
        connection.close()


def _add_model(user_id: int, weight_path: Path) -> int:
    connection = psycopg2.connect(_database_url())
    try:
        model_id = _insert_model(connection, user_id, weight_path, uuid4().hex)
        connection.commit()
        return model_id
    finally:
        connection.close()


def _result_files(result_id: str):
    connection = psycopg2.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT status, classes_path, valid_mask_path
                FROM classification_results
                WHERE id = %s
                """,
                (result_id,),
            )
            return cursor.fetchone()
    finally:
        connection.close()


def _result_area(result_id: str):
    connection = psycopg2.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT area_status, class_area_m2, area_completed_at, area_failure_detail
                FROM classification_results
                WHERE id = %s
                """,
                (result_id,),
            )
            return cursor.fetchone()
    finally:
        connection.close()


def _wait_for_status(client: TestClient, result_id: str, expected: str):
    deadline = time.monotonic() + 10
    while True:
        fetched = client.get(f"/api/identification-results/{result_id}")
        assert fetched.status_code == 200
        if fetched.json()["data"]["status"] == expected:
            return fetched
        assert time.monotonic() < deadline
        time.sleep(0.05)


def test_postgres_api_create_complete_read_reuse_and_authorization(
    tmp_path,
    monkeypatch,
):
    source_path = tmp_path / "source.tif"
    weight_path = tmp_path / "weights.pth"
    _write_source(source_path)
    weight_path.write_bytes(b"integration-weight")
    weight_path.with_suffix(".py").write_text("# fixture", encoding="utf-8")
    user_id, image_id, model_id = _seed_sources(source_path, weight_path)

    app = FastAPI()
    app.include_router(identification_results.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    monkeypatch.setattr(
        "services.classification_generation.model_tile_predictor",
        _fake_predictor,
    )
    monkeypatch.setattr(
        identification_results,
        "RESULT_STORAGE_ROOT",
        tmp_path / "results",
    )

    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/identification-results",
                json={"image_id": image_id, "model_id": model_id},
            )
            assert created.status_code == 202
            result_id = created.json()["data"]["result_id"]

            fetched = _wait_for_status(client, result_id, "SUCCEEDED")

            data = fetched.json()["data"]
            assert data["grid"]["width"] == 6
            assert data["grid"]["height"] == 5
            assert "4326" in data["grid"]["crs"]

            resolved = client.get(
                "/api/identification-results/resolve",
                params={"image_id": image_id, "model_id": model_id},
            )
            assert resolved.status_code == 200
            assert resolved.json()["data"]["status"] == "SUCCEEDED"
            assert resolved.json()["data"]["result"]["result_id"] == result_id

            calculated = client.post(
                f"/api/identification-results/{result_id}/areas"
            )
            assert calculated.status_code == 200
            area_data = calculated.json()["data"]
            assert area_data["area_status"] == "SUCCEEDED"
            assert area_data["class_area_m2"][0] == pytest.approx(
                1069626.783188343,
                rel=1e-10,
                abs=1e-5,
            )
            assert area_data["class_area_m2"][1:5] == [0.0] * 4
            assert area_data["class_area_m2"][5] > area_data["class_area_m2"][0]
            persisted_area = _result_area(result_id)
            assert persisted_area[0] == "SUCCEEDED"
            assert persisted_area[1] == area_data["class_area_m2"]
            assert persisted_area[2] is not None
            assert persisted_area[3] is None

            area_detail = client.get(f"/api/identification-results/{result_id}")
            assert area_detail.json()["data"]["class_area_m2"] == area_data[
                "class_area_m2"
            ]

            status, classes_path, valid_mask_path = _result_files(result_id)
            assert status == "SUCCEEDED"
            with rasterio.open(classes_path) as classes_ds:
                classes = classes_ds.read(1)
            with rasterio.open(valid_mask_path) as valid_ds:
                valid = valid_ds.read(1)
            assert classes[0, 0] == 0
            assert valid[0, 0] == 0
            assert classes[0, 1] == 0
            assert valid[0, 1] == 1

            reused = client.post(
                "/api/identification-results",
                json={"image_id": image_id, "model_id": model_id},
            )
            assert reused.status_code == 200
            assert reused.json()["data"]["result_id"] == result_id

            rendered = client.get(
                f"/api/identification-results/{result_id}/render",
                params={
                    "bbox": "110,29.95,110.06,30",
                    "width": 6,
                    "height": 5,
                    "srs": "EPSG:4326",
                },
            )
            assert rendered.status_code == 200
            assert rendered.headers["content-type"] == "image/png"

            second_weight = tmp_path / "second.pth"
            second_weight.write_bytes(b"different-integration-weight")
            second_weight.with_suffix(".py").write_text("# fixture", encoding="utf-8")
            second_model_id = _add_model(user_id, second_weight)
            different = client.post(
                "/api/identification-results",
                json={"image_id": image_id, "model_id": second_model_id},
            )
            assert different.status_code == 202
            different_id = different.json()["data"]["result_id"]
            assert different_id != result_id
            _wait_for_status(client, different_id, "SUCCEEDED")

            failed_weight = tmp_path / "failed.pth"
            failed_weight.write_bytes(b"failing-integration-weight")
            failed_weight.with_suffix(".py").write_text("# fixture", encoding="utf-8")
            failed_model_id = _add_model(user_id, failed_weight)
            monkeypatch.setattr(
                "services.classification_generation.model_tile_predictor",
                _failing_predictor,
            )
            failed = client.post(
                "/api/identification-results",
                json={"image_id": image_id, "model_id": failed_model_id},
            )
            assert failed.status_code == 202
            failed_id = failed.json()["data"]["result_id"]
            _wait_for_status(client, failed_id, "FAILED")
            assert _result_files(failed_id) == ("FAILED", None, None)
            assert not (tmp_path / "results" / failed_id).exists()

            app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
                id=user_id + 1_000_000
            )
            denied = client.get(f"/api/identification-results/{result_id}")
            assert denied.status_code == 404
    finally:
        _cleanup_sources(user_id)
