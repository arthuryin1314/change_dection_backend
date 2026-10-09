import io
import json
import os
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zipfile import ZipFile

import numpy as np
import pytest
import rasterio
from affine import Affine
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")

from config.db_config import get_db
from router import change_results
from utils.get_user_by_token import get_current_user


REQUEST_ID = "90f0bdba-6b7e-4a45-a078-dce12340f6d2"
DOWNLOAD_URLS = [
    f"/api/change-results/{REQUEST_ID}/download.zip",
    "/api/change-results/history/41/download.zip",
]


@pytest.fixture
def download_case(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    directory = root / "41"
    directory.mkdir(parents=True)
    with rasterio.open(
        directory / "transition.tif", "w", driver="GTiff", width=2, height=1,
        count=1, dtype="uint8", crs="EPSG:3857",
        transform=Affine(10, 0, 100, 0, -10, 200), nodata=255,
    ) as dataset:
        dataset.write(np.array([[1, 255]], dtype=np.uint8), 1)
    Image.new("RGBA", (2, 1), (230, 25, 75, 255)).save(directory / "visualization.png")
    (directory / "legend.json").write_text(
        json.dumps({"classification_scheme_version": "land-cover-6/v1",
                    "nodata_value": 255, "transitions": [{"code": 1, "color": "#e6194b"}]}),
        encoding="utf-8",
    )
    row = SimpleNamespace(
        id=41, request_id=REQUEST_ID, user_id=7, status="SUCCEEDED",
        matrix_m2=[[0] * 6 for _ in range(6)], calculated_at=datetime.now(timezone.utc),
        artifact_metadata={
            "artifact_version": "change-artifacts-v1",
            "transition_geotiff": {"filename": "transition.tif", "media_type": "image/tiff"},
            "visualization_png": {"filename": "visualization.png", "media_type": "image/png"},
            "legend": {"filename": "legend.json", "media_type": "application/json"},
        },
    )
    async def lookup(db, key, user_id):
        return row if user_id == row.user_id and key in (41, REQUEST_ID) else None
    monkeypatch.setattr(change_results.crud_results, "get_by_request", AsyncMock(side_effect=lookup))
    monkeypatch.setattr(change_results.crud_results, "get_succeeded_history_by_id", AsyncMock(side_effect=lookup))
    monkeypatch.setenv("CHANGE_RESULT_ARTIFACT_DIR", str(root))
    temporary = tmp_path / "downloads"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    app = FastAPI()
    app.include_router(change_results.router)
    async def database():
        yield None
    app.dependency_overrides[get_db] = database
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)
    return app, row, directory, temporary


@pytest.mark.parametrize("url", DOWNLOAD_URLS)
def test_download_zip_preserves_persisted_files_and_cleans_temporary_archive(download_case, url):
    app, row, directory, temporary = download_case
    with TestClient(app) as client:
        response = client.get(url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == 'attachment; filename="change-result-41.zip"'
    assert response.headers["cache-control"] == "no-store"
    with ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"transition.tif", "visualization.png", "legend.json"}
        for filename in archive.namelist():
            assert archive.read(filename) == (directory / filename).read_bytes()
        with rasterio.MemoryFile(archive.read("transition.tif")) as memory:
            with memory.open() as dataset:
                np.testing.assert_array_equal(dataset.read(1), [[1, 255]])
                assert dataset.nodata == 255
                assert dataset.crs.to_epsg() == 3857
                assert dataset.transform == Affine(10, 0, 100, 0, -10, 200)
    assert list(temporary.iterdir()) == []


def test_download_requires_owner_and_complete_persisted_artifact(download_case):
    app, row, directory, _ = download_case
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=8)
    with TestClient(app) as client:
        response = client.get(DOWNLOAD_URLS[0])
    assert response.status_code == 404

    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)
    (directory / "legend.json").unlink()
    with TestClient(app) as client:
        response = client.get(DOWNLOAD_URLS[0])
    assert response.status_code == 404
    assert list(directory.parent.glob(".change-result-*.zip")) == []
