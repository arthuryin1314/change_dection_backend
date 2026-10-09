import copy
import hashlib
import os
from datetime import datetime, timezone
from io import BytesIO
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault('DATABASE_URL', 'postgresql+asyncpg://user:pass@localhost/test')

import numpy as np
import pytest
import rasterio
from affine import Affine
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.db_config import get_db
from router import change_results
from services import change_report
from utils.change_report_pdf import format_crs_label, format_report_datetime
from utils.classification_storage import RasterGrid, write_classification_result
from utils.exception_handler import register_exception_handlers
from utils.get_user_by_token import get_current_user


@pytest.fixture
def report_case(tmp_path, monkeypatch):
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    records, images, snapshots = {}, {}, {}
    grid = RasterGrid(40, 20, 'EPSG:3857', Affine(2, 0, 0, 0, -2, 40))
    for index, period in enumerate(('before', 'after'), 1):
        original = tmp_path / f'{period}.tif'
        with rasterio.open(original, 'w', driver='GTiff', width=40, height=20,
                           count=3, dtype='uint16', crs=grid.crs, transform=grid.transform) as ds:
            ds.write(np.arange(2400, dtype=np.uint16).reshape(3, 20, 40))
        stored = write_classification_result(tmp_path, period, grid,
                                            np.ones((20, 40), dtype=np.uint8),
                                            np.ones((20, 40), dtype=np.uint8))
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        records[period] = SimpleNamespace(
            id=period, user_id=7, status='SUCCEEDED', identity_sha256=str(index)*64,
            image_content_sha256=digest, weight_content_sha256='a'*64,
            completed_at=now, classes_path=str(stored.classes_path),
            valid_mask_path=str(stored.valid_mask_path), crs=grid.crs,
            transform=list(grid.transform)[:6], raster_width=40, raster_height=20,
            area_status='SUCCEEDED', class_area_m2=[0, 3200, 0, 0, 0, 0],
        )
        images[index] = SimpleNamespace(
            img_path=str(original), image_name=f'{period} catalog image',
            capture_date=now.date(), resolution=2.0 + index, satellite=f'SAT-{index}',
        )
        snapshots[period] = dict(
            result_id=period, identity_sha256=str(index)*64, source_image_id=index,
            image_content_sha256=digest, weight_content_sha256='a'*64,
            completed_at=now.isoformat(), classification_scheme_version='land-cover-6/v1',
            area_status='SUCCEEDED', class_area_m2=[0, 3200, 0, 0, 0, 0],
            source={'image': {'id': index, 'name': 'image<font>&test',
                              'crs': grid.crs, 'width': 40, 'height': 20},
                    'model': {'name': 'model'}},
        )
    row = SimpleNamespace(id=150, status='SUCCEEDED', calculated_at=now,
                          matrix_m2=[[0]*6 for _ in range(6)], common_valid_area_m2=None,
                          before_snapshot=snapshots['before'], after_snapshot=snapshots['after'])
    row.matrix_m2[1][1] = 2000
    monkeypatch.setattr(change_report.change_crud, 'get_succeeded_history_by_id', AsyncMock(return_value=row))
    monkeypatch.setattr(change_report.classification_crud, 'get_result_by_id', AsyncMock(side_effect=lambda db, key, user: records.get(key)))
    monkeypatch.setattr(change_report.image_crud, 'get_image_by_id', AsyncMock(side_effect=lambda db, key, user: images.get(key)))
    app = FastAPI()
    app.include_router(change_results.router)
    register_exception_handlers(app)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)
    app.dependency_overrides[get_db] = lambda: object()
    return TestClient(app, raise_server_exceptions=False), row, records, images


def test_missing_saved_area_is_explained_not_500(report_case):
    client, row, records, _ = report_case
    row.before_snapshot['class_area_m2'] = None
    records['before'].class_area_m2 = None
    response = client.get('/api/change-results/history/150/report.pdf')
    assert response.status_code == 409
    assert response.headers['content-type'].startswith('application/json')
    assert 'before' in response.text and 'area' in response.text


def test_report_uses_current_image_catalog_metadata(report_case, monkeypatch):
    client, _, _, _ = report_case
    captured = {}

    def capture(report):
        captured.update(report)
        return b"%PDF-1.4"

    monkeypatch.setattr(change_report, "build_pdf", capture)
    response = client.get('/api/change-results/history/150/report.pdf')

    assert response.status_code == 200
    assert captured["before"]["name"] == "before catalog image"
    assert captured["after"]["name"] == "after catalog image"
    assert captured["before"]["capture_date"] == "2026-09-20"
    assert captured["before"]["satellite"] == "SAT-1"
    assert captured["after"]["resolution"] == "4 m"
    assert captured["before"]["crs"] == "EPSG:3857"
    assert captured["detection_time"] == "2026-09-20 08:00:00"
    assert len(captured["generated_at"]) == 19
    assert captured["generated_at"][4] == "-" and captured["generated_at"][10] == " "


def test_report_datetime_uses_24_hour_local_format():
    assert format_report_datetime("2026-09-19T13:20:34.496561+00:00") == "2026-09-19 21:20:34"


def test_crs_label_uses_readable_wkt_name():
    value = 'PROJCS["CGCS2000 / 3-degree Gauss-Kruger zone 40",GEOGCS["CGCS2000"]]'

    assert format_crs_label(value) == "CGCS2000 / 3-degree Gauss-Kruger zone 40"
    assert format_crs_label("EPSG:4528") == "EPSG:4528"


def test_report_falls_back_to_classification_area_when_snapshot_area_is_missing(report_case, monkeypatch):
    client, row, records, _ = report_case
    row.before_snapshot['class_area_m2'] = None
    records['before'].class_area_m2 = [0, 6400, 0, 0, 0, 0]
    captured = {}

    monkeypatch.setattr(change_report, 'build_pdf', lambda report: captured.update(report) or b'%PDF-1.4')
    response = client.get('/api/change-results/history/150/report.pdf')

    assert response.status_code == 200
    assert captured['before_area'] == [0, 6400, 0, 0, 0, 0]


def test_report_allows_legacy_snapshot_without_image_hash(report_case, monkeypatch):
    client, row, _, _ = report_case
    row.before_snapshot.pop('image_content_sha256')
    monkeypatch.setattr(change_report, 'build_pdf', lambda report: b'%PDF-1.4')

    response = client.get('/api/change-results/history/150/report.pdf')

    assert response.status_code == 200


def test_report_rejects_mismatched_image_hash(report_case, monkeypatch):
    client, row, _, _ = report_case
    row.before_snapshot['image_content_sha256'] = 'different'
    monkeypatch.setattr(change_report, 'build_pdf', lambda report: b'%PDF-1.4')

    response = client.get('/api/change-results/history/150/report.pdf')

    assert response.status_code == 409
    assert 'before' in response.text and 'image' in response.text


def test_report_includes_persisted_transition_artifact_and_snapshot(report_case, monkeypatch, tmp_path):
    client, row, _, _ = report_case
    artifact_root = tmp_path / "change-artifacts"
    artifact_dir = artifact_root / str(row.id)
    artifact_dir.mkdir(parents=True)
    from PIL import Image

    image = Image.new("RGBA", (1800, 900), (255, 0, 0, 255))
    image.save(artifact_dir / "visualization.png")
    legend = {
        "artifact_version": "change-artifacts-v1",
        "classification_scheme_version": "land-cover-6/v1",
        "classes": [{"id": 0, "name": "其他／背景", "color": "#000000"}],
        "transitions": [{"code": 1, "before": {"id": 0, "name": "其他／背景"}, "after": {"id": 1, "name": "水系"}, "color": "#e6194b"}],
    }
    (artifact_dir / "legend.json").write_text(json.dumps(legend, ensure_ascii=False), encoding="utf-8")
    row.artifact_metadata = {
        "artifact_version": "change-artifacts-v1",
        "visualization_png": {"filename": "visualization.png"},
        "legend": {"filename": "legend.json"},
    }
    row.transition_snapshot = {
        "classification_scheme_version": "land-cover-6/v1",
        "nodata_value": 255,
        "transition_pixel_counts": [0, 1] + [0] * 34,
        "transition_area_m2": [0.0, 100.0] + [0.0] * 34,
        "unchanged_pixel_count": 2,
        "unchanged_area_m2": 200.0,
        "nodata_pixel_count": 1,
        "nodata_area_m2": 100.0,
    }
    monkeypatch.setenv("CHANGE_RESULT_ARTIFACT_DIR", str(artifact_root))
    captured = {}
    monkeypatch.setattr(change_report, "build_pdf", lambda report: captured.update(report) or b"%PDF-1.4")

    response = client.get('/api/change-results/history/150/report.pdf?unit=ha')

    assert response.status_code == 200
    assert captured["transition_snapshot"] == row.transition_snapshot
    assert captured["transition_legend"] == legend
    assert captured["transition_png"]
    with Image.open(BytesIO(captured["transition_png"])) as rendered:
        assert max(rendered.size) <= 1600


def test_legacy_report_marks_missing_multi_value_transition_artifact(report_case, monkeypatch):
    client, row, _, _ = report_case
    row.artifact_metadata = {}
    captured = {}
    monkeypatch.setattr(change_report, "build_pdf", lambda report: captured.update(report) or b"%PDF-1.4")

    response = client.get('/api/change-results/history/150/report.pdf')

    assert response.status_code == 200
    assert captured["transition_snapshot"] is None
    assert captured["transition_legend"] is None
    assert "未保存多值转移结果" in captured["legacy_transition_note"]
