import asyncio
import os
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")
os.environ.setdefault("GEOSERVER_URL", "http://example.com/geoserver")
os.environ.setdefault("GEOSERVER_USER", "admin")
os.environ.setdefault("GEOSERVER_PASSWORD", "geoserver")
os.environ.setdefault("GEOSERVER_WORKSPACE", "ws")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import Select

from config.db_config import get_db
from crud import images as crud_images
from router import images
from utils.get_user_by_token import get_current_user


async def _override_db():
    yield SimpleNamespace()


def _override_user():
    return SimpleNamespace(id=7)


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(images.router)
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_user
    return TestClient(app)


def _image(image_id: int = 1, image_name: str = "河流影像") -> SimpleNamespace:
    return SimpleNamespace(
        id=image_id,
        image_name=image_name,
        resolution=10.0,
        capture_date=date(2026, 8, 29),
        satellite="高分一号",
        image_type="多光谱",
        region_code="330100",
        img_path="uploads/images/river.tif",
        bbox=[120.0, 30.0, 121.0, 31.0],
        layer_name="river",
        wms_url="http://example.com/river",
        boundary_files=[],
        upload_time=datetime(2026, 8, 29, 12, 0, 0),
    )


def test_query_images_returns_one_paginated_contract():
    client = _make_client()

    with patch.object(
        images.crud_images,
        "query_images",
        new=AsyncMock(return_value=([_image()], 3)),
    ):
        response = client.get(
            "/api/images",
            params={"page": 2, "pageSize": 2, "keyword": " 河流 "},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"]["items"][0]["image_name"] == "河流影像"
    assert body["data"] | {"items": []} == {
        "items": [],
        "total": 3,
        "page": 2,
        "pageSize": 2,
        "totalPages": 2,
    }


def test_legacy_image_catalog_routes_are_not_published():
    paths = _make_client().get("/openapi.json").json()["paths"]

    assert "/api/images/list" not in paths
    assert "/api/images/search" not in paths


def test_query_images_rejects_invalid_pagination_and_keyword_length():
    client = _make_client()

    assert client.get("/api/images", params={"page": 0}).status_code == 422
    assert client.get("/api/images", params={"pageSize": 101}).status_code == 422
    assert client.get("/api/images", params={"keyword": "x" * 101}).status_code == 422


def test_get_image_keeps_the_image_record_contract():
    client = _make_client()

    with patch.object(
        images.crud_images,
        "get_image_by_id",
        new=AsyncMock(return_value=_image(image_id=9)),
    ):
        response = client.get("/api/images/9")

    assert response.status_code == 200
    assert response.json()["data"]["id"] == 9
    assert response.json()["data"]["image_name"] == "河流影像"


def test_get_image_returns_404_when_the_record_is_missing():
    client = _make_client()

    with patch.object(
        images.crud_images,
        "get_image_by_id",
        new=AsyncMock(return_value=None),
    ):
        response = client.get("/api/images/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "影像不存在"


def test_query_images_uses_one_escaped_filter_for_count_and_page():
    expected_items = [_image()]
    db = SimpleNamespace(execute=AsyncMock(side_effect=[
        SimpleNamespace(scalar=lambda: 12),
        SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: expected_items)),
    ]))

    items, total = asyncio.run(crud_images.query_images(db, 7, 2, 5, "  %_\\  "))

    count_query, page_query = [call.args[0] for call in db.execute.await_args_list]
    assert isinstance(count_query, Select) and isinstance(page_query, Select)
    assert count_query.whereclause.compare(page_query.whereclause)
    assert "%\\%\\_\\\\%" in count_query.compile(dialect=postgresql.dialect()).params.values()
    sql = " ".join(str(page_query.compile(
        dialect=postgresql.dialect(),
        compile_kwargs={"literal_binds": True},
    )).split())
    assert "LIMIT 5 OFFSET 5" in sql
    assert items == expected_items
    assert total == 12
