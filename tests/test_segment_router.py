import base64
import io
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")
os.environ.setdefault("GEOSERVER_URL", "http://example.com/geoserver")
os.environ.setdefault("GEOSERVER_USER", "admin")
os.environ.setdefault("GEOSERVER_PASSWORD", "geoserver")
os.environ.setdefault("GEOSERVER_WORKSPACE", "ws")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from config.db_config import get_db
from router import segment
from utils.get_user_by_token import get_current_user


async def _override_db():
    yield SimpleNamespace()


def _override_user():
    return SimpleNamespace(id=7)


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(segment.router)
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_user
    return TestClient(app)


def _png_bytes(mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (2, 2), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_build_getmap_url_overrides_view_params():
    url = segment._build_getmap_url(
        wms_url="http://example.com/geoserver/ws/wms?service=WMS&layers=ws:old&format=image/jpeg",
        layer_name="new_layer",
        bbox="116.3,39.8,116.5,40.0",
        width=800,
        height=600,
        srs="EPSG:4326",
    )

    assert "request=GetMap" in url
    assert "layers=ws%3Aold" in url
    assert "format=image%2Fpng" in url
    assert "bbox=116.3%2C39.8%2C116.5%2C40.0" in url
    assert "width=800" in url
    assert "height=600" in url
    assert "srs=EPSG%3A4326" in url


def test_segment_endpoint_returns_data_url():
    client = _make_client()
    image = SimpleNamespace(
        id=1,
        user_id=7,
        wms_url="http://example.com/geoserver/ws/wms?service=WMS&layers=ws:image_1",
        layer_name="image_1",
    )
    result_png = _png_bytes("RGBA")

    with patch.object(segment.crud_images, "get_image_by_id", new=AsyncMock(return_value=image)), \
         patch.object(segment, "_fetch_wms_png", new=AsyncMock(return_value=_png_bytes())), \
         patch.object(segment.deeplab_service, "segment_rgba_png", return_value=result_png):
        response = client.post(
            "/api/segment",
            json={
                "image_id": 1,
                "bbox": "116.3,39.8,116.5,40.0",
                "width": 800,
                "height": 600,
                "srs": "EPSG:4326",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "识别成功"
    assert body["data"]["bbox"] == "116.3,39.8,116.5,40.0"
    assert body["data"]["image"] == "data:image/png;base64," + base64.b64encode(result_png).decode("ascii")


def test_segment_endpoint_passes_classes_to_service():
    client = _make_client()
    image = SimpleNamespace(
        id=1,
        user_id=7,
        wms_url="http://example.com/geoserver/ws/wms?service=WMS&layers=ws:image_1",
        layer_name="image_1",
    )
    result_png = _png_bytes("RGBA")

    with patch.object(segment.crud_images, "get_image_by_id", new=AsyncMock(return_value=image)), \
         patch.object(segment, "_fetch_wms_png", new=AsyncMock(return_value=_png_bytes())), \
         patch.object(segment.deeplab_service, "segment_rgba_png", return_value=result_png) as segment_rgba:
        response = client.post(
            "/api/segment",
            json={
                "image_id": 1,
                "bbox": "116.3,39.8,116.5,40.0",
                "width": 800,
                "height": 600,
                "srs": "EPSG:4326",
                "classes": [1, 2, 2],
            },
        )

    assert response.status_code == 200
    assert segment_rgba.call_args.args[1] == [1, 2]


def test_segment_endpoint_rejects_invalid_classes():
    client = _make_client()

    response = client.post(
        "/api/segment",
        json={
            "image_id": 1,
            "bbox": "116.3,39.8,116.5,40.0",
            "width": 800,
            "height": 600,
            "srs": "EPSG:4326",
            "classes": [0, 6],
        },
    )

    assert response.status_code == 422
    assert "classes 中的类别 ID 必须在 1-5 之间" in response.text


def test_segment_endpoint_checks_current_user_image():
    client = _make_client()

    with patch.object(segment.crud_images, "get_image_by_id", new=AsyncMock(return_value=None)):
        response = client.post(
            "/api/segment",
            json={
                "image_id": 999,
                "bbox": "116.3,39.8,116.5,40.0",
                "width": 800,
                "height": 600,
                "srs": "EPSG:4326",
            },
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "影像不存在"
