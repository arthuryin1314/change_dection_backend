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
from utils.tif_reader import NoOverlapError, TifReadResult, UnsupportedSrsError


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


def _png_bytes(mode: str = "RGB", size: tuple[int, int] = (2, 2), color=None) -> bytes:
    buf = io.BytesIO()
    if color is None:
        color = (255, 255, 255, 180) if mode == "RGBA" else (255, 255, 255)
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def _read_result() -> TifReadResult:
    return TifReadResult(
        image=Image.new("RGB", (4, 3), (255, 255, 255)),
        requested_native_width=4,
        requested_native_height=3,
        native_width=4,
        native_height=3,
        read_width=4,
        read_height=3,
        effective_offset=(0, 0),
        effective_size=(4, 3),
        capped=False,
    )


def test_segment_endpoint_returns_data_url():
    client = _make_client()
    image = SimpleNamespace(
        id=1,
        user_id=7,
        img_path=__file__,
    )
    result_png = _png_bytes("RGBA", size=(4, 3))

    with patch.object(segment.crud_images, "get_image_by_id", new=AsyncMock(return_value=image)), \
         patch.object(segment, "read_tif_rgb_window", return_value=_read_result()), \
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
    encoded = body["data"]["image"].removeprefix("data:image/png;base64,")
    returned_image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert returned_image.size == (800, 600)


def test_segment_endpoint_passes_classes_to_service():
    client = _make_client()
    image = SimpleNamespace(
        id=1,
        user_id=7,
        img_path=__file__,
    )
    result_png = _png_bytes("RGBA", size=(4, 3))

    with patch.object(segment.crud_images, "get_image_by_id", new=AsyncMock(return_value=image)), \
         patch.object(segment, "read_tif_rgb_window", return_value=_read_result()), \
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


def test_segment_endpoint_aligns_partial_overlap_to_full_bbox_canvas():
    client = _make_client()
    image = SimpleNamespace(id=1, user_id=7, img_path=__file__)
    read_result = TifReadResult(
        image=Image.new("RGB", (2, 2), (255, 255, 255)),
        requested_native_width=4,
        requested_native_height=4,
        native_width=2,
        native_height=2,
        read_width=2,
        read_height=2,
        effective_offset=(0, 2),
        effective_size=(2, 2),
        capped=False,
    )
    result_png = _png_bytes("RGBA", size=(2, 2), color=(255, 0, 0, 180))

    with patch.object(segment.crud_images, "get_image_by_id", new=AsyncMock(return_value=image)), \
         patch.object(segment, "read_tif_rgb_window", return_value=read_result), \
         patch.object(segment.deeplab_service, "segment_rgba_png", return_value=result_png):
        response = client.post(
            "/api/segment",
            json={
                "image_id": 1,
                "bbox": "116.3,39.8,116.5,40.0",
                "width": 4,
                "height": 4,
                "srs": "EPSG:4326",
            },
        )

    assert response.status_code == 200
    encoded = response.json()["data"]["image"].removeprefix("data:image/png;base64,")
    returned_image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")
    for y in range(4):
        for x in range(4):
            alpha = returned_image.getpixel((x, y))[3]
            if x < 2 and y >= 2:
                assert alpha == 180
            else:
                assert alpha == 0


def test_segment_endpoint_handles_large_intermediate_png():
    """模型输出超过 PIL 默认 bomb 阈值时，中间 PNG 仍然能正常合成。"""
    client = _make_client()
    image = SimpleNamespace(id=1, user_id=7, img_path=__file__)

    big_size = 14000
    read_result = TifReadResult(
        image=Image.new("RGB", (big_size, big_size), (255, 255, 255)),
        requested_native_width=big_size,
        requested_native_height=big_size,
        native_width=big_size,
        native_height=big_size,
        read_width=big_size,
        read_height=big_size,
        effective_offset=(0, 0),
        effective_size=(big_size, big_size),
        capped=False,
    )
    big_png = _png_bytes("RGBA", size=(big_size, big_size), color=(255, 0, 0, 180))

    with patch.object(segment.crud_images, "get_image_by_id", new=AsyncMock(return_value=image)), \
         patch.object(segment, "read_tif_rgb_window", return_value=read_result), \
         patch.object(segment.deeplab_service, "segment_rgba_png", return_value=big_png):
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


def test_segment_endpoint_rejects_missing_img_path():
    client = _make_client()
    image = SimpleNamespace(id=1, user_id=7, img_path="Z:\\missing\\source.tif")

    with patch.object(segment.crud_images, "get_image_by_id", new=AsyncMock(return_value=image)):
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

    assert response.status_code == 422
    assert response.json()["detail"] == "影像源文件不存在或不可访问"


def test_segment_endpoint_maps_unsupported_srs_to_422():
    client = _make_client()
    image = SimpleNamespace(id=1, user_id=7, img_path=__file__)

    with patch.object(segment.crud_images, "get_image_by_id", new=AsyncMock(return_value=image)), \
         patch.object(segment, "read_tif_rgb_window", side_effect=UnsupportedSrsError("EPSG:99999")):
        response = client.post(
            "/api/segment",
            json={
                "image_id": 1,
                "bbox": "116.3,39.8,116.5,40.0",
                "width": 800,
                "height": 600,
                "srs": "EPSG:99999",
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "srs 不被支持: EPSG:99999"


def test_segment_endpoint_maps_no_overlap_to_422():
    client = _make_client()
    image = SimpleNamespace(id=1, user_id=7, img_path=__file__)

    with patch.object(segment.crud_images, "get_image_by_id", new=AsyncMock(return_value=image)), \
         patch.object(segment, "read_tif_rgb_window", side_effect=NoOverlapError("bbox 与影像无重叠")):
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

    assert response.status_code == 422
    assert response.json()["detail"] == "bbox 与影像无重叠"


def test_segment_endpoint_maps_read_errors_to_500():
    client = _make_client()
    image = SimpleNamespace(id=1, user_id=7, img_path=__file__)

    with patch.object(segment.crud_images, "get_image_by_id", new=AsyncMock(return_value=image)), \
         patch.object(segment, "read_tif_rgb_window", side_effect=RuntimeError("boom")):
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

    assert response.status_code == 500
    assert response.json()["detail"] == "读取影像失败: boom"
