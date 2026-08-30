import asyncio
import os
import time
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
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
from router.image import image_lifecycle as workflow
from router.image import images
from router.image import upload_sessions
from utils.get_user_by_token import get_current_user


async def _override_db():
    yield SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())


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
    assert "/api/images/upload" not in paths
    assert "/api/images/upload/init" not in paths
    assert "/api/images/upload/status" not in paths
    assert "/api/images/upload/chunk" not in paths
    assert "/api/images/upload/complete" not in paths
    assert "/api/images/update/{image_id}" not in paths


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


def test_delete_image_keeps_contract_through_lifecycle_owner():
    client = _make_client()

    with patch.object(
        images.image_lifecycle,
        "delete_image",
        new=AsyncMock(return_value=True),
    ):
        response = client.delete("/api/images/delete/9")

    assert response.status_code == 200
    assert response.json()["data"] == {"id": 9}


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


def test_begin_upload_returns_the_resumable_session_contract():
    client = _make_client()
    session = {
        "upload_id": "a" * 32,
        "uploaded_chunks": [0, 2],
        "total_chunks": 3,
    }

    with patch.object(images, "begin_upload_session", return_value=session, create=True):
        response = client.post("/api/images/uploads", json={
            "fileName": "river.tif",
            "fileSize": 12,
            "chunkSize": 4,
            "totalChunks": 3,
            "fileHash": "b" * 32,
        })

    assert response.status_code == 200
    assert response.json()["data"] == {
        "uploadId": "a" * 32,
        "uploadedChunks": [0, 2],
        "totalChunks": 3,
    }


def test_upload_chunk_is_reported_when_the_session_is_resumed():
    with TemporaryDirectory(dir=Path("tests")) as temp_dir, patch.object(
        upload_sessions,
        "TMP_UPLOAD_DIR",
        Path(temp_dir),
    ):
        client = _make_client()
        payload = {
            "fileName": "river.tif",
            "fileSize": 8,
            "chunkSize": 4,
            "totalChunks": 2,
            "fileHash": "e8dc4081b13434b45189a720b77b6818",
        }

        upload_id = client.post("/api/images/uploads", json=payload).json()["data"]["uploadId"]
        with patch.object(
            upload_sessions,
            "save_session",
            wraps=upload_sessions.save_session,
        ) as save_meta:
            response = client.put(
                f"/api/images/uploads/{upload_id}/chunks/0",
                files={"chunk": ("0.part", b"abcd", "application/octet-stream")},
            )

        assert response.status_code == 200
        save_meta.assert_called_once()
        assert response.json()["data"] == {
            "uploadId": upload_id,
            "uploadedChunks": [0],
            "totalChunks": 2,
        }
        assert client.post("/api/images/uploads", json=payload).json()["data"] == response.json()["data"]


def test_create_image_accepts_the_new_multipart_contract():
    client = _make_client()
    files = [
        ("boundaryFiles", ("river.shp", b"shp", "application/octet-stream")),
        ("boundaryFiles", ("river.dbf", b"dbf", "application/octet-stream")),
        ("boundaryFiles", ("river.prj", b"prj", "application/octet-stream")),
    ]

    with patch.object(
        images,
        "create_image_asset",
        new=AsyncMock(return_value=_image()),
        create=True,
    ):
        response = client.post("/api/images", data={
            "uploadId": "a" * 32,
            "imageName": "河流影像",
            "resolution": "10",
            "captureDate": "2026-08-29",
            "satellite": "高分一号",
            "imageType": "多光谱",
            "regionCode": "330100",
        }, files=files)

    assert response.status_code == 200
    assert response.json()["data"]["image_name"] == "河流影像"


def test_edit_image_accepts_metadata_without_a_new_tif():
    client = _make_client()

    with patch.object(
        images,
        "edit_image_asset",
        new=AsyncMock(return_value=_image(image_name="新名称")),
    ) as edit:
        response = client.put("/api/images/1", data={"imageName": "新名称"})

    assert response.status_code == 200
    assert response.json()["data"]["image_name"] == "新名称"
    assert edit.await_args.args[3] is None
    assert edit.await_args.args[-1] is None


def test_edit_image_rejects_an_incomplete_boundary_group():
    client = _make_client()

    with patch.object(
        workflow.crud_images,
        "get_image_by_id",
        new=AsyncMock(return_value=_image()),
    ):
        response = client.put(
            "/api/images/1",
            files={
                "boundaryFiles": (
                    "river.shp",
                    b"shp",
                    "application/octet-stream",
                ),
            },
        )

    assert response.status_code == 422
    assert "完整 shp/dbf/prj" in response.json()["detail"]


def test_create_rejects_a_tif_whose_backend_md5_does_not_match():
    with TemporaryDirectory(dir=Path("tests")) as temp_dir:
        temp_root = Path(temp_dir)
        image_dir = temp_root / "images"
        image_dir.mkdir()
        with (
            patch.object(upload_sessions, "TMP_UPLOAD_DIR", temp_root),
            patch.object(workflow, "IMAGE_DIR", image_dir),
        ):
            client = _make_client()
            upload_id = client.post("/api/images/uploads", json={
                "fileName": "river.tif",
                "fileSize": 4,
                "chunkSize": 4,
                "totalChunks": 1,
                "fileHash": "0" * 32,
            }).json()["data"]["uploadId"]
            assert client.put(
                f"/api/images/uploads/{upload_id}/chunks/0",
                files={"chunk": ("0.part", b"abcd", "application/octet-stream")},
            ).status_code == 200

            response = client.post("/api/images", data={
                "uploadId": upload_id,
                "imageName": "河流影像",
                "resolution": "10",
                "captureDate": "2026-08-29",
                "satellite": "高分一号",
                "imageType": "多光谱",
                "regionCode": "330100",
            }, files=[
                ("boundaryFiles", ("river.shp", b"shp")),
                ("boundaryFiles", ("river.dbf", b"dbf")),
                ("boundaryFiles", ("river.prj", b"prj")),
            ])

        assert response.status_code == 422
        assert "MD5" in response.json()["detail"]


def test_missing_upload_session_returns_404_instead_of_500():
    with TemporaryDirectory(dir=Path("tests")) as temp_dir, patch.object(
        upload_sessions,
        "TMP_UPLOAD_DIR",
        Path(temp_dir),
    ):
        response = _make_client().post("/api/images", data={
            "uploadId": "a" * 32,
            "imageName": "河流影像",
            "resolution": "10",
            "captureDate": "2026-08-29",
            "satellite": "高分一号",
            "imageType": "多光谱",
            "regionCode": "330100",
        }, files=[
            ("boundaryFiles", ("river.shp", b"shp")),
            ("boundaryFiles", ("river.dbf", b"dbf")),
            ("boundaryFiles", ("river.prj", b"prj")),
        ])

    assert response.status_code == 404


def test_unknown_workflow_errors_do_not_leak_internal_details():
    with patch.object(
        images,
        "create_image_asset",
        new=AsyncMock(side_effect=RuntimeError(r"E:\\secret\\image.tif")),
    ):
        response = _make_client().post("/api/images", data={
            "uploadId": "a" * 32,
            "imageName": "河流影像",
            "resolution": "10",
            "captureDate": "2026-08-29",
            "satellite": "高分一号",
            "imageType": "多光谱",
            "regionCode": "330100",
        }, files=[
            ("boundaryFiles", ("river.shp", b"shp")),
            ("boundaryFiles", ("river.dbf", b"dbf")),
            ("boundaryFiles", ("river.prj", b"prj")),
        ])

    assert response.status_code == 500
    assert response.json()["detail"] == "影像处理失败，请稍后重试"
    assert "secret" not in response.text


def test_cleanup_reclaims_a_stale_completion_lock():
    with TemporaryDirectory(dir=Path("tests")) as temp_dir, patch.object(
        upload_sessions,
        "TMP_UPLOAD_DIR",
        Path(temp_dir),
    ):
        session_dir = Path(temp_dir) / ("a" * 32)
        session_dir.mkdir()
        meta_file = session_dir / upload_sessions.SESSION_META_FILE
        meta_file.write_text("{}", encoding="utf-8")
        lock_file = session_dir / upload_sessions.COMPLETE_LOCK_FILE
        lock_file.write_text("", encoding="utf-8")
        stale_time = time.time() - upload_sessions.UPLOAD_TTL_SECONDS - 1
        os.utime(lock_file, (stale_time, stale_time))

        upload_sessions.cleanup_expired_tmp_uploads()

        assert session_dir.exists()
        assert not lock_file.exists()


def test_create_recovers_a_committed_result_from_a_completing_session():
    with TemporaryDirectory(dir=Path("tests")) as temp_dir, patch.object(
        upload_sessions,
        "TMP_UPLOAD_DIR",
        Path(temp_dir),
    ):
        session = upload_sessions.begin_upload_session(
            7,
            file_name="river.tif",
            file_size=4,
            chunk_size=4,
            total_chunks=1,
            file_hash="0" * 32,
        )
        upload_id = session["upload_id"]
        meta = upload_sessions.load_session(upload_id)
        meta.update(status="completing", operation="create", result_image_id=1)
        upload_sessions.save_session(upload_id, meta)
        image = _image()
        db = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
        boundary_streams = [
            ("river.shp", BytesIO(b"shp")),
            ("river.dbf", BytesIO(b"dbf")),
            ("river.prj", BytesIO(b"prj")),
        ]

        with (
            patch.object(workflow.crud_images, "get_image_by_id", new=AsyncMock(return_value=image)),
            patch.object(workflow.crud_images, "create_image", new=AsyncMock()) as create_record,
        ):
            result = asyncio.run(workflow.create_image(
                db,
                7,
                upload_id,
                "河流影像",
                10,
                "2026-08-29",
                "高分一号",
                "多光谱",
                "330100",
                boundary_streams,
            ))

        assert result is image
        create_record.assert_not_awaited()
        assert upload_sessions.load_session(upload_id)["status"] == "completed"


def test_edit_recovers_a_committed_result_from_a_completing_session():
    with TemporaryDirectory(dir=Path("tests")) as temp_dir, patch.object(
        upload_sessions,
        "TMP_UPLOAD_DIR",
        Path(temp_dir),
    ):
        session = upload_sessions.begin_upload_session(
            7,
            file_name="river.tif",
            file_size=4,
            chunk_size=4,
            total_chunks=1,
            file_hash="0" * 32,
        )
        upload_id = session["upload_id"]
        meta = upload_sessions.load_session(upload_id)
        meta.update(
            status="completing",
            operation="edit:1",
            result_image_id=1,
            tif_path="uploads/images/new.tif",
            old_tif_path="uploads/images/old.tif",
            old_layer_name="old-layer",
            old_boundary_paths=[
                "uploads/shapefiles/old/boundary.shp",
                "uploads/shapefiles/old/boundary.dbf",
                "uploads/shapefiles/old/boundary.prj",
            ],
        )
        upload_sessions.save_session(upload_id, meta)
        image = _image()
        image.img_path = "uploads/images/new.tif"
        db = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())

        with (
            patch.object(workflow.crud_images, "get_image_by_id", new=AsyncMock(return_value=image)),
            patch.object(workflow.crud_images, "update_image_fields", new=AsyncMock()) as update,
            patch.object(workflow, "_remove_path") as remove_path,
            patch.object(workflow, "delete_geotiff_layer", new=AsyncMock()) as delete_layer,
        ):
            result = asyncio.run(workflow.edit_image(
                db,
                7,
                1,
                upload_id,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ))

        assert result is image
        update.assert_not_awaited()
        assert {call.args[0] for call in remove_path.call_args_list} == {
            "uploads/images/old.tif",
            "uploads/shapefiles/old/boundary.shp",
            "uploads/shapefiles/old/boundary.dbf",
            "uploads/shapefiles/old/boundary.prj",
        }
        delete_layer.assert_awaited_once_with("old-layer")
        assert upload_sessions.load_session(upload_id)["status"] == "completed"


def test_edit_persists_old_resource_cleanup_before_commit():
    with TemporaryDirectory(dir=Path("tests")) as temp_dir, patch.object(
        upload_sessions,
        "TMP_UPLOAD_DIR",
        Path(temp_dir),
    ):
        session = upload_sessions.begin_upload_session(
            7,
            file_name="river.tif",
            file_size=4,
            chunk_size=4,
            total_chunks=1,
            file_hash="0" * 32,
        )
        upload_id = session["upload_id"]
        image = _image()
        image.img_path = "uploads/images/old.tif"
        image.layer_name = "old-layer"
        image.boundary_files = [SimpleNamespace(
            shp_path="uploads/shapefiles/old/boundary.shp",
            dbf_path="uploads/shapefiles/old/boundary.dbf",
            prj_path="uploads/shapefiles/old/boundary.prj",
        )]

        async def update_fields(db, record, updates):
            for field, value in updates.items():
                setattr(record, field, value)
            return record

        async def commit_after_cleanup_is_durable():
            meta = upload_sessions.load_session(upload_id)
            assert meta["old_tif_path"] == "uploads/images/old.tif"
            assert meta["old_layer_name"] == "old-layer"
            assert "old_boundary_paths" not in meta

        db = SimpleNamespace(
            commit=AsyncMock(side_effect=commit_after_cleanup_is_durable),
            rollback=AsyncMock(),
        )
        with (
            patch.object(workflow.crud_images, "get_image_by_id", new=AsyncMock(return_value=image)),
            patch.object(workflow.crud_images, "update_image_fields", new=AsyncMock(side_effect=update_fields)),
            patch.object(
                workflow,
                "_merge_tif",
                return_value=(Path("uploads/images/new.tif"), [1, 2, 3, 4]),
            ),
            patch.object(workflow, "publish_geotiff_layer", new=AsyncMock(return_value="new-wms")),
            patch.object(workflow, "_remove_path") as remove_path,
            patch.object(workflow, "delete_geotiff_layer", new=AsyncMock()) as delete_layer,
        ):
            asyncio.run(workflow.edit_image(
                db,
                7,
                1,
                upload_id,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ))

        db.commit.assert_awaited_once()
        assert {call.args[0] for call in remove_path.call_args_list} == {
            "uploads/images/old.tif"
        }
        delete_layer.assert_awaited_once_with("old-layer")


def test_create_does_not_compensate_resources_after_commit_succeeds():
    with TemporaryDirectory(dir=Path("tests")) as temp_dir, patch.object(
        upload_sessions,
        "TMP_UPLOAD_DIR",
        Path(temp_dir),
    ):
        session = upload_sessions.begin_upload_session(
            7,
            file_name="river.tif",
            file_size=4,
            chunk_size=4,
            total_chunks=1,
            file_hash="0" * 32,
        )
        async def commit_after_intent_is_durable():
            meta = upload_sessions.load_session(session["upload_id"])
            assert meta["status"] == "completing"
            assert meta["operation"] == "create"
            assert meta["result_image_id"] == 1

        db = SimpleNamespace(
            commit=AsyncMock(side_effect=commit_after_intent_is_durable),
            rollback=AsyncMock(),
        )
        image = _image()
        boundary_streams = [
            ("river.shp", BytesIO(b"shp")),
            ("river.dbf", BytesIO(b"dbf")),
            ("river.prj", BytesIO(b"prj")),
        ]

        with (
            patch.object(workflow, "_merge_tif", return_value=(Path("new.tif"), [1, 2, 3, 4])),
            patch.object(
                workflow,
                "_save_boundary_files",
                return_value=(
                    {"shp_path": "new.shp", "dbf_path": "new.dbf", "prj_path": "new.prj"},
                    Path("new-boundaries"),
                ),
            ),
            patch.object(workflow.crud_images, "create_image", new=AsyncMock(return_value=image)),
            patch.object(workflow.crud_images, "create_boundary_files", new=AsyncMock()),
            patch.object(workflow.crud_images, "update_image_fields", new=AsyncMock()),
            patch.object(
                workflow.crud_images,
                "get_image_by_id",
                new=AsyncMock(side_effect=RuntimeError("refresh failed")),
            ),
            patch.object(workflow, "publish_geotiff_layer", new=AsyncMock(return_value="wms")),
            patch.object(workflow, "delete_geotiff_layer", new=AsyncMock()) as delete_layer,
            patch.object(workflow, "_remove_path") as remove_path,
        ):
            result = asyncio.run(workflow.create_image(
                db,
                7,
                session["upload_id"],
                "河流影像",
                10,
                "2026-08-29",
                "高分一号",
                "多光谱",
                "330100",
                boundary_streams,
            ))

        assert result is image
        db.commit.assert_awaited_once()
        db.rollback.assert_not_awaited()
        delete_layer.assert_not_awaited()
        remove_path.assert_not_called()
