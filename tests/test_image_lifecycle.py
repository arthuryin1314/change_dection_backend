import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("GEOSERVER_URL", "http://example.com/geoserver")
os.environ.setdefault("GEOSERVER_USER", "admin")
os.environ.setdefault("GEOSERVER_PASSWORD", "geoserver")
os.environ.setdefault("GEOSERVER_WORKSPACE", "ws")

from router.image import image_lifecycle as lifecycle
from router.image import upload_sessions


class FakeDatabase:
    def __init__(self):
        self.committed = False
        self.pending_image_deletion = False

    async def commit(self):
        self.committed = True


class FakeImageRepository:
    def __init__(self, image, single_deletion, bulk_deletion):
        self.image = image
        self.single_deletion = single_deletion
        self.bulk_deletion = bulk_deletion

    async def get_image_by_id(self, db, image_id, user_id):
        if image_id == self.image.id and user_id == self.image.user_id:
            return self.image
        return None

    async def delete_image_with_files(self, db, image_id, user_id):
        db.pending_image_deletion = True
        return self.single_deletion

    async def delete_images_by_user_with_files(self, db, user_id):
        db.pending_image_deletion = True
        return self.bulk_deletion


class FakeGeoServer:
    def __init__(self, db):
        self.db = db
        self.available = True
        self.deleted_layers = set()

    async def delete_layer(self, layer_name):
        assert self.db.committed
        if not self.available:
            raise RuntimeError("GeoServer unavailable")
        self.deleted_layers.add(layer_name)


class FakeSessionContext:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self.db

    async def __aexit__(self, exc_type, exc, traceback):
        return False


def _asset_environment(tmp_path):
    upload_root = tmp_path / "uploads"
    image_dir = upload_root / "images"
    boundary_dir = upload_root / "shapefiles"
    image_dir.mkdir(parents=True)
    boundary_dir.mkdir(parents=True)

    paths = {
        "tif": image_dir / "river.tif",
        "shp": boundary_dir / "river.shp",
        "dbf": boundary_dir / "river.dbf",
        "prj": boundary_dir / "river.prj",
    }
    for path in paths.values():
        path.write_bytes(b"asset")

    boundary = SimpleNamespace(
        shp_path=str(paths["shp"]),
        dbf_path=str(paths["dbf"]),
        prj_path=str(paths["prj"]),
    )
    image = SimpleNamespace(
        id=1,
        user_id=7,
        img_path=str(paths["tif"]),
        layer_name="river-layer",
        boundary_files=[boundary],
    )
    boundary_paths = [
        str(paths["shp"]),
        str(paths["dbf"]),
        str(paths["prj"]),
    ]
    single_deletion = {
        "image_id": image.id,
        "img_path": image.img_path,
        "boundary_paths": boundary_paths,
    }
    bulk_deletion = {
        "deleted_count": 1,
        "image_ids": [image.id],
        "img_paths": [image.img_path],
        "boundary_paths": boundary_paths,
        "layer_names": [image.layer_name],
    }
    return paths, image, single_deletion, bulk_deletion


def _install_system_seams(monkeypatch, repository, geoserver):
    monkeypatch.setattr(lifecycle, "crud_images", repository)
    monkeypatch.setattr(lifecycle, "delete_geotiff_layer", geoserver.delete_layer)


def test_delete_image_cleans_all_assets_after_database_commit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    paths, image, single_deletion, bulk_deletion = _asset_environment(tmp_path)
    db = FakeDatabase()
    repository = FakeImageRepository(image, single_deletion, bulk_deletion)
    geoserver = FakeGeoServer(db)
    _install_system_seams(monkeypatch, repository, geoserver)

    real_unlink = Path.unlink

    def unlink_after_commit(path, missing_ok=False):
        if path.exists():
            assert db.committed
        return real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink_after_commit)

    result = asyncio.run(lifecycle.delete_image(db, 7, image.id))

    assert result is True
    assert db.committed
    assert not paths["tif"].exists()
    assert not paths["shp"].exists()
    assert not paths["dbf"].exists()
    assert not paths["prj"].exists()
    assert geoserver.deleted_layers == {image.layer_name}


def test_cleanup_failure_is_successfully_retryable(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    paths, image, single_deletion, bulk_deletion = _asset_environment(tmp_path)
    db = FakeDatabase()
    repository = FakeImageRepository(image, single_deletion, bulk_deletion)
    geoserver = FakeGeoServer(db)
    _install_system_seams(monkeypatch, repository, geoserver)

    prepared = asyncio.run(lifecycle.prepare_user_image_deletion(db, image.user_id))
    asyncio.run(db.commit())
    geoserver.available = False

    first_attempt = asyncio.run(lifecycle.finish_cleanup(prepared.operation_id))

    assert first_attempt is False
    assert db.committed

    geoserver.available = True
    second_attempt = asyncio.run(lifecycle.finish_cleanup(prepared.operation_id))

    assert second_attempt is True
    assert not paths["tif"].exists()
    assert not paths["shp"].exists()
    assert not paths["dbf"].exists()
    assert not paths["prj"].exists()
    assert geoserver.deleted_layers == {image.layer_name}


def test_prepare_user_image_deletion_does_not_clean_external_assets(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    paths, image, single_deletion, bulk_deletion = _asset_environment(tmp_path)
    db = FakeDatabase()
    repository = FakeImageRepository(image, single_deletion, bulk_deletion)
    geoserver = FakeGeoServer(db)
    _install_system_seams(monkeypatch, repository, geoserver)

    result = asyncio.run(lifecycle.prepare_user_image_deletion(db, image.user_id))

    assert result.deleted_count == 1
    assert isinstance(result.operation_id, str)
    assert result.operation_id
    assert paths["tif"].exists()
    assert paths["shp"].exists()
    assert paths["dbf"].exists()
    assert paths["prj"].exists()
    assert geoserver.deleted_layers == set()


def test_delete_image_rejects_assets_outside_upload_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    paths, image, single_deletion, bulk_deletion = _asset_environment(tmp_path)
    outside_path = tmp_path / "outside" / "secret.tif"
    outside_path.parent.mkdir()
    outside_path.write_bytes(b"secret")
    image.img_path = str(outside_path)
    single_deletion["img_path"] = image.img_path

    db = FakeDatabase()
    repository = FakeImageRepository(image, single_deletion, bulk_deletion)
    geoserver = FakeGeoServer(db)
    _install_system_seams(monkeypatch, repository, geoserver)

    with pytest.raises(ValueError):
        asyncio.run(lifecycle.delete_image(db, image.user_id, image.id))

    assert outside_path.exists()
    assert paths["shp"].exists()
    assert not db.committed
    assert not db.pending_image_deletion
    assert geoserver.deleted_layers == set()


def test_directory_cleanup_is_limited_to_shapefile_assets(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    allowed = tmp_path / "uploads" / "shapefiles" / "new-boundary"
    outside = tmp_path / "outside"
    allowed.mkdir(parents=True)
    outside.mkdir()
    (allowed / "boundary.shp").write_bytes(b"asset")
    (outside / "keep.txt").write_bytes(b"keep")

    assert lifecycle._remove_permanent_asset_directory(str(outside)) is False
    assert outside.exists()
    assert lifecycle._remove_permanent_asset_directory(str(allowed)) is True
    assert not allowed.exists()


def test_recovery_removes_uncommitted_boundary_replacement(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(upload_sessions, "TMP_UPLOAD_DIR", tmp_path / "uploads" / "tmp")
    paths, image, single_deletion, bulk_deletion = _asset_environment(tmp_path)
    new_dir = tmp_path / "uploads" / "shapefiles" / "new-boundary"
    new_dir.mkdir()
    new_paths = []
    for suffix in ("shp", "dbf", "prj"):
        path = new_dir / f"boundary.{suffix}"
        path.write_bytes(b"new")
        new_paths.append(str(path))

    repository = FakeImageRepository(image, single_deletion, bulk_deletion)
    monkeypatch.setattr(lifecycle, "crud_images", repository)
    operation_id = lifecycle._prepare_cleanup(
        "replacement",
        image.user_id,
        [image.id],
        [],
        [str(paths["shp"]), str(paths["dbf"]), str(paths["prj"])],
        [],
        new_paths,
    )

    asyncio.run(lifecycle.recover_cleanup_operations(
        lambda: FakeSessionContext(FakeDatabase())
    ))

    assert all(paths[suffix].exists() for suffix in ("shp", "dbf", "prj"))
    assert not new_dir.exists()
    assert not lifecycle._operation_path(operation_id).exists()


def test_completed_edit_retries_old_layer_cleanup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(upload_sessions, "TMP_UPLOAD_DIR", tmp_path / "uploads" / "tmp")
    db = FakeDatabase()
    db.committed = True
    geoserver = FakeGeoServer(db)
    monkeypatch.setattr(lifecycle, "delete_geotiff_layer", geoserver.delete_layer)

    session = upload_sessions.begin_upload_session(
        7,
        file_name="river.tif",
        file_size=4,
        chunk_size=4,
        total_chunks=1,
        file_hash="e2fc714c4727ee9395f324cd2e7f331f",
    )
    upload_id = session["upload_id"]
    meta = upload_sessions.load_session(upload_id)
    meta.update(
        status="completed",
        operation="edit:1",
        result_image_id=1,
        old_layer_name="old-river-layer",
    )
    upload_sessions.save_session(upload_id, meta)

    geoserver.available = False
    asyncio.run(lifecycle.retry_pending_cleanup())
    assert upload_sessions.load_session(upload_id)["old_layer_name"] == "old-river-layer"

    geoserver.available = True
    asyncio.run(lifecycle.retry_pending_cleanup())
    assert "old_layer_name" not in upload_sessions.load_session(upload_id)
    assert geoserver.deleted_layers == {"old-river-layer"}
