import asyncio
import threading
from contextlib import contextmanager

import numpy as np
import pytest
import rasterio
from affine import Affine

from services import classification_generation as generation


def _write_source(path, *, width=6, height=5):
    data = np.full((3, height, width), 7, dtype=np.uint8)
    data[:, 0, 0] = 0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs="EPSG:4528",
        transform=Affine(0.8, 0, 500000, 0, -0.8, 3200000),
        nodata=0,
    ) as dataset:
        dataset.write(data)


@contextmanager
def _fake_predictor(weight_file_path, *, weight_sha256):
    def predict(rgb):
        return np.full(rgb.shape[:2], 5, dtype=np.uint8)

    yield predict


class RecordingLifecycle:
    def __init__(self):
        self.succeeded = None
        self.failure = None
        self.heartbeats = 0

    async def heartbeat(self):
        self.heartbeats += 1

    async def mark_succeeded(self, outcome):
        assert outcome.stored.classes_path.exists()
        assert outcome.stored.valid_mask_path.exists()
        self.succeeded = outcome

    async def mark_failed(self, detail):
        self.failure = detail


def _request(tmp_path, source_path):
    return generation.GenerationRequest(
        result_id="result-1",
        image_path=source_path,
        weight_file_path="weights/model.pth",
        weight_sha256="b" * 64,
        storage_root=tmp_path / "results",
    )


def test_programmatic_generation_persists_native_grid_and_spatial_metadata(
    tmp_path,
    monkeypatch,
):
    source_path = tmp_path / "source.tif"
    _write_source(source_path)
    monkeypatch.setattr(generation, "model_tile_predictor", _fake_predictor)

    outcome = generation.generate_classification_files(
        _request(tmp_path, source_path)
    )

    assert outcome.metadata.width == 6
    assert outcome.metadata.height == 5
    assert outcome.metadata.crs == "EPSG:4528"
    assert outcome.metadata.transform == (0.8, 0.0, 500000.0, 0.0, -0.8, 3200000.0)
    assert outcome.metadata.resolution == (0.8, 0.8)
    assert outcome.metrics.effective_tiles == 1
    assert outcome.metrics.source_read_seconds > 0
    assert outcome.metrics.inference_seconds > 0
    assert outcome.metrics.compressed_write_seconds > 0
    assert outcome.metrics.total_seconds > 0
    assert outcome.metrics.peak_rss_bytes > 0
    assert outcome.metrics.peak_gpu_bytes == 0
    with rasterio.open(outcome.stored.classes_path) as classes_ds:
        classes = classes_ds.read(1)
    with rasterio.open(outcome.stored.valid_mask_path) as valid_ds:
        valid = valid_ds.read(1)
    assert classes[0, 0] == 0
    assert valid[0, 0] == 0
    assert np.all(classes[valid == 1] == 5)


def test_programmatic_generation_writes_every_streamed_band(tmp_path, monkeypatch):
    source_path = tmp_path / "source.tif"
    _write_source(source_path, height=900)
    monkeypatch.setattr(generation, "model_tile_predictor", _fake_predictor)

    outcome = generation.generate_classification_files(
        _request(tmp_path, source_path)
    )

    with rasterio.open(outcome.stored.classes_path) as classes_ds:
        classes = classes_ds.read(1)
    with rasterio.open(outcome.stored.valid_mask_path) as valid_ds:
        valid = valid_ds.read(1)

    assert valid[0, 1] == 1
    assert classes[0, 1] == 5
    assert valid[-1, 1] == 1
    assert classes[-1, 1] == 5


def test_database_success_transition_happens_only_after_files_are_readable(
    tmp_path,
    monkeypatch,
):
    source_path = tmp_path / "source.tif"
    _write_source(source_path)
    monkeypatch.setattr(generation, "model_tile_predictor", _fake_predictor)
    lifecycle = RecordingLifecycle()

    outcome = asyncio.run(
        generation.run_generation(_request(tmp_path, source_path), lifecycle)
    )

    assert lifecycle.succeeded is outcome
    assert lifecycle.failure is None


def test_generation_failure_is_marked_failed_without_visible_files(
    tmp_path,
    monkeypatch,
):
    source_path = tmp_path / "source.tif"
    _write_source(source_path)

    @contextmanager
    def failing_predictor(*args, **kwargs):
        def predict(rgb):
            raise RuntimeError("inference failed")

        yield predict

    monkeypatch.setattr(generation, "model_tile_predictor", failing_predictor)
    lifecycle = RecordingLifecycle()

    with pytest.raises(RuntimeError, match="inference failed"):
        asyncio.run(
            generation.run_generation(_request(tmp_path, source_path), lifecycle)
        )

    assert lifecycle.succeeded is None
    assert lifecycle.failure == "inference failed"
    assert not (_request(tmp_path, source_path).storage_root / "result-1").exists()


def test_retry_replaces_a_corrupt_published_directory(tmp_path, monkeypatch):
    source_path = tmp_path / "source.tif"
    _write_source(source_path)
    request = _request(tmp_path, source_path)
    corrupt = request.storage_root / request.result_id
    corrupt.mkdir(parents=True)
    (corrupt / "classes.tif").write_bytes(b"truncated")
    (corrupt / "valid_mask.tif").write_bytes(b"truncated")
    monkeypatch.setattr(generation, "model_tile_predictor", _fake_predictor)

    outcome = generation.generate_classification_files(request)

    with rasterio.open(outcome.stored.classes_path) as dataset:
        assert dataset.width == 6
        assert dataset.height == 5
    assert list(request.storage_root.glob(".result-1.*.corrupt")) == []


def test_failed_rebuild_restores_quarantined_result(tmp_path, monkeypatch):
    source_path = tmp_path / "source.tif"
    _write_source(source_path)
    request = _request(tmp_path, source_path)
    existing = request.storage_root / request.result_id
    existing.mkdir(parents=True)
    classes_path = existing / "classes.tif"
    mask_path = existing / "valid_mask.tif"
    classes_path.write_bytes(b"original classes")
    mask_path.write_bytes(b"original mask")
    monkeypatch.setattr(
        generation,
        "validate_stored_classification",
        lambda *args, **kwargs: False,
    )

    @contextmanager
    def failing_predictor(*args, **kwargs):
        def predict(rgb):
            raise RuntimeError("rebuild failed")

        yield predict

    monkeypatch.setattr(generation, "model_tile_predictor", failing_predictor)

    with pytest.raises(RuntimeError, match="rebuild failed"):
        generation.generate_classification_files(request)

    assert classes_path.read_bytes() == b"original classes"
    assert mask_path.read_bytes() == b"original mask"
    assert list(request.storage_root.glob(".result-1.*.corrupt")) == []


def test_failed_old_worker_never_replaces_a_concurrently_published_result(
    tmp_path,
    monkeypatch,
):
    source_path = tmp_path / "source.tif"
    _write_source(source_path)
    request = _request(tmp_path, source_path)
    existing = request.storage_root / request.result_id
    existing.mkdir(parents=True)
    (existing / "old-marker").write_text("corrupt", encoding="utf-8")
    monkeypatch.setattr(
        generation,
        "validate_stored_classification",
        lambda *args, **kwargs: False,
    )

    @contextmanager
    def losing_predictor(*args, **kwargs):
        def predict(rgb):
            existing.mkdir()
            (existing / "new-marker").write_text("published", encoding="utf-8")
            raise RuntimeError("old worker lost")

        yield predict

    monkeypatch.setattr(generation, "model_tile_predictor", losing_predictor)

    with pytest.raises(RuntimeError, match="old worker lost"):
        generation.generate_classification_files(request)

    assert (existing / "new-marker").read_text(encoding="utf-8") == "published"
    assert not (existing / "old-marker").exists()
    assert list(request.storage_root.glob(".result-1.*.corrupt")) == []


def test_existing_result_reuse_uses_metadata_validation_only(tmp_path, monkeypatch):
    source_path = tmp_path / "source.tif"
    _write_source(source_path)
    request = _request(tmp_path, source_path)
    existing = request.storage_root / request.result_id
    existing.mkdir(parents=True)
    (existing / "classes.tif").write_bytes(b"placeholder")
    (existing / "valid_mask.tif").write_bytes(b"placeholder")
    calls = []

    def validate(*args, **kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(generation, "validate_stored_classification", validate)

    outcome = generation.generate_classification_files(request)

    assert outcome.stored.directory == existing
    assert calls == [{"verify_pixels": False}]


def test_cancelling_generation_stops_worker_before_publish(tmp_path, monkeypatch):
    source_path = tmp_path / "source.tif"
    _write_source(source_path, height=900)
    request = _request(tmp_path, source_path)
    entered = threading.Event()
    release = threading.Event()

    @contextmanager
    def blocking_predictor(*args, **kwargs):
        def predict(rgb):
            entered.set()
            assert release.wait(5)
            return np.full(rgb.shape[:2], 5, dtype=np.uint8)

        yield predict

    monkeypatch.setattr(generation, "model_tile_predictor", blocking_predictor)
    lifecycle = RecordingLifecycle()

    async def cancel_running_generation():
        task = asyncio.create_task(generation.run_generation(request, lifecycle))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_running_generation())

    assert lifecycle.failure == "服务关闭，任务可重新提交"
    assert not (request.storage_root / request.result_id).exists()


def test_heartbeat_failure_stops_worker_and_prevents_publish(tmp_path, monkeypatch):
    source_path = tmp_path / "source.tif"
    _write_source(source_path, height=900)
    request = _request(tmp_path, source_path)
    monkeypatch.setattr(generation, "model_tile_predictor", _fake_predictor)
    lifecycle = RecordingLifecycle()

    async def failing_heartbeat():
        raise RuntimeError("lease renewal failed")

    lifecycle.heartbeat = failing_heartbeat

    with pytest.raises(RuntimeError, match="lease renewal failed"):
        asyncio.run(generation.run_generation(request, lifecycle))

    assert lifecycle.failure == "lease renewal failed"
    assert lifecycle.succeeded is None
    assert not (request.storage_root / request.result_id).exists()


def test_lost_lease_at_publication_fences_stale_worker(tmp_path, monkeypatch):
    source_path = tmp_path / "source.tif"
    _write_source(source_path)
    request = _request(tmp_path, source_path)
    monkeypatch.setattr(generation, "model_tile_predictor", _fake_predictor)
    lifecycle = RecordingLifecycle()

    async def lose_lease_on_publication():
        lifecycle.heartbeats += 1
        if lifecycle.heartbeats == 2:
            raise RuntimeError("lease lost before publish")

    lifecycle.heartbeat = lose_lease_on_publication

    with pytest.raises(RuntimeError, match="lease lost before publish"):
        asyncio.run(generation.run_generation(request, lifecycle))

    assert lifecycle.failure == "lease lost before publish"
    assert lifecycle.succeeded is None
    assert not (request.storage_root / request.result_id).exists()
