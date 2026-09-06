from pathlib import Path

from utils.content_hash import resolve_content_sha256


def test_same_bytes_have_same_identity_across_different_source_ids(tmp_path):
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    first.write_bytes(b"same raster bytes")
    second.write_bytes(b"same raster bytes")

    assert resolve_content_sha256(first).sha256 == resolve_content_sha256(second).sha256


def test_unchanged_file_uses_persisted_stat_key_without_reading(tmp_path, monkeypatch):
    path = tmp_path / "large.tif"
    path.write_bytes(b"content")
    initial = resolve_content_sha256(path)

    def unexpected_open(*args, **kwargs):
        raise AssertionError("unchanged cached file must not be read again")

    monkeypatch.setattr(Path, "open", unexpected_open)
    cached = resolve_content_sha256(
        path,
        cached_sha256=initial.sha256,
        cached_size=initial.size,
        cached_mtime_ns=initial.mtime_ns,
    )

    assert cached == initial


def test_stat_change_recomputes_digest(tmp_path):
    path = tmp_path / "source.tif"
    path.write_bytes(b"before")
    initial = resolve_content_sha256(path)
    path.write_bytes(b"after!")

    changed = resolve_content_sha256(
        path,
        cached_sha256=initial.sha256,
        cached_size=initial.size,
        cached_mtime_ns=initial.mtime_ns,
    )

    assert changed.sha256 != initial.sha256
