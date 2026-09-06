import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContentHash:
    sha256: str
    size: int
    mtime_ns: int


def resolve_content_sha256(
    path: str | Path,
    *,
    cached_sha256: str | None = None,
    cached_size: int | None = None,
    cached_mtime_ns: int | None = None,
) -> ContentHash:
    source = Path(path)
    stat = source.stat()
    if (
        cached_sha256 is not None
        and cached_size == stat.st_size
        and cached_mtime_ns == stat.st_mtime_ns
    ):
        return ContentHash(cached_sha256, stat.st_size, stat.st_mtime_ns)

    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return ContentHash(digest.hexdigest(), stat.st_size, stat.st_mtime_ns)
