import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrozenFile:
    path: Path
    sha256: str
    size: int


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def freeze_file(source: str | Path, directory: str | Path, filename: str) -> FrozenFile:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    target_directory = Path(directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    target = target_directory / filename
    try:
        os.link(source_path, target)
    except OSError:
        shutil.copy2(source_path, target)
    sha256, size = _sha256(target)
    return FrozenFile(path=target, sha256=sha256, size=size)
