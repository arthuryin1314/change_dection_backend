import asyncio
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from fastapi import UploadFile


CHUNK_SIZE = 1024 * 1024


class ModelAssetTooLargeError(ValueError):
    def __init__(self, max_size: int):
        super().__init__(f"model asset exceeds {max_size} bytes")
        self.max_size = max_size


def _copy_with_limit(source: BinaryIO, save_path: Path, max_size: int) -> None:
    size = 0
    with save_path.open("wb") as output:
        while chunk := source.read(CHUNK_SIZE):
            size += len(chunk)
            if size > max_size:
                raise ModelAssetTooLargeError(max_size)
            output.write(chunk)


async def save_upload_file(
    upload_file: UploadFile,
    root_dir: Path,
    user_id: int,
    max_size: int,
) -> Path:
    if upload_file.filename is None:
        raise ValueError("uploaded asset has no filename")
    filename = Path(upload_file.filename.replace("\\", "/")).name
    if not filename:
        raise ValueError("uploaded asset has no basename")
    revision_dir = root_dir.resolve() / str(user_id) / uuid4().hex
    save_path = revision_dir / filename

    revision_dir.mkdir(parents=True, exist_ok=False)
    try:
        await upload_file.seek(0)
        await asyncio.to_thread(
            _copy_with_limit,
            upload_file.file,
            save_path,
            max_size,
        )
    except Exception:
        save_path.unlink(missing_ok=True)
        try:
            revision_dir.rmdir()
        except OSError:
            pass
        raise

    return save_path
