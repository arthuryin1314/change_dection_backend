from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import UploadFile


async def save_upload_file(upload_file: UploadFile, dest_dir: Path) -> Path:
    suffix = Path(upload_file.filename or "").suffix
    save_path = (dest_dir / f"{uuid4().hex}{suffix}").resolve()

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        async with aiofiles.open(save_path, "wb") as f:
            while chunk := await upload_file.read(8192):
                await f.write(chunk)
    except Exception:
        save_path.unlink(missing_ok=True)
        raise

    return save_path
