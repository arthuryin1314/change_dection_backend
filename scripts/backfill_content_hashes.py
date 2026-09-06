import asyncio
from pathlib import Path
import sys

from dotenv import load_dotenv


load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from config.db_config import AsyncSessionLocal
from models.images import Image
from models.ml_models import MLModel
import models.users  # noqa: F401  # register user_info for ORM foreign keys
from utils.content_hash import resolve_content_sha256


async def _hash(path, sha256, size, mtime_ns):
    return await asyncio.to_thread(
        resolve_content_sha256,
        Path(path),
        cached_sha256=sha256,
        cached_size=size,
        cached_mtime_ns=mtime_ns,
    )


async def backfill() -> tuple[int, int]:
    image_count = 0
    model_count = 0
    async with AsyncSessionLocal() as session:
        images = list((await session.scalars(select(Image))).all())
        for image in images:
            if image.img_path is None:
                continue
            content_hash = await _hash(
                image.img_path,
                image.content_sha256,
                image.content_sha256_size,
                image.content_sha256_mtime_ns,
            )
            image.content_sha256 = content_hash.sha256
            image.content_sha256_size = content_hash.size
            image.content_sha256_mtime_ns = content_hash.mtime_ns
            image_count += 1

        models = list((await session.scalars(select(MLModel))).all())
        for model in models:
            content_hash = await _hash(
                model.weight_file_path,
                model.weight_content_sha256,
                model.weight_content_sha256_size,
                model.weight_content_sha256_mtime_ns,
            )
            model.weight_content_sha256 = content_hash.sha256
            model.weight_content_sha256_size = content_hash.size
            model.weight_content_sha256_mtime_ns = content_hash.mtime_ns
            model_count += 1

        await session.commit()
    return image_count, model_count


if __name__ == "__main__":
    images, models = asyncio.run(backfill())
    print(f"content hashes ready: images={images}, models={models}")
