from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.db_config import AsyncSessionLocal, async_engine
from models.Base import Base
from models import ml_models as ml_model_metadata
from router import users
from utils.exception_handler import register_exception_handlers
from router.image import images
from router.image import image_lifecycle
from router.image import upload_sessions
from router import segment
from router import ml_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await image_lifecycle.recover_cleanup_operations(AsyncSessionLocal)
    upload_sessions.start_tmp_cleanup_task()
    image_lifecycle.start_cleanup_task()
    try:
        yield
    finally:
        await image_lifecycle.stop_cleanup_task()
        await upload_sessions.stop_tmp_cleanup_task()


origins = ['http://localhost:5173']
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.include_router(users.router)
app.include_router(images.router)
app.include_router(segment.router)
app.include_router(ml_models.router)
register_exception_handlers(app)
