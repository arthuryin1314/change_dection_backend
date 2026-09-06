from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.db_config import AsyncSessionLocal, async_engine
from models.Base import Base
from models import ml_models as ml_model_metadata
from models import classification_results as classification_result_metadata
from router import users
from utils.exception_handler import register_exception_handlers
from router.image import images
from router.image import image_lifecycle
from router.image import upload_sessions
from router import segment
from router import ml_models
from router import identification_results
from services.generation_lifecycle import recover_expired_processing_results
from datetime import datetime, timezone


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await recover_expired_processing_results(
        AsyncSessionLocal,
        now=datetime.now(timezone.utc),
    )
    await image_lifecycle.recover_cleanup_operations(AsyncSessionLocal)
    upload_sessions.start_tmp_cleanup_task()
    image_lifecycle.start_cleanup_task()
    try:
        yield
    finally:
        await image_lifecycle.stop_cleanup_task()
        await upload_sessions.stop_tmp_cleanup_task()
        await identification_results.stop_generation_tasks()


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
app.include_router(identification_results.router)
register_exception_handlers(app)
