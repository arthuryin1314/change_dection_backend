from dotenv import load_dotenv
load_dotenv()

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.db_config import async_engine
from models.Base import Base
from models import ml_models as ml_model_metadata
from router import users
from utils.exception_handler import register_exception_handlers
from router import images
from router import segment
from router import ml_models
from utils import deeplab_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await asyncio.to_thread(deeplab_service.load_model)
    images.start_tmp_cleanup_task()
    try:
        yield
    finally:
        await images.stop_tmp_cleanup_task()


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
