from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import users
from utils.exception_handler import register_exception_handlers
from router import images


@asynccontextmanager
async def lifespan(app: FastAPI):
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
register_exception_handlers(app)
