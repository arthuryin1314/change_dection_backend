from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import users
from utils.exception_handler import register_exception_handlers
from router import images
origins = ['http://localhost:5173']
app = FastAPI()
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
