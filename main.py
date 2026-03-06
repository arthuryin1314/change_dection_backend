from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import users
from utils.exception_handler import register_exception_handlers

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
register_exception_handlers(app)
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
