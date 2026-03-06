from pydantic import BaseModel

class UserRequest(BaseModel):
    name: str
    telNum: str
    password: str

class UserLoginRequest(BaseModel):
    telNum: str
    password: str