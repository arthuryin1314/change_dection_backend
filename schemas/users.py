from pydantic import BaseModel
from fastapi import Query
class UserRequest(BaseModel):
    name: str = Query(..., description="用户名")
    telNum: str = Query(..., description="电话号码")
    password: str = Query(..., description="密码")

class UserLoginRequest(BaseModel):
    telNum: str = Query(..., description="电话号码")
    password: str = Query(..., description="密码")