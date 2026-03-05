from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from schemas.users import UserRequest
from config.db_config import get_db
from crud.users import get_user_by_username,create_user, createToken

router = APIRouter(prefix='/api/users', tags=['users'])


@router.post('/register')
async def register_user(user_data: UserRequest, db: AsyncSession = Depends(get_db)):
    existing_user = await get_user_by_username(db, user_data.name)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户已经存在")
    user = await create_user(db,user_data)
    token = await createToken(db,user.id)
    return {
        "code": 200,
        "message": "注册成功",
        "data": {
            'token':token,
            "userInfo": {
                "id": user.id,
                "username": user.username,
                "telNum": user.phone,
            },
        },
    }