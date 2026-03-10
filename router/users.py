from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from schemas.users import UserRequest, UserLoginRequest
from config.db_config import get_db
from crud.users import get_user_by_username,create_user, createToken,get_user_by_telNum
from utils.get_user_by_token import get_current_user
from utils.response import success_response
from utils.security import verify_password

router = APIRouter(prefix='/api/users', tags=['users'])


@router.post('/register',summary='注册用户')
async def register_user(user_data: UserRequest, db: AsyncSession = Depends(get_db)):
    existing_user = await get_user_by_username(db, user_data.name)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户已经存在")
    user = await create_user(db,user_data)
    token = await createToken(db,user.id)
    await db.commit()
    return success_response(message='注册成功', data={
        'token':token,
         "userInfo": {
            "id": user.id,
            "username": user.username,
            "telNum": user.phone,
        },
    }
    )

@router.post('/login',summary='用户登录')
async def login_user(form_data:UserLoginRequest, db: AsyncSession = Depends(get_db)):
    #逻辑:查找用户存不存在,如果存在,验证密码是否正确,如果正确,生成token返回
    db_user = await get_user_by_telNum(db, form_data.telNum)
    if not db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户不存在")
    if not verify_password(form_data.password, db_user.password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="密码错误")
    token = await createToken(db, db_user.id)
    await db.commit()
    return success_response(message='登录成功', data={
        'token': token
    })

@router.get('/info',summary='获取用户信息')
async def get_user_info(current_user = Depends(get_current_user)):
    return success_response(message='获取用户信息成功', data={
        'id': current_user.id,
        'username': current_user.username,
        'telNum': current_user.phone,
    })
