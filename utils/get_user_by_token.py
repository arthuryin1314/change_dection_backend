from fastapi import Header, HTTPException
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from starlette import status

from config.db_config import get_db
from crud.users import get_user_by_token
from models.users import User


async def get_current_user(
        db: AsyncSession = Depends(get_db),
        authorization: str = Header(..., alias='Authorization',description='示例: Bearer token')
):
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='未授权')

    token = authorization.split(' ', 1)[1]
    user_id = await get_user_by_token(db, token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Token无效或已过期')

    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    db_user = result.scalar_one_or_none()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='用户不存在')

    return db_user