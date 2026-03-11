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
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")
    token = authorization.split(' ')[1]

    user_id = await get_user_by_token(db, token)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token过期或无效")
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户未被发现")
    return user
