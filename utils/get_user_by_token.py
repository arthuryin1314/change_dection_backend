from fastapi import Header, HTTPException
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from starlette import status

from config.db_config import get_db
from models.users import User
from utils.jwt_utils import verify_access_token


async def get_current_user(
        db: AsyncSession = Depends(get_db),
        authorization: str = Header(..., alias='Authorization',description='示例: Bearer token')
):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")
    token = authorization.split(' ', 1)[1].strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")

    payload = verify_access_token(token)
    query = select(User).where(User.id == payload["user_id"])
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token过期或无效")
    if user.token_version != payload["token_version"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token过期或无效")
    return user
