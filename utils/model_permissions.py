from typing import Annotated

from fastapi import Depends, HTTPException

from models.users import User
from utils.get_user_by_token import get_current_user


async def require_model_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="仅管理员可以修改模型目录")
    return user
