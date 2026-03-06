import uuid
from sqlalchemy import func
from datetime import timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from models.users import User, UserToken
from sqlalchemy import select
from schemas.users import UserRequest
from utils.security import get_password_hash
async def get_user_by_username(db:AsyncSession,username:str):
    query = select(User).where(User.username == username)
    result = await db.execute(query)
    return result.scalar_one_or_none()

async def get_user_by_telNum(db:AsyncSession,telNum:str):
    query = select(User).where(User.phone == telNum)
    result = await db.execute(query)
    return result.scalar_one_or_none()

async def create_user(db:AsyncSession,user_data:UserRequest):
    hashed_password = get_password_hash(user_data.password)
    user = User(username=user_data.name, password=hashed_password, phone=user_data.telNum)
    db.add(user)
    await db.flush()
    return user

async def createToken(db:AsyncSession,user_id:int):
    token = str(uuid.uuid4())
    query=select(UserToken).where(UserToken.user_id == user_id)
    result = await db.execute(query)
    existing_token = result.scalar_one_or_none()
    if existing_token:
        existing_token.token = token
        existing_token.created_at = func.now()
        existing_token.expires_at = func.now() + timedelta(hours=1)
        await db.flush()
        return existing_token.token
    user_token = UserToken(user_id=user_id, token=token)
    db.add(user_token)
    await db.flush()
    return token