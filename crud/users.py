import uuid
from sqlalchemy import func
from datetime import timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from models.users import User, UserToken
from sqlalchemy import select
from schemas.users import UserRequest, UserUpdateRequest
from utils.security import get_password_hash, verify_password


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
        existing_token.create_at = func.now()
        existing_token.expire_at = func.now() + timedelta(days=1)
        await db.flush()
        return existing_token.token
    user_token = UserToken(user_id=user_id, token=token)
    db.add(user_token)
    await db.flush()
    return token


async def get_user_by_token(db:AsyncSession,token:str):
    query = select(UserToken).where(
        UserToken.token == token,
        UserToken.expire_at > func.now()
    )
    result = await db.execute(query)
    db_user_token = result.scalar_one_or_none()
    if db_user_token:
        return db_user_token.user_id
    return None

async def update_user_info(db:AsyncSession,user_info:UserUpdateRequest,user_id:int):
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    if user:
        user.username = user_info.name
        user.phone = user_info.telNum
        await db.flush()
        return user
    return None


async def check_old_password(
        db:AsyncSession,
        old_password:str,
        user_id:int
):
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    db_user = result.scalar_one_or_none()
    if verify_password(old_password, db_user.password):
        return True
    return False

async def update_password(
        db:AsyncSession,
        new_password:str,
        user_id:int
):
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    db_user = result.scalar_one_or_none()
    if db_user:
        db_user.password = get_password_hash(new_password)
        await db.flush()
        return True
    return False

async def clear_user_token(db:AsyncSession,user_id:int):
    query = select(UserToken).where(UserToken.user_id == user_id)
    result = await db.execute(query)
    user_token = result.scalar_one_or_none()
    if not user_token:
        return False
    await db.delete(user_token)
    await db.flush()
    return True
