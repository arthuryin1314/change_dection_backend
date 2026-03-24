from sqlalchemy.ext.asyncio import AsyncSession
from models.users import User
from sqlalchemy import select, update
from schemas.users import UserRequest, UserUpdateRequest
from utils.security import get_password_hash, verify_password
from utils.jwt_utils import create_access_token
from crud import images as crud_images


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


async def create_token(db:AsyncSession,user_id:int):
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    if not user:
        raise ValueError("用户不存在")
    return create_access_token(user.id, user.token_version)


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
    stmt = (
        update(User)
        .where(User.id == user_id)
        .values(token_version=User.token_version + 1)
        .returning(User.id)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none() is not None

async def delete_user(db:AsyncSession,user_id:int):
    deleted_images = await crud_images.delete_images_by_user_with_files(db, user_id)

    query_user = select(User).where(User.id == user_id)
    result = await db.execute(query_user)
    db_user = result.scalar_one_or_none()
    if not db_user:
        return None

    await db.delete(db_user)
    await db.flush()

    return {
        "user_id": user_id,
        "deleted_images": deleted_images,
    }
