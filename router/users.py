from pathlib import Path
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from schemas.users import UserRequest, UserLoginRequest, UserUpdateRequest, UserUpdatePassword
from config.db_config import get_db
from crud.users import create_user, create_token, get_user_by_telNum, update_user_info as crud_update_user_info, \
    check_old_password, clear_user_token, update_password as crud_update_password, delete_user as crud_delete_user
from utils.get_user_by_token import get_current_user
from utils.response import success_response
from utils.security import verify_password

router = APIRouter(prefix='/api/users', tags=['users'])
logger = logging.getLogger(__name__)


@router.post('/register', summary='注册用户')
async def register_user(user_data: UserRequest, db: AsyncSession = Depends(get_db)):
    try:
        user = await create_user(db, user_data)
        token = await create_token(db, user.id)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名或手机号已被注册")

    return success_response(message='注册成功', data={
        'token': token,
        "userInfo": {
            "id": user.id,
            "username": user.username,
            "telNum": user.phone,
        },
    })


@router.post('/login', summary='用户登录')
async def login_user(form_data: UserLoginRequest, db: AsyncSession = Depends(get_db)):
    # 逻辑:查找用户存不存在,如果存在,验证密码是否正确,如果正确,生成token返回
    db_user = await get_user_by_telNum(db, form_data.telNum)
    if not db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户不存在")
    if not verify_password(form_data.password, db_user.password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="密码错误")
    token = await create_token(db, db_user.id)
    await db.commit()
    return success_response(message='登录成功', data={
        'token': token
    })


@router.get('/info', summary='获取用户信息')
async def get_user_info(current_user=Depends(get_current_user)):
    return success_response(message='获取用户信息成功', data={
        'id': current_user.id,
        'username': current_user.username,
        'telNum': current_user.phone,
    })


@router.put('/updateInfo', summary='更新用户信息')
async def update_user_info(
    user_info: UserUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    new_user = await crud_update_user_info(db, user_info, current_user.id)
    if not new_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    await db.commit()
    return success_response(
        message='更新用户信息成功',
        data={
            'id': new_user.id,
            'username': new_user.username,
            'telNum': new_user.phone,
        }
    )


@router.put('/updatePassword', summary='更新用户密码')
async def update_password(
        pass_form: UserUpdatePassword,
        db: AsyncSession = Depends(get_db),
        current_user=Depends(get_current_user)
):
    is_old_password_right = await check_old_password(db, pass_form.oldPassword, current_user.id)
    if is_old_password_right is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    if not is_old_password_right:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="旧密码错误")
    updated = await crud_update_password(db, pass_form.password, current_user.id)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    await clear_user_token(db, current_user.id)
    await db.commit()
    return success_response(message='更新密码成功,请重新登录')


@router.post('/logout', summary='退出登录')
async def logout(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    is_logout = await clear_user_token(db, current_user.id)
    if not is_logout:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    await db.commit()
    return success_response(message='退出登录成功')


def _safe_unlink(file_path: str | None) -> None:
    if not file_path:
        return
    try:
        Path(file_path).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("清理文件失败, 请手动处理: %s, error=%s", file_path, exc)


@router.delete('/deleteUser', summary='注销用户')
async def delete_user(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        deleted = await crud_delete_user(db, current_user.id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
        await db.commit()
    except HTTPException:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"注销失败: {str(exc)}")

    image_deleted = deleted.get("deleted_images", {})
    for img_path in image_deleted.get("img_paths", []):
        _safe_unlink(img_path)
    for boundary_path in image_deleted.get("boundary_paths", []):
        _safe_unlink(boundary_path)

    return success_response(
        message='注销成功',
        data={
            'id': current_user.id,
            'deleted_images': image_deleted.get('deleted_count', 0),
        },
    )
