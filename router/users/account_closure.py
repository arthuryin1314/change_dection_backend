from fastapi import HTTPException
from starlette import status

from crud.users import delete_user as delete_user_record
from crud.users import get_user_by_id
from router.image import image_lifecycle


async def close_account(db, user_id: int) -> dict:
    try:
        if not await get_user_by_id(db, user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在",
            )

        prepared = await image_lifecycle.prepare_user_image_deletion(db, user_id)
    except HTTPException:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注销失败，请稍后重试",
        ) from exc

    try:
        deleted = await delete_user_record(db, user_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在",
            )
        await db.commit()
    except HTTPException:
        await db.rollback()
        image_lifecycle.cancel_cleanup(prepared.operation_id)
        raise
    except Exception as exc:
        await db.rollback()
        image_lifecycle.cancel_cleanup(prepared.operation_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注销失败，请稍后重试",
        ) from exc

    try:
        await image_lifecycle.finish_cleanup(prepared.operation_id)
    except Exception:
        pass

    return {
        "id": user_id,
        "deleted_images": prepared.deleted_count,
    }
