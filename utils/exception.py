import os
import traceback
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette import status

from utils.response import error_response

# 开发模式：返回详细错误信息
# 生产模式：返回简化错误信息
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"


def _normalize_validation_msg(raw_msg: str) -> str:
    msg = (raw_msg or "").strip()
    for prefix in ("Value error,", "value error,", "Assertion error,"):
        if msg.startswith(prefix):
            msg = msg[len(prefix):].strip()
            break
    return msg or "请求参数校验失败"


def _stringify_detail(detail) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    if isinstance(detail, list):
        return str(detail[0]) if detail else "请求参数校验失败"
    return str(detail)


def _format_validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors() or []
    if not errors:
        return "请求参数校验失败"

    first = errors[0]
    ctx = first.get("ctx") or {}
    if isinstance(ctx, dict) and ctx.get("error_message"):
        return str(ctx["error_message"])

    loc = list(first.get("loc") or [])
    raw_msg = first.get("msg") or "请求参数校验失败"
    msg = _normalize_validation_msg(raw_msg)

    # `("body",)` usually means model-level validation; return message directly.
    visible_loc = [str(part) for part in loc if str(part) != "body"]
    if not visible_loc:
        return msg

    field = visible_loc[-1]
    return f"{field}: {msg}"


async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    message = _format_validation_message(exc)
    if DEBUG_MODE:
        print(f"[ValidationError] path={request.url} message={message} errors={exc.errors()}")
    return error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, message)


async def http_exception_handler(request: Request, exc: HTTPException):
    return error_response(exc.status_code, _stringify_detail(exc.detail))


async def integrity_error_handler(request: Request, exc: IntegrityError):
    error_msg = str(exc.orig)
    if "username_UNIQUE" in error_msg or "Duplicate entry" in error_msg:
        detail = "用户名已存在"
    elif "FOREIGN KEY" in error_msg:
        detail = "关联数据不存在"
    else:
        detail = "数据约束冲突，请检查输入"

    if DEBUG_MODE:
        print(f"[IntegrityError] path={request.url} detail={error_msg}")
    return error_response(status.HTTP_400_BAD_REQUEST, detail)


async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError):
    if DEBUG_MODE:
        print(
            f"[SQLAlchemyError] path={request.url} type={type(exc).__name__} detail={str(exc)}\n"
            f"{traceback.format_exc()}"
        )
    return error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "数据库操作失败，请稍后重试")


async def general_exception_handler(request: Request, exc: Exception):
    if DEBUG_MODE:
        print(
            f"[UnhandledException] path={request.url} type={type(exc).__name__} detail={str(exc)}\n"
            f"{traceback.format_exc()}"
        )
    return error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "服务器内部错误")
