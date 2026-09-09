from typing import Any

from fastapi.responses import JSONResponse


def api_response(status_code: int, message: str, data: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"code": status_code, "message": message, "data": data},
    )


def success_response(message: str = "success", data: Any = None) -> JSONResponse:
    """Return a unified success payload for API responses."""
    return api_response(200, message, data)


def error_response(code: int, message: str) -> JSONResponse:
    """Return a unified error payload with only code and message."""
    return api_response(code, message, None)
