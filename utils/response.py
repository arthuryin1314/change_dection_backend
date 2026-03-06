from typing import Any

from fastapi.responses import JSONResponse


def success_response(message: str = "success", data: Any = None) -> JSONResponse:
    """Return a unified success payload for API responses."""
    return JSONResponse(
        status_code=200,
        content={
            "code": 200,
            "message": message,
            "data": data,
        },
    )

