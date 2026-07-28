from typing import Any

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Structured Error Schema returned by all API endpoints on failure."""
    success: bool = False
    error_code: str
    message: str
    timestamp: str
    details: dict[str, Any] | None = None
