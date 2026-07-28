import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.exceptions import (
    AppException,
    DataFetchError,
    InsufficientDataError,
    ModelCorruptedError,
    ModelNotFoundError,
    ModelTrainingError,
    TaskNotFoundError,
    TaskNotReadyError,
)
from app.schemas.error import ErrorDetail

logger = logging.getLogger("app")


def create_error_response(
    status_code: int,
    error_code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Helper function to build a consistent JSONResponse format."""
    payload = ErrorDetail(
        success=False,
        error_code=error_code,
        message=message,
        timestamp=datetime.now(timezone.utc).isoformat(),
        details=details,
    )
    return JSONResponse(status_code=status_code, content = payload.dict(exclude_none=True))


# Domain Exception Handlers

async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Fallback handler for generic domain AppException."""
    logger.error(f"Application error on {request.url.path}: {exc.message}")
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code=exc.code,
        message=exc.message,
    )


async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    logger.warning(f"Model not found on {request.url.path}: {exc.message}")
    return create_error_response(
        status_code=status.HTTP_404_NOT_FOUND,
        error_code=exc.code,
        message=exc.message,
        details={"ticker": exc.ticker, "forecast_days": exc.forecast_days},
    )


async def model_corrupted_handler(request: Request, exc: ModelCorruptedError) -> JSONResponse:
    logger.error(f"Model corrupted on {request.url.path}: {exc.message}")
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code=exc.code,
        message=exc.message,
        details={"ticker": exc.ticker, "forecast_days": exc.forecast_days},
    )


async def insufficient_data_handler(request: Request, exc: InsufficientDataError) -> JSONResponse:
    logger.warning(f"Insufficient data on {request.url.path}: {exc.message}")
    return create_error_response(
        status_code=status.HTTP_400_BAD_REQUEST,
        error_code=exc.code,
        message=exc.message,
        details={
            "ticker": exc.ticker,
            "required_samples": exc.required_samples,
            "actual_samples": exc.actual_samples,
        },
    )


async def data_fetch_handler(request: Request, exc: DataFetchError) -> JSONResponse:
    logger.error(f"Data fetch error on {request.url.path}: {exc.message}")
    return create_error_response(
        status_code=status.HTTP_502_BAD_GATEWAY,
        error_code=exc.code,
        message=exc.message,
        details={"ticker": exc.ticker},
    )


async def task_not_found_handler(request: Request, exc: TaskNotFoundError) -> JSONResponse:
    logger.warning(f"Task not found on {request.url.path}: {exc.message}")
    return create_error_response(
        status_code=status.HTTP_404_NOT_FOUND,
        error_code=exc.code,
        message=exc.message,
        details={"task_id": exc.task_id},
    )


async def task_not_ready_handler(request: Request, exc: TaskNotReadyError) -> JSONResponse:
    logger.warning(f"Task not ready on {request.url.path}: {exc.message}")
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code=exc.code,
        message=exc.message,
        details={"task_id": exc.task_id, "current_status": exc.status},
    )


async def model_training_handler(request: Request, exc: ModelTrainingError) -> JSONResponse:
    logger.error(f"Training failed on {request.url.path}: {exc.message}")
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code=exc.code,
        message=exc.message,
        details={"ticker": exc.ticker},
    )


# FastAPI Validation & Global Exception Handlers

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Overriding default FastAPI Pydantic validation error to match custom structure."""
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    return create_error_response(
        status_code=status.HTTP_400_BAD_REQUEST,
        error_code="VALIDATION_ERROR",
        message="Request payload validation failed.",
        details={"errors": exc.errors()},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected system crashes (HTTP 500)."""
    logger.critical(f"Unhandled exception on {request.url.path}: {exc!s}")
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected server error occurred.",
        details={"error_class": exc.__class__.__name__},
    )


# Register all exception handlers to the FastAPI app instance

def register_exception_handlers(app: FastAPI) -> None:
    """Registers all custom exception handlers to the FastAPI app instance."""
    app.add_exception_handler(ModelNotFoundError, model_not_found_handler)
    app.add_exception_handler(ModelCorruptedError, model_corrupted_handler)
    app.add_exception_handler(InsufficientDataError, insufficient_data_handler)
    app.add_exception_handler(DataFetchError, data_fetch_handler)
    app.add_exception_handler(TaskNotFoundError, task_not_found_handler)
    app.add_exception_handler(TaskNotReadyError, task_not_ready_handler)
    app.add_exception_handler(ModelTrainingError, model_training_handler)
    
    # Base domain exception fallback
    app.add_exception_handler(AppException, app_exception_handler)
    
    # Framework & Global fallbacks
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)