import sys
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi_throttle import RateLimiter

from app.api.models import router as models_router
from app.api.predict import router as predict_router
from app.api.train import router as train_router
from app.config import REQUIRED_ENV_VARS
from app.utils.validation_utils import validate_env_variables

logger = logging.getLogger("app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app's lifecycle (Startup and Shutdown)."""
    # Startup
    missing_vars = validate_env_variables(REQUIRED_ENV_VARS)

    if missing_vars:
        logger.critical(f"Critical Fail at startup: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    logger.info("All required environment variables are set. Starting the application.")

    yield # Application execution happens here

    # Shutdown
    logger.info("Shutting down the application...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Implement throttler to limit requests to 100 per minute
    router_limiter = RateLimiter(times=100, seconds=60)

    app = FastAPI(
        title="Stock Prediction API",
        description="API for training and predicting stock prices using linear regression.",
        version="0.1.0",
        dependencies=[Depends(router_limiter)],
    )

    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=600,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add routers
    api_prefix = "/api/v1"
    app.include_router(models_router, prefix=api_prefix)
    app.include_router(predict_router, prefix=api_prefix)
    app.include_router(train_router, prefix=api_prefix)

    # Serve frontend files in root path
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

    return app

# Create the FastAPI application
app = create_app()

# Run the application with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)