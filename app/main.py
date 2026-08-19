import logging
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi_throttle import RateLimiter

from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.middleware import RequestIDMiddleware
from app.api.models import router as models_router
from app.api.predict import router as predict_router
from app.api.train import router as train_router
from app.config.logging import setup_logging
from app.config.settings import CONFIG
from app.schemas.config import FullConfigSchema
from app.utils.validation_utils import validate_env_variables

logger = logging.getLogger('app')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app's lifecycle (Startup and Shutdown)."""

    # Startup
    config: FullConfigSchema = app.state.config
    missing_vars = validate_env_variables(config.env.required_env_vars)
    if missing_vars:
        logger.critical(f"Critical Fail at startup: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    logger.info("All required environment variables are set. Starting the application.")

    yield # Application execution happens here

    # Shutdown
    logger.info("Shutting down the application...")


def create_app(config: FullConfigSchema) -> FastAPI:
    """Create and configure the FastAPI application."""

    # Init custom logger
    setup_logging(level=config.api.logger_level, environment=config.env.environment)

    # Implement throttler to limit requests to 100 per minute
    router_limiter = RateLimiter(times=100, seconds=60)

    app = FastAPI(
        title=config.api.title,
        description=config.api.description,
        version=config.api.version,
        lifespan=lifespan
    )

    app.state.config = config

    register_exception_handlers(app)

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=600,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    api_prefix = "/api/v1"
    app.include_router(models_router, prefix=api_prefix, dependencies=[Depends(router_limiter)])
    app.include_router(predict_router, prefix=api_prefix, dependencies=[Depends(router_limiter)])
    app.include_router(train_router, prefix=api_prefix, dependencies=[Depends(router_limiter)])
    app.include_router(health_router, prefix=api_prefix)

    # Serve frontend files in root path
    app.mount("/", StaticFiles(directory=config.paths.frontend_dir, html=True), name="frontend")

    return app

# Create the FastAPI application
app = create_app(config=CONFIG)

# Run the application with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)