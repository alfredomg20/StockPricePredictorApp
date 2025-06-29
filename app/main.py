from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi_throttle import RateLimiter
from app.api.models import router as models_router
from app.api.predict import router as predict_router
from app.api.train import router as train_router

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
    app.include_router(models_router)
    app.include_router(predict_router)
    app.include_router(train_router)

    return app

# Create the FastAPI application
app = create_app()

# Run the application with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)