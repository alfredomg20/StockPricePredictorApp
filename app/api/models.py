import logging

from cachetools import TTLCache
from fastapi import APIRouter, Depends

from app.api.deps import get_config
from app.exceptions import ModelNotFoundError
from app.ml.model_store import ModelStore
from app.schemas.config import FullConfigSchema
from app.schemas.models import (
    DeleteModelResponse,
    ForecastDaysPath,
    LastTrainedTimePath,
    ModelInfo,
    ModelsListResponse,
    TickerPath,
)

logger = logging.getLogger('app')

router = APIRouter(prefix="/models", tags=["models"])

# Configure caching
TIME_TO_LIVE = 3600  # 1 hour lifetime
cached_models = TTLCache(maxsize=1000, ttl=TIME_TO_LIVE)


def invalidate_models_cache() -> None:
    """Clears all cached model metadata."""
    cached_models.clear()


def get_model_store(
    config: FullConfigSchema = Depends(get_config)
) -> ModelStore:
    """Dependency to get model store instance."""
    return ModelStore(model_dir=config.paths.models_dir, tz=config.env.timezone)


@router.get("/", response_model=ModelsListResponse)
async def get_all_models(
    model_store: ModelStore = Depends(get_model_store)
):
    """
    Get a list of all available trained models with their metadata.

    Returns a list of models with their metrics and metadata.
    """
    if "all_models" in cached_models:
        logger.info("Cache hit: Returning cached models list")
        return cached_models["all_models"]

    logger.info("Cache miss: Fetching all models from store...")
    all_models = model_store.get_all_models()
    cached_models["all_models"] = all_models
    return all_models


@router.get("/{ticker}/{forecast_days}/{last_trained_time}", response_model=ModelInfo)
async def get_model(
    ticker: TickerPath,
    forecast_days: ForecastDaysPath,
    last_trained_time: LastTrainedTimePath,
    model_store: ModelStore = Depends(get_model_store),
):
    """
    Get a specific model by its parameters.

    - **ticker**: Stock ticker symbol (e.g., AAPL)
    - **forecast_days**: Number of days the model was trained to forecast
    - **last_trained_time**: Last trained time of the model (YYYY-MM-DD-HH:MM:SS)

    Returns the model information including metrics and metadata.
    """
    key = f"{ticker}:{forecast_days}:{last_trained_time}"
    if key in cached_models:
        logger.info(f"Cache hit: Using cached model metadata for {ticker}")
        return cached_models[key]

    logger.info(f"Cache miss: Retrieving model for {ticker} ({forecast_days} days) trained at {last_trained_time}")
    model_data = model_store.get_model(ticker, forecast_days, last_trained_time)
    
    if not model_data:
        raise ModelNotFoundError(ticker=ticker, forecast_days=forecast_days)

    cached_models[key] = model_data
    return model_data


@router.delete("/{ticker}/{forecast_days}/{last_trained_time}", response_model=DeleteModelResponse)
async def delete_model(
    ticker: TickerPath,
    forecast_days: ForecastDaysPath,
    last_trained_time: LastTrainedTimePath,
    model_store: ModelStore = Depends(get_model_store),
):
    """
    Delete a specific model by its parameters.

    - **ticker**: Stock ticker symbol (e.g., AAPL)
    - **forecast_days**: Number of days the model was trained to forecast
    - **last_trained_time**: Last trained time of the model (YYYY-MM-DD-HH:MM:SS)

    Returns status of the deletion operation.
    """
    invalidate_models_cache()
    logger.info(f"Deleting model for {ticker} ({forecast_days} days) trained at {last_trained_time}")
    return model_store.delete_model(ticker, forecast_days, last_trained_time)