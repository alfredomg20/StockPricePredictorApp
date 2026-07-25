from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Depends, status
from app.ml.model_store import ModelStore
from app.schemas.models import (
    DeleteModelResponse,
    ForecastDaysPath,
    ModelInfo,
    ModelsListResponse,
    TickerPath,
    LastTrainedTimePath
)
from app.config import logger, MODELS_DIR

router = APIRouter(prefix="/models", tags=["models"])

# Configure caching
time_to_live = 3600 # 1 hour lifetime
cached_models = TTLCache(maxsize=1000, ttl=time_to_live)

def invalidate_models_cache():
    cached_models.clear()

# Dependency to get model store instance
def get_model_store():
    return ModelStore(MODELS_DIR)

@router.get("/", response_model=ModelsListResponse)
async def get_all_models(
    model_store: ModelStore = Depends(get_model_store)
) -> dict[str, ModelsListResponse]:
    """
    Get a list of all available trained models with their metadata.
    
    Returns a list of models with their metrics and metadata.
    """
    try:
        logger.info("Fetching all models...")
        if "all_models" in cached_models:
            logger.info(f"Returning {len(cached_models)} cached models")
            return cached_models["all_models"]
        all_models = model_store.get_all_models()
        logger.info(f"Retrieved {len(all_models)} models")
        cached_models["all_models"] = all_models
        return all_models
    except Exception as e:
        logger.error(f"Model listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve models"
        )

@router.get("/{ticker}/{forecast_days}/{last_trained_time}", response_model=ModelInfo)
async def get_model(
    ticker: TickerPath,
    forecast_days: ForecastDaysPath,
    last_trained_time: LastTrainedTimePath,
    model_store: ModelStore = Depends(get_model_store)
):
    """
    Get a specific model by its parameters.
    
    - **ticker**: Stock ticker symbol (e.g., AAPL)
    - **forecast_days**: Number of days the model was trained to forecast
    - **last_trained_time**: Last trained time of the model (YYYY-MM-DD-HH:MM:SS)
    
    Returns the model information including metrics and metadata.
    """
    try:
        logger.info(f"Retrieving model for {ticker} with {forecast_days} days forecast, last trained at {last_trained_time}")
        key = f"{ticker}:{forecast_days}:{last_trained_time}"
        if key in cached_models:
            logger.info(f"Using cached model for {ticker}")
            return cached_models[key]
        model_data = model_store.get_model(ticker, forecast_days, last_trained_time)
        logger.info(f"Model retrieved successfully for {ticker}")
        cached_models[key] = model_data
        return model_data
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Model retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model"
        )

@router.delete("/{ticker}/{forecast_days}/{last_trained_time}", response_model=DeleteModelResponse)
async def delete_model(
    ticker: TickerPath,
    forecast_days: ForecastDaysPath,
    last_trained_time: LastTrainedTimePath,
    model_store: ModelStore = Depends(get_model_store)
) -> dict[str, DeleteModelResponse]:
    """
    Delete a specific model by its parameters.
    
    - **ticker**: Stock ticker symbol (e.g., AAPL)
    - **forecast_days**: Number of days the model was trained to forecast
    - **last_trained_time**: Last trained time of the model (YYYY-MM-DD-HH:MM:SS)
    
    Returns status of the deletion operation.
    """
    try:
        invalidate_models_cache()
        return model_store.delete_model(ticker, forecast_days, last_trained_time)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Model deletion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete model"
        )
