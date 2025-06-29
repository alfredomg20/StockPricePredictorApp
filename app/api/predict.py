from cachetools import TTLCache
from fastapi import APIRouter, HTTPException
from app.config import logger
from app.ml.predictor import StockPricePredictor
from app.schemas.predict import PredictionResult, PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict", tags=["prediction"])
    
# Configure caching
time_to_live = 3600  # 1 hour lifetime
cached_predictions = TTLCache(maxsize=100, ttl=time_to_live)

def invalidate_prediction_cache(ticker: str, forecast_days: int):
    """
    Deletes all cached predictions for a specific ticker and forecast days.
    
    Args:
        ticker (str): Stock ticker symbol.
        forecast_days (int): Number of days to forecast.
    """
    cache_key = f"{ticker}:{forecast_days}"
    if cache_key in cached_predictions:
        del cached_predictions[cache_key]
    
@router.post("/", response_model=PredictionResponse)
async def predict_stock_price(request: PredictionRequest):
    """
    Generate stock price predictions for a specified number of future days.
    
    - **ticker**: Stock ticker symbol (e.g., AAPL)
    - **days**: Number of days to predict into the future (1-30)
    
    Returns predictions with dates and prices.
    """
    try:
        logger.info(f"Prediction request received for {request.ticker}, {request.forecast_days} days forecast")
        cache_key = f"{request.ticker}:{request.forecast_days}"

        if cache_key not in cached_predictions:
            # Initialize predictor
            logger.info(f"Predicting prices for {request.ticker} and {request.forecast_days} days ahead...")
            predictor = StockPricePredictor()
            # Generate predictions
            predictions_df = predictor.predict_prices(request.ticker, request.forecast_days)
            cached_predictions[cache_key] = predictions_df
        else:
            # Use cached predictions
            predictions_df = cached_predictions[cache_key]

        # Convert DataFrame to list of Pydantic models for response compatibility
        prediction_results = [
            PredictionResult(predicted_date=row["date"], predicted_price=row["predicted_price"]) 
            for row in predictions_df.to_dicts()
        ]

        logger.info(f"Prediction successful for {request.ticker}, {len(prediction_results)} days forecasted")
        return PredictionResponse(
            success=True,
            ticker=request.ticker,
            predictions=prediction_results,
        )      
            
    except FileNotFoundError as e:
        logger.error(f"No model found for {request.ticker} with {request.forecast_days} day forecast")
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for ticker {request.ticker}" if "not found" in str(e) else str(e)
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate prediction: {str(e)}"
        )
