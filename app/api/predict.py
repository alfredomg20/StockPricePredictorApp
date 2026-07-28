from cachetools import TTLCache
from fastapi import APIRouter

from app.config import logger
from app.ml.predictor import StockPricePredictor
from app.schemas.predict import PredictionRequest, PredictionResponse, PredictionResult

router = APIRouter(prefix="/predict", tags=["prediction"])

# Configure caching
TIME_TO_LIVE = 3600  # 1 hour lifetime
cached_predictions = TTLCache(maxsize=100, ttl=TIME_TO_LIVE)


def invalidate_prediction_cache(ticker: str, forecast_days: int) -> None:
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
    - **forecast_days**: Number of days to predict into the future (1-30)

    Returns predictions with dates and prices.
    """
    cache_key = f"{request.ticker}:{request.forecast_days}"

    if cache_key not in cached_predictions:
        logger.info(f"Cache miss: Generating new prediction for {request.ticker} ({request.forecast_days} days)")
        predictor = StockPricePredictor()
        predictions_df = predictor.predict_prices(request.ticker, request.forecast_days)
        cached_predictions[cache_key] = predictions_df
    else:
        logger.info(f"Cache hit: Returning cached prediction for {request.ticker} ({request.forecast_days} days)")
        predictions_df = cached_predictions[cache_key]

    # Convert DataFrame to list of Pydantic models for response compatibility
    prediction_results = [
        PredictionResult(predicted_date=row["date"], predicted_price=row["predicted_price"])
        for row in predictions_df.to_dicts()
    ]

    return PredictionResponse(
        success=True,
        ticker=request.ticker,
        predictions=prediction_results,
    )