from datetime import date, datetime
from typing import List
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """Request for stock price predictions"""
    ticker: str = Field(..., example="MSFT", min_length=1, max_length=10, description="Stock ticker symbol (e.g., AAPL)")
    forecast_days: int = Field(default=5, ge=1, le=120, example=5, description="Number of days to forecast stock prices")
    
class PredictionResult(BaseModel):
    """Single prediction result"""
    predicted_date: date = Field(..., description="Date of the prediction")
    predicted_price: float = Field(..., description="Predicted stock price for the date")
    
class PredictionResponse(BaseModel):
    """Prediction API response"""
    success: bool = Field(..., description="Indicates if the prediction was successful")
    ticker: str = Field(..., description="Stock ticker symbol")
    predictions: List[PredictionResult] = Field(..., description="List of predicted stock prices")
    generated_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the predictions were generated")