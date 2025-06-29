from datetime import date
from typing import Annotated
from fastapi import Path
from pydantic import BaseModel, Field

TickerPath = Annotated[str, Path(description="Stock ticker symbol (e.g., AAPL)")]
ForecastDaysPath = Annotated[int, Path(description="Number of forecast days")]
LastTrainedTimePath = Annotated[str, Path(description="Last trained time (New York timezone) of the model (YYYY-MM-DD-HH:MM:SS)")]

class ModelMetrics(BaseModel):
    """Metrics for model evaluation"""
    mae: float = Field(..., description="Mean Absolute Error of the model")
    r2: float = Field(..., description="R-squared value of the model")
    mape: float = Field(..., description="Mean Absolute Percentage Error of the model")
    max_ae: float = Field(..., description="Maximum Absolute Error of the model")
    features: list[str] = Field(..., description="List of features used in the model")
    coefficients: dict[str, float] = Field(..., description="Coefficients of the model features")
    intercept: float = Field(..., description="Intercept of the model")
    train_samples: int = Field(..., description="Number of samples used for training")
    test_samples: int = Field(..., description="Number of samples used for testing")
    last_train_date: date = Field(..., description="Date of the last training (YYYY-MM-DD)")

class ModelInfo(BaseModel):
    """Information about a trained model"""
    ticker: str = Field(..., description="Stock ticker symbol")
    model_type: str = Field(..., description="Type of the model (e.g., linear, random_forest)")
    version: str = Field(..., description="Version of the model")
    metrics: ModelMetrics = Field(..., description="Evaluation metrics of the model")

class ModelsListResponse(BaseModel):
    """Response model for listing all models"""
    models: list[ModelInfo] = Field(..., description="List of trained models")
    count: int = Field(..., description="Total number of models available")

class DeleteModelResponse(BaseModel):
    """Response model for model deletion"""
    status: str = Field(..., description="Status of the deletion operation")
    version: str = Field(..., description="Version of the deleted model")
