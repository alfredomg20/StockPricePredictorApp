from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field 
from enum import Enum
from app.schemas.models import ModelMetrics

class TrainingStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TrainingRequest(BaseModel):
    """Request to initiate model training"""
    ticker: str = Field(..., example="MSFT", min_length=1, max_length=10, description="Stock ticker symbol (e.g., AAPL)")
    forecast_days: int = Field(default=5, ge=1, le=120, example=5, description="Number of days to forecast stock prices")

class TrainingTaskResponse(BaseModel):
    """Immediate response for training request"""
    task_id: str = Field(..., description="Unique ID for tracking training progress")
    status: TrainingStatus = Field(TrainingStatus.PENDING, description="Initial task status")
    submitted_at: datetime = Field(default_factory=datetime.now)
    ticker: str = Field(..., description="Stock ticker symbol")
    forecast_days: int = Field(..., description="Number of days forecasted")
    message: str = Field("Training task has been submitted and is being processed.",
                        description="Status message for the training task")
    metrics: Optional[ModelMetrics] = Field(None, description="Model results and evaluation metrics, if available")

class TrainingResult(BaseModel):
    """Final training results (available via separate endpoint)"""
    task_id: str = Field(..., description="Unique ID for the training task")
    duration_seconds: float = Field(..., description="Total duration of the training in seconds")
    status: str = Field(..., description="status of the training task")
    ticker: str = Field(..., description="Stock ticker symbol")
    forecast_days: int = Field(..., description="Number of days the model was trained to forecast")
    model_path: str = Field(..., description="Path to the saved model file")
    metrics: Optional[ModelMetrics] = Field(None, description="Model results and evaluation metrics if available")
    error: Optional[str] = Field(None, description="Error message if training failed")
    message: Optional[str] = Field(None, description="Additional message about the training result if applicable")