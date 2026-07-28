from app.exceptions.base import AppException


class ModelNotFoundError(AppException):
    """Raised when a requested trained model file is not found in storage."""
    def __init__(self, ticker: str, forecast_days: int):
        self.ticker = ticker
        self.forecast_days = forecast_days
        super().__init__(
            message=f"No trained model found for ticker '{ticker}' with {forecast_days}-day forecast.",
            code="MODEL_NOT_FOUND"
        )


class ModelCorruptedError(AppException):
    """Raised when a model file exists but cannot be loaded or deserialized."""
    def __init__(self, ticker: str, forecast_days: int):
        self.ticker = ticker
        self.forecast_days = forecast_days
        super().__init__(
            message=f"Trained model file for '{ticker}' ({forecast_days} days) is corrupted or incompatible.",
            code="MODEL_CORRUPTED"
        )