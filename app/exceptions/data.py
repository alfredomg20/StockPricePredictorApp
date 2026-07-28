from app.exceptions.base import AppException


class InsufficientDataError(AppException):
    """Raised when the data source returns empty or insufficient rows for training/prediction."""
    def __init__(self, ticker: str, required_samples: int = 30, actual_samples: int = 0):
        self.ticker = ticker
        self.required_samples = required_samples
        self.actual_samples = actual_samples
        super().__init__(
            message=f"Insufficient stock data for ticker '{ticker}'. Required at least {required_samples} records, found {actual_samples}.",
            code="INSUFFICIENT_DATA"
        )


class DataFetchError(AppException):
    """Raised when external data loading fails due to connection/auth errors."""
    def __init__(self, ticker: str, original_error: str):
        self.ticker = ticker
        self.original_error = original_error
        super().__init__(
            message=f"Failed to fetch stock data for '{ticker}': {original_error}",
            code="DATA_FETCH_FAILED"
        )