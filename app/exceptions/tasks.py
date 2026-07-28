from app.exceptions.base import AppException


class TaskNotFoundError(AppException):
    """Raised when requesting the status/result of a non-existent background task."""
    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(
            message=f"Training task with ID '{task_id}' was not found.",
            code="TASK_NOT_FOUND"
        )


class TaskNotReadyError(AppException):
    """Raised when attempting to fetch results for a task that is still PENDING or IN_PROGRESS."""
    def __init__(self, task_id: str, status: str):
        self.task_id = task_id
        self.status = status
        super().__init__(
            message=f"Training task '{task_id}' is not finished yet (Current status: '{status}').",
            code="TASK_NOT_READY"
        )


class ModelTrainingError(AppException):
    """Raised when the scikit-learn training pipeline crashes during execution."""
    def __init__(self, ticker: str, details: str):
        self.ticker = ticker
        self.details = details
        super().__init__(
            message=f"Training pipeline failed for ticker '{ticker}': {details}",
            code="TRAINING_FAILED"
        )