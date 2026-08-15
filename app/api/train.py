import logging
import time
import uuid
from datetime import datetime, timedelta

import pytz
from fastapi import APIRouter, BackgroundTasks, Depends, status

from app.api.deps import get_config
from app.api.models import get_model_store
from app.exceptions import TaskNotFoundError, TaskNotReadyError
from app.ml.data_loader import get_latest_date, load_ticker_data
from app.ml.model_store import ModelStore
from app.ml.trainer import StockPriceTrainer
from app.schemas.config import FullConfigSchema
from app.schemas.models import ModelMetrics
from app.schemas.train import (
    TrainingRequest,
    TrainingResult,
    TrainingStatus,
    TrainingTaskResponse,
)

logger = logging.getLogger('app')

# Create the router
router = APIRouter(
    prefix="/train",
    tags=["training"],
)

# In-memory storage for training tasks
training_tasks = {}


def get_trainer(
    config: FullConfigSchema = Depends(get_config)
) -> StockPriceTrainer:
    """Dependency to get StockPriceTrainer instance"""
    return StockPriceTrainer(model_dir=config.paths.models_dir, tz=config.env.timezone)


async def train_stock_model_task(
    task_id: str, 
    ticker: str, 
    forecast_days: int,
    trainer: StockPriceTrainer,
    config: FullConfigSchema | None = None
):
    """Background task to train a stock price prediction model."""
    timezone = config.env.timezone if config else None
    try:
        # Update task status
        training_tasks[task_id]["status"] = TrainingStatus.IN_PROGRESS
        start_time = time.time()
        
        # Fetch last 3 years of data
        latest_date = get_latest_date(ticker, config.gcloud)
        years = 3
        days_in_year = 365
        
        # Handle datetime generation with fallback if timezone is not provided
        now_dt = datetime.now(timezone) if timezone else datetime.now(pytz.timezone("UTC"))
        start_date = (now_dt - timedelta(days=years * days_in_year)).strftime("%Y-%m-%d")
        df = load_ticker_data(ticker=ticker, start_date=start_date, end_date=latest_date, gcloud_config=config.gcloud)
        
        # Run the training pipeline
        results = trainer.training_pipeline(ticker=ticker, df=df, forecast_days=forecast_days)
        
        # Calculate training duration
        duration = time.time() - start_time
        
        # Append additional results
        results['task_id'] = task_id
        results['duration_seconds'] = duration

        # Update task status and save results
        training_tasks[task_id].update({
            "status": TrainingStatus.COMPLETED,
            "result": results
        })
        logger.info(f"Training completed for task {task_id}, ticker {ticker}")

    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as e:
        logger.exception(f"Training failed for task {task_id}")
        training_tasks[task_id].update({
            "status": TrainingStatus.FAILED,
            "result": {
                "error": str(e),
                "task_id": task_id,
                "ticker": ticker,
                "forecast_days": forecast_days
            }
        })


@router.post("/", response_model=TrainingTaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_training(
    request: TrainingRequest, 
    background_tasks: BackgroundTasks,
    trainer: StockPriceTrainer = Depends(get_trainer),
    model_store: ModelStore = Depends(get_model_store),
    config: FullConfigSchema = Depends(get_config)
):
    """
    Endpoint to start a new model training task.
    Returns a task ID that can be used to check the training status.
    """
    timezone = config.env.timezone
    if not trainer.needs_training(request.ticker, request.forecast_days):
        logger.info(
            f"Up-to-date model already exists for {request.ticker} "
            f"with {request.forecast_days} days forecast, skipping training."
        )
        # Get metrics from the latest model
        latest_model_path = trainer._get_latest_model_path(request.ticker, request.forecast_days)
        last_trained_time_str = str(latest_model_path).split('_')[-1].replace('.joblib', '')
        parsed_dt = datetime.strptime(last_trained_time_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone)
        last_trained_time = parsed_dt.strftime("%Y-%m-%d-%H:%M:%S")
        latest_model = model_store.get_model(request.ticker, request.forecast_days, last_trained_time)

        return TrainingTaskResponse(
            task_id="",
            status=TrainingStatus.SKIPPED,
            submitted_at=datetime.now(timezone),
            ticker=request.ticker,
            forecast_days=request.forecast_days,
            metrics=ModelMetrics(
                mae=latest_model["metrics"]["mae"],
                r2=latest_model["metrics"]["r2"],
                mape=latest_model["metrics"]["mape"],
                max_ae=latest_model["metrics"]["max_ae"],
                features=latest_model["metrics"]["features"],
                coefficients=latest_model["metrics"]["coefficients"],
                intercept=latest_model["metrics"]["intercept"],
                train_samples=latest_model["metrics"]["train_samples"],
                test_samples=latest_model["metrics"]["test_samples"],
                last_train_date=latest_model["metrics"]["last_train_date"]
            ),
            message="Up-to-date model already exists, skipping training."
        )
    
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Create task response
    task_response = TrainingTaskResponse(
        task_id=task_id,
        status=TrainingStatus.PENDING,
        submitted_at=datetime.now(timezone),
        ticker=request.ticker,
        forecast_days=request.forecast_days,
        message="Training started in background"
    )
    
    # Store task info
    training_tasks[task_id] = {
        "ticker": request.ticker,
        "forecast_days": request.forecast_days,
        "status": TrainingStatus.PENDING,
        "created_at": datetime.now(timezone),
        "result": None
    }
    
    # Start background task passing resolved dependencies
    background_tasks.add_task(
        train_stock_model_task,
        task_id=task_id, 
        ticker=request.ticker, 
        forecast_days=request.forecast_days,
        trainer=trainer,
        config=config
    )
    
    logger.info(f"Training task {task_id} started for ticker {request.ticker}")
    return task_response


@router.get("/{task_id}/status", response_model=TrainingTaskResponse)
async def get_training_status(task_id: str):
    """Get the current status of a training task."""
    if task_id not in training_tasks:
        raise TaskNotFoundError(task_id=task_id)
    
    task_info = training_tasks[task_id]

    return TrainingTaskResponse(
        task_id=task_id,
        status=task_info["status"],
        submitted_at=task_info["created_at"],
        ticker=task_info["ticker"],
        forecast_days=task_info["forecast_days"],
        message=f"Training status: {task_info['status']}"
    )


@router.get("/{task_id}/result", response_model=TrainingResult)
async def get_training_result(task_id: str):
    """Get the results of a completed training task."""
    if task_id not in training_tasks:
        raise TaskNotFoundError(task_id=task_id)
    
    task_info = training_tasks[task_id]
    current_status = task_info["status"]
    
    if current_status not in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
        raise TaskNotReadyError(task_id=task_id, status=current_status)
    
    return task_info["result"]