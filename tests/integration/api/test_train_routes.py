import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import uuid
from datetime import date, datetime, timedelta
import polars as pl
import numpy as np
from app.api.train import train_stock_model_task, training_tasks
from app.main import app
from app.schemas.train import TrainingStatus

client = TestClient(app)

# Mock data
MOCK_TASK_ID = str(uuid.uuid4())
MOCK_TRAINING_RESULT = {
    "task_id": MOCK_TASK_ID,
    "duration_seconds": 5.25,
    "status": "success",
    "ticker": "AAPL",
    "forecast_days": 7,
    "model_path": "models/AAPL_linear_7day_20230601.joblib",
    "metrics": {
        "mae": 1.5,
        "r2": 0.85,
        "mape": 0.9,
        "max_ae": 2.0,
        "features": ["lag1", "lag5", "ma5", "ma20"],
        "coefficients": {
            "lag1": 0.2,
            "lag5": 0.1,
            "ma5": 0.3,
            "ma20": 0.4
        },
        "intercept": 0.5,
        "train_samples": 500,
        "test_samples": 100,
        "last_train_date": date(2023, 6, 1)
    },
}

# Create mock stock data
@pytest.fixture
def mock_stock_data():
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(500)]
    prices = [100 + i * 0.1 + np.random.normal(0, 1) for i in range(500)]
    
    return pl.DataFrame({
        "date": dates,
        "price": prices,
        "lag1": prices[1:] + [0],
        "lag5": prices[5:] + [0] * 5,
        "lag20": prices[20:] + [0] * 20,
        "lag100": prices[100:] + [0] * 100,
        "ma5": [sum(prices[max(0, i-4):i+1])/min(i+1, 5) for i in range(500)],
        "ma20": [sum(prices[max(0, i-19):i+1])/min(i+1, 20) for i in range(500)],
        "ma100": [sum(prices[max(0, i-99):i+1])/min(i+1, 100) for i in range(500)]
    })

class TestTrainingRoutes:
    @patch("app.api.train.train_stock_model_task")
    def test_start_training(self, mock_train_task):
        """Test starting a new training task"""
        # Setup mock
        mock_train_task.return_value = None
        
        # Make request
        response = client.post(
            "/api/v1/train/",
            json={"ticker": "AAPL", "forecast_days": 7}
        )
        
        # Verify response
        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data
        assert data["status"] == TrainingStatus.PENDING
        assert data["ticker"] == "AAPL"
        assert data["forecast_days"] == 7
        assert "submitted_at" in data
        assert "message" in data
    
    @patch("app.api.train.training_tasks")
    def test_get_training_status_found(self, mock_tasks):
        """Test getting status of an existing training task"""
        # Setup mock
        mock_tasks.get.return_value = True
        mock_tasks.__contains__.return_value = True
        mock_tasks.__getitem__.return_value = {
            "status": TrainingStatus.IN_PROGRESS,
            "created_at": datetime.now(),
            "ticker": "AAPL",
            "forecast_days": 7
        }
        
        # Make request
        response = client.get(f"/api/v1/train/{MOCK_TASK_ID}/status")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == MOCK_TASK_ID
        assert data["status"] == TrainingStatus.IN_PROGRESS
        assert data["ticker"] == "AAPL"
        assert data["forecast_days"] == 7
        assert "submitted_at" in data
        assert "message" in data
    
    def test_get_training_status_not_found(self):
        """Test getting status of a non-existent training task"""
        # Make request with a random UUID that shouldn't exist
        response = client.get(f"/api/v1/train/{uuid.uuid4()}/status")
        
        # Verify response
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"]
    
    @patch("app.api.train.training_tasks")
    def test_get_training_result_completed(self, mock_tasks):
        """Test getting results of a completed training task"""
        # Setup mock
        mock_tasks.__contains__.return_value = True
        mock_tasks.__getitem__.return_value = {
            "status": TrainingStatus.COMPLETED,
            "created_at": datetime.now(),
            "ticker": "AAPL",
            "forecast_days": 7,
            "result": MOCK_TRAINING_RESULT
        }
        
        # Make request
        response = client.get(f"/api/v1/train/{MOCK_TASK_ID}/result")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["ticker"] == "AAPL"
        assert data["forecast_days"] == 7
        assert data["task_id"] == MOCK_TASK_ID
        assert "model_path" in data
        assert "metrics" in data
        assert "duration_seconds" in data
    
    @patch("app.api.train.training_tasks")
    def test_get_training_result_failed(self, mock_tasks):
        """Test getting results of a failed training task"""
        # Setup mock with a failed task
        failed_result = MOCK_TRAINING_RESULT.copy()
        failed_result["status"] = "error"
        failed_result["error"] = "Training failed due to insufficient data"
        
        mock_tasks.__contains__.return_value = True
        mock_tasks.__getitem__.return_value = {
            "status": TrainingStatus.FAILED,
            "created_at": datetime.now(),
            "ticker": "AAPL",
            "forecast_days": 7,
            "result": failed_result
        }
        
        # Make request
        response = client.get(f"/api/v1/train/{MOCK_TASK_ID}/result")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "error" in data
    
    def test_get_training_result_not_found(self):
        """Test getting results of a non-existent training task"""
        # Make request with a random UUID that shouldn't exist
        response = client.get(f"/api/v1/train/{uuid.uuid4()}/result")
        
        # Verify response
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"]
    
    @patch("app.api.train.training_tasks")
    def test_get_training_result_in_progress(self, mock_tasks):
        """Test getting results of a training task that is still in progress"""
        # Setup mock with an in-progress task
        mock_tasks.__contains__.return_value = True
        mock_tasks.__getitem__.return_value = {
            "status": TrainingStatus.IN_PROGRESS,
            "created_at": datetime.now(),
            "ticker": "AAPL",
            "forecast_days": 7,
            "result": None
        }
        
        # Make request
        response = client.get(f"/api/v1/train/{MOCK_TASK_ID}/result")
        
        # Verify response
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "still in progress" in data["detail"]
    
    @patch("app.api.train.training_tasks")
    def test_get_training_result_no_result(self, mock_tasks):
        """Test getting results when they are missing despite completed status"""
        # Setup mock with a completed task but missing results
        mock_tasks.__contains__.return_value = True
        mock_tasks.__getitem__.return_value = {
            "status": TrainingStatus.COMPLETED,
            "created_at": datetime.now(),
            "ticker": "AAPL",
            "forecast_days": 7,
            "result": None
        }
        
        # Make request
        response = client.get(f"/api/v1/train/{MOCK_TASK_ID}/result")
        
        # Verify response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "not available" in data["detail"]
    
    @patch("app.api.train.get_latest_date")
    @patch("app.api.train.load_ticker_data")
    @patch("app.ml.trainer.StockPriceTrainer.training_pipeline")
    def test_train_stock_model_task(self, mock_train_pipeline, mock_load_data, 
                                  mock_get_latest_date, mock_stock_data):
        """Test the background training task function"""        
        # Setup mocks
        mock_get_latest_date.return_value = "2023-06-01"
        mock_load_data.return_value = mock_stock_data
        mock_train_pipeline.return_value = MOCK_TRAINING_RESULT
        
        # Initialize the task in the global dict
        task_id = MOCK_TASK_ID
        training_tasks[task_id] = {
            "ticker": "AAPL",
            "forecast_days": 7,
            "status": TrainingStatus.PENDING,
            "created_at": datetime.now(),
            "result": None
        }
        
        # Run the task (we'll run it directly rather than as a background task)
        import asyncio
        asyncio.run(train_stock_model_task(task_id, "AAPL", 7))
        
        # Verify the task was updated correctly
        assert training_tasks[task_id]["status"] == TrainingStatus.COMPLETED
        assert training_tasks[task_id]["result"] == MOCK_TRAINING_RESULT
        
        # Verify the function calls
        mock_get_latest_date.assert_called_once_with("AAPL")
        mock_load_data.assert_called_once()
        mock_train_pipeline.assert_called_once_with(ticker="AAPL", df=mock_stock_data, forecast_days=7)
    
    @patch("app.api.train.get_latest_date")
    def test_train_stock_model_task_error(self, mock_get_latest_date):
        """Test the background training task function with an error"""
        from app.api.train import train_stock_model_task, training_tasks
        
        # Setup mock to raise an exception
        mock_get_latest_date.side_effect = Exception("API error")
        
        # Initialize the task in the global dict
        task_id = MOCK_TASK_ID
        training_tasks[task_id] = {
            "ticker": "AAPL",
            "forecast_days": 7,
            "status": TrainingStatus.PENDING,
            "created_at": datetime.now(),
            "result": None
        }
        
        # Run the task
        import asyncio
        asyncio.run(train_stock_model_task(task_id, "AAPL", 7))
        
        # Verify the task was updated to failed status
        assert training_tasks[task_id]["status"] == TrainingStatus.FAILED
        
        # Verify the function calls
        mock_get_latest_date.assert_called_once_with("AAPL")
