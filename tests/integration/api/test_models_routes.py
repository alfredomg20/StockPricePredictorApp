from datetime import date
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)

MOCK_METRICS = {
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
    }
# Mock data for models
MOCK_MODEL_LIST = {
    "models": [
        {
            "ticker": "AAPL",
            "model_type": "linear",
            "version": "AAPL_linear_7day_20230101120000",
            "metrics": MOCK_METRICS.copy(),
        },
        {
            "ticker": "MSFT",
            "model_type": "linear",
            "version": "MSFT_linear_14day_20230115120000",
            "metrics": MOCK_METRICS.copy(),
        }
    ],
    "count": 2
}

MOCK_MODEL_INFO = {
    "ticker": "AAPL",
    "model_type": "linear",
    "version": "AAPL_linear_7day_20230101120000",
    "metrics": MOCK_METRICS.copy(),
    "model": MagicMock()
}

class TestModelRoutes:
    @classmethod
    def setup_class(cls):
        """Desactivate cache for tests using patching"""
        patcher = patch("app.api.models.cached_models")
        cls.mock_cache = patcher.start()
        cls.mock_cache.get.return_value = None
        cls.patcher = patcher

    @classmethod
    def teardown_class(cls):
        """Stop patcher after tests"""
        cls.patcher.stop()

    @patch("app.ml.model_store.ModelStore.get_all_models")
    def test_get_all_models_success(self, mock_get_all_models):
        """Test successful retrieval of all models"""
        # Setup mock
        mock_get_all_models.return_value = MOCK_MODEL_LIST
        
        # Make request
        response = client.get("/models/")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "count" in data
        assert data["count"] == 2
        assert len(data["models"]) == 2
        assert data["models"][0]["ticker"] == "AAPL"
        assert data["models"][1]["ticker"] == "MSFT"
    
    @patch("app.ml.model_store.ModelStore.get_all_models")
    def test_get_all_models_error(self, mock_get_all_models):
        """Test error handling when retrieving all models"""
        # Setup mock to raise exception
        mock_get_all_models.side_effect = Exception("Database error")
        
        # Make request
        response = client.get("/models/")
        
        # Verify response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Failed to retrieve models" in data["detail"]
    
    @patch("app.ml.model_store.ModelStore.get_model")
    def test_get_model_success(self, mock_get_model):
        """Test successful retrieval of a specific model"""
        # Setup mock
        mock_get_model.return_value = MOCK_MODEL_INFO
        
        # Make request
        response = client.get("/models/AAPL/7/2023-01-01-12:00:00")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["model_type"] == "linear"
        assert data["version"] == "AAPL_linear_7day_20230101120000"
        assert "metrics" in data
        assert data["metrics"]["mae"] == 1.5
        assert data["metrics"]["r2"] == 0.85
        assert data["metrics"]["features"] == ["lag1", "lag5", "ma5", "ma20"]
        assert data["metrics"]["coefficients"] == {"lag1": 0.2, "lag5": 0.1, "ma5": 0.3, "ma20": 0.4}
        assert data["metrics"]["intercept"] == 0.5
        assert data["metrics"]["train_samples"] == 500
        assert data["metrics"]["test_samples"] == 100
        assert data["metrics"]["last_train_date"] == "2023-06-01"
    
    @patch("app.ml.model_store.ModelStore.get_model")
    def test_get_model_not_found(self, mock_get_model):
        """Test 404 response when model is not found"""
        # Setup mock to raise FileNotFoundError
        mock_get_model.side_effect = FileNotFoundError("Model not found")
        
        # Make request
        response = client.get("/models/UNKNOWN/7/2023-01-01-12:00:00")
        
        # Verify response
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "Model not found" in data["detail"]
    
    @patch("app.ml.model_store.ModelStore.get_model")
    def test_get_model_invalid_input(self, mock_get_model):
        """Test 400 response when input is invalid"""
        # Setup mock to raise ValueError
        mock_get_model.side_effect = ValueError("Invalid ticker")
        
        # Make request
        response = client.get("/models/AAPL/7/2023-01-01-12:00:00")
        
        # Verify response
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Invalid ticker" in data["detail"]
    
    @patch("app.ml.model_store.ModelStore.get_model")
    def test_get_model_server_error(self, mock_get_model):
        """Test 500 response when server error occurs"""
        # Setup mock to raise general exception
        mock_get_model.side_effect = Exception("Database connection error")
        
        # Make request
        response = client.get("/models/AAPL/7/2023-01-01-12:00:00")
        
        # Verify response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Failed to retrieve model" in data["detail"]
    
    @patch("app.ml.model_store.ModelStore.delete_model")
    def test_delete_model_success(self, mock_delete_model):
        """Test successful model deletion"""
        # Setup mock
        mock_delete_model.return_value = {
            "status": "success",
            "version": "AAPL_linear_7day_20230101120000"
        }
        
        # Make request
        response = client.delete("/models/AAPL/7/2023-01-01-12:00:00")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["version"] == "AAPL_linear_7day_20230101120000"
    
    @patch("app.ml.model_store.ModelStore.delete_model")
    def test_delete_model_not_found(self, mock_delete_model):
        """Test 404 response when model to delete is not found"""
        # Setup mock to raise FileNotFoundError
        mock_delete_model.side_effect = FileNotFoundError("Model not found")
        
        # Make request
        response = client.delete("/models/UNKNOWN/7/2023-01-01-12:00:00")
        
        # Verify response
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "Model not found" in data["detail"]
    
    @patch("app.ml.model_store.ModelStore.delete_model")
    def test_delete_model_invalid_input(self, mock_delete_model):
        """Test 400 response when input for deletion is invalid"""
        # Setup mock to raise ValueError
        mock_delete_model.side_effect = ValueError("Invalid forecast days")
        
        # Make request
        response = client.delete("/models/AAPL/7/2023-01-01-12:00:00")
        
        # Verify response
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Invalid forecast days" in data["detail"]
    
    @patch("app.ml.model_store.ModelStore.delete_model")
    def test_delete_model_server_error(self, mock_delete_model):
        """Test 500 response when server error occurs during deletion"""
        # Setup mock to raise general exception
        mock_delete_model.side_effect = Exception("Filesystem error")
        
        # Make request
        response = client.delete("/models/AAPL/7/2023-01-01-12:00:00")
        
        # Verify response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Failed to delete model" in data["detail"]
    
    def test_get_model_invalid_path_params(self):
        """Test validation of path parameters"""
        # Test invalid ticker (too long)
        response = client.get("/models/AAAAAAAAAAAA/7/2023-01-01-12:00:00")
        assert response.status_code == 400
        
        # Test invalid forecast days (negative)
        response = client.get("/models/AAPL/-1/2023-01-01-12:00:00")
        assert response.status_code == 400
        
        # Test invalid forecast days (too large)
        response = client.get("/models/AAPL/366/2023-01-01-12:00:00")
        assert response.status_code == 400
        
        # Test invalid date format
        response = client.get("/models/AAPL/7/01-01-2023-12:00:00")
        assert response.status_code == 400
        assert response.status_code == 400
