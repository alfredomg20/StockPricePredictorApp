import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from app.ml.model_store import ModelStore
from app.config import MODELS_DIR

# Mock data for models
MOCK_MODEL_DATA = {
    "metadata": {
        "ticker": "AAPL",
        "forecast_days": 7,
        "model_type": "linear",
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
        "last_train_date": datetime(2023, 1, 1, 12, 0, 0)
    },
    "model": MagicMock()
}

@pytest.fixture
def mock_path():
    with patch("app.ml.model_store.Path") as mock:
        # Setup mock directory that exists
        mock_dir = MagicMock()
        mock_dir.mkdir.return_value = None
        mock_dir.exists.return_value = True
        
        # Setup mock model file
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_file.stem = "AAPL_linear_7day_20230101120000"
        
        # Make Path return the mock directory and have glob return mock files
        mock.return_value = mock_dir
        mock_dir.__truediv__.return_value = mock_file
        mock_dir.glob.return_value = [mock_file]
        
        yield mock

@pytest.fixture
def mock_joblib():
    with patch("app.ml.model_store.joblib") as mock:
        mock.load.return_value = MOCK_MODEL_DATA
        yield mock

class TestModelStore:
    def test_init_creates_directory(self, mock_path):
        """Test that constructor creates model directory"""
        model_store = ModelStore(MODELS_DIR)
        mock_path.assert_called_once_with(MODELS_DIR)
        mock_path.return_value.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    def test_get_all_models_success(self, mock_path, mock_joblib):
        """Test successful retrieval of all models"""
        model_store = ModelStore(MODELS_DIR)
        result = model_store.get_all_models()
        
        assert isinstance(result, dict)
        assert "models" in result
        assert "count" in result
        assert len(result["models"]) == 1
        assert result["count"] == 1
        
        model = result["models"][0]
        assert model["ticker"] == "AAPL"
        assert model["model_type"] == "linear"
        assert "metrics" in model
        assert "mae" in model["metrics"]
        assert "r2" in model["metrics"]
        assert "mape" in model["metrics"]
        assert "max_ae" in model["metrics"]

    
    def test_get_all_models_no_models(self, mock_path):
        """Test when no models are found"""
        mock_path.return_value.glob.return_value = []
        model_store = ModelStore(MODELS_DIR)
        result = model_store.get_all_models()
        
        assert result["count"] == 0
        assert result["models"] == []
    
    def test_get_all_models_load_error(self, mock_path, mock_joblib):
        """Test handling of joblib load errors"""
        mock_joblib.load.side_effect = Exception("Failed to load model")
        
        model_store = ModelStore(MODELS_DIR)
        result = model_store.get_all_models()
        
        assert result["count"] == 0
        assert result["models"] == []
    
    def test_get_model_success(self, mock_path, mock_joblib):
        """Test successful retrieval of specific model"""
        model_store = ModelStore()
        result = model_store.get_model("AAPL", 7, "2023-01-01-12:00:00")
        
        assert isinstance(result, dict)
        assert "ticker" in result
        assert "model_type" in result
        assert "version" in result
        assert "model" in result
        assert result["ticker"] == "AAPL"
        assert result["metrics"]["mae"] == 1.5
        assert result["metrics"]["r2"] == 0.85
        assert result["metrics"]["mape"] == 0.9
        assert result["metrics"]["max_ae"] == 2.0
    
    def test_get_model_not_found(self, mock_path, mock_joblib):
        """Test when model is not found"""
        mock_path.return_value.__truediv__.return_value.exists.return_value = False
        
        model_store = ModelStore()
        with pytest.raises(FileNotFoundError):
            model_store.get_model("UNKNOWN", 7, "2023-01-01-12:00:00")
    
    def test_get_model_load_error(self, mock_path, mock_joblib):
        """Test handling of joblib load errors in get_model"""
        mock_joblib.load.side_effect = Exception("Failed to load model")
        
        model_store = ModelStore()
        with pytest.raises(Exception):
            model_store.get_model("AAPL", 7, "2023-01-01-12:00:00")
    
    @pytest.mark.parametrize("ticker", ["", " ", None, "A"*11])
    def test_get_model_invalid_ticker(self, ticker, mock_path):
        """Test invalid ticker values"""
        model_store = ModelStore(MODELS_DIR)
        with pytest.raises(ValueError):
            model_store.get_model(ticker, 7, "2023-01-01-12:00:00")
    
    @pytest.mark.parametrize("forecast_days", [0, -1, "7", 366])
    def test_get_model_invalid_forecast_days(self, forecast_days, mock_path):
        """Test invalid forecast days values"""
        model_store = ModelStore(MODELS_DIR)
        with pytest.raises(ValueError):
            model_store.get_model("AAPL", forecast_days, "2023-01-01-12:00:00")
    
    @pytest.mark.parametrize("date", ["", "invalid_date", "2023/01/01", "01-01-2023", "2023-13-01-12:00:00", "2023-01-32-12:00:00", "2023-01-01 12:00:00"])
    def test_get_model_invalid_date(self, date, mock_path):
        """Test invalid date values"""
        model_store = ModelStore(MODELS_DIR)
        with pytest.raises(ValueError):
            model_store.get_model("AAPL", 7, date)
    
    def test_delete_model_success(self, mock_path):
        """Test successful model deletion"""
        model_store = ModelStore(MODELS_DIR)
        result = model_store.delete_model("AAPL", 7, "2023-01-01-12:00:00")
        
        assert isinstance(result, dict)
        assert "status" in result
        assert "version" in result
        assert result["status"] == "success"
        mock_path.return_value.__truediv__.return_value.unlink.assert_called_once()
    
    def test_delete_model_not_found(self, mock_path):
        """Test when model to delete is not found"""
        mock_path.return_value.__truediv__.return_value.exists.return_value = False
        
        model_store = ModelStore(MODELS_DIR)
        with pytest.raises(FileNotFoundError):
            model_store.delete_model("UNKNOWN", 7, "2023-01-01-12:00:00")
    
    def test_delete_model_error(self, mock_path):
        """Test handling of errors during deletion"""
        mock_path.return_value.__truediv__.return_value.unlink.side_effect = Exception("Failed to delete")
        
        model_store = ModelStore(MODELS_DIR)
        with pytest.raises(Exception):
            model_store.delete_model("AAPL", 7, "2023-01-01-12:00:00")
    
    @pytest.mark.parametrize("ticker", ["", " ", None, "A"*11])
    def test_delete_model_invalid_ticker(self, ticker, mock_path):
        """Test invalid ticker values for deletion"""
        model_store = ModelStore(MODELS_DIR)
        with pytest.raises(ValueError):
            model_store.delete_model(ticker, 7, "2023-01-01-12:00:00")
    
    @pytest.mark.parametrize("forecast_days", [0, -1, "7", 366])
    def test_delete_model_invalid_forecast_days(self, forecast_days, mock_path):
        """Test invalid forecast days values for deletion"""
        model_store = ModelStore(MODELS_DIR)
        with pytest.raises(ValueError):
            model_store.delete_model("AAPL", forecast_days, "2023-01-01-12:00:00")
    
    @pytest.mark.parametrize("date", ["", "2023/01/01", "01-01-2023", "2023-13-01-12:00:00", "2023-01-01 12:00:00"])
    def test_delete_model_invalid_date(self, date, mock_path):
        """Test invalid date values for deletion"""
        model_store = ModelStore(MODELS_DIR)
        with pytest.raises(ValueError):
            model_store.delete_model("AAPL", 7, date)
