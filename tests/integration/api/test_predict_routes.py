from fastapi.testclient import TestClient
from unittest.mock import patch
import polars as pl
from datetime import date, timedelta
from app.main import app

client = TestClient(app)

# Mock data for predictions
MOCK_PREDICTIONS = pl.DataFrame({
    "date": [
        date.today() + timedelta(days=1),
        date.today() + timedelta(days=2),
        date.today() + timedelta(days=3),
    ],
    "predicted_price": [150.25, 152.75, 148.50]
})

class TestPredictionRoutes:
    @classmethod
    def setup_class(cls):
        """Desactivate cache for tests using patching"""
        patcher = patch("app.api.predict.cached_predictions")
        cls.mock_cache = patcher.start()
        cls.mock_cache.get.return_value = None
        cls.patcher = patcher

    @classmethod
    def teardown_class(cls):
        """Stop patcher after tests"""
        cls.patcher.stop()

    @patch("app.ml.predictor.StockPricePredictor.predict_prices")
    def test_predict_stock_price_success(self, mock_predict_prices):
        """Test successful stock price prediction"""
        # Setup mock
        mock_predict_prices.return_value = MOCK_PREDICTIONS
        
        # Make request
        response = client.post(
            "/predict/",
            json={"ticker": "AAPL", "forecast_days": 7}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["ticker"] == "AAPL"
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        
        # Check prediction structure
        first_prediction = data["predictions"][0]
        assert "predicted_date" in first_prediction
        assert "predicted_price" in first_prediction
        assert first_prediction["predicted_price"] == 150.25
        
        # Verify the function call
        mock_predict_prices.assert_called_once_with("AAPL", 7)

    @patch("app.ml.predictor.StockPricePredictor.predict_prices")
    def test_predict_stock_price_model_not_found(self, mock_predict_prices):
        """Test 404 error when model is not found"""
        # Setup mock to simulate model not found
        mock_predict_prices.side_effect = FileNotFoundError("No model found for UNKNOWN")
        
        # Make request
        response = client.post(
            "/predict/",
            json={"ticker": "UNKNOWN", "forecast_days": 7}
        )
        
        # Verify response
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "No model found" in data["detail"]
    
    @patch("app.ml.predictor.StockPricePredictor.predict_prices")
    def test_predict_stock_price_validation_error(self, mock_predict_prices):
        """Test 400 error when validation fails"""
        # Setup mock to simulate validation error
        mock_predict_prices.side_effect = ValueError("Invalid ticker symbol")
        
        # Make request
        response = client.post(
            "/predict/",
            json={"ticker": "INVALID", "forecast_days": 7}
        )
        
        # Verify response
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Invalid ticker symbol" in data["detail"]
    
    @patch("app.ml.predictor.StockPricePredictor.predict_prices")
    def test_predict_stock_price_internal_error(self, mock_predict_prices):
        """Test 500 error when unexpected exception occurs"""
        # Setup mock to simulate internal error
        mock_predict_prices.side_effect = Exception("Database connection error")
        
        # Make request
        response = client.post(
            "/predict/",
            json={"ticker": "AAPL", "forecast_days": 7}
        )
        
        # Verify response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Failed to generate prediction" in data["detail"]
    
    def test_predict_stock_price_invalid_request(self):
        """Test validation errors in request body"""
        # Test with missing ticker
        response = client.post(
            "/predict/",
            json={"forecast_days": 7}
        )
        assert response.status_code == 422
        
        # Test with invalid forecast days (too high)
        response = client.post(
            "/predict/",
            json={"ticker": "AAPL", "forecast_days": 366}
        )
        assert response.status_code == 422
        
        # Test with invalid forecast days (negative)
        response = client.post(
            "/predict/",
            json={"ticker": "AAPL", "forecast_days": -1}
        )
        assert response.status_code == 422
        
        # Test with too long ticker
        response = client.post(
            "/predict/",
            json={"ticker": "TOOLONGTICKERCODE", "forecast_days": 7}
        )
        assert response.status_code == 422
    
    @patch("app.ml.predictor.StockPricePredictor.predict_prices")
    def test_predict_stock_price_error_response(self, mock_predict_prices):
        """Test when predictor returns error response instead of DataFrame"""
        # Setup mock to return error dict instead of DataFrame
        mock_predict_prices.return_value = {
            "status": "error",
            "ticker": "AAPL",
            "error": "Insufficient historical data"
        }
        
        # Make request
        response = client.post(
            "/predict/",
            json={"ticker": "AAPL", "forecast_days": 7}
        )
        
        # Verify response shows an error
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Failed to generate prediction" in data["detail"]

    @patch("app.ml.predictor.StockPricePredictor.predict_prices")
    def test_predict_stock_price_empty_predictions(self, mock_predict_prices):
        """Test when predictor returns empty DataFrame"""
        # Setup mock to return empty DataFrame
        mock_predict_prices.return_value = pl.DataFrame({
            "date": [],
            "predicted_price": []
        })
        
        # Make request
        response = client.post(
            "/predict/",
            json={"ticker": "AAPL", "forecast_days": 7}
        )
        
        # Verify response succeeds but has empty predictions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["ticker"] == "AAPL"
        assert "predictions" in data
        assert len(data["predictions"]) == 0
