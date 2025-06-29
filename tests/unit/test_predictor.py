import pytest
from unittest.mock import patch, MagicMock
import polars as pl
from pathlib import Path
from datetime import date, timedelta
import numpy as np
from app.ml.predictor import StockPricePredictor

# Mock data for historical stock data
@pytest.fixture
def mock_historical_data():
    dates = [(date(2023, 1, 1) + timedelta(days=i)) for i in range(150)]
    prices = [100 + i * 0.1 + np.random.normal(0, 1) for i in range(150)]
    
    return pl.DataFrame({
        "date": dates,
        "price": prices,
        "lag1": [0.0] * 150,
        "lag5": [0.0] * 150,
        "lag20": [0.0] * 150,
        "lag100": [0.0] * 150,
        "ma5": [0.0] * 150,
        "ma20": [0.0] * 150,
        "ma100": [0.0] * 150
    })

# Mock model data
@pytest.fixture
def mock_model_data():
    model = MagicMock()
    model.predict.return_value = np.array([105.5])
    
    return {
        "model": model,
        "metadata": {
            "ticker": "AAPL",
            "forecast_days": 5,
            "features": ["lag1", "lag5", "ma5", "ma20"],
            "model_type": "LinearRegression"
        }
    }

@pytest.fixture
def predictor():
    return StockPricePredictor(model_dir="test_models")

class TestStockPricePredictor:
    def test_init(self):
        """Test initialization of StockPricePredictor"""
        predictor = StockPricePredictor(model_dir="custom_dir")
        assert predictor.model_dir == Path("custom_dir")
    
    def test_validate_prediction_inputs_valid(self, predictor):
        """Test validation with valid inputs"""
        # Should not raise exceptions
        predictor._validate_prediction_inputs("AAPL", 7)
    
    @pytest.mark.parametrize("ticker", ["", " ", None, "A"*11])
    def test_validate_prediction_inputs_invalid_ticker(self, predictor, ticker):
        """Test validation with invalid ticker"""
        with pytest.raises(ValueError):
            predictor._validate_prediction_inputs(ticker, 7)
    
    @pytest.mark.parametrize("forecast_days", [0, -1, "7", 366])
    def test_validate_prediction_inputs_invalid_forecast_days(self, predictor, forecast_days):
        """Test validation with invalid forecast days"""
        with pytest.raises(ValueError):
            predictor._validate_prediction_inputs("AAPL", forecast_days)
    
    @pytest.mark.parametrize("forecast_days,expected_features", [
        (1, ["lag1", "lag5", "ma5", "ma20"]),  # short_term
        (5, ["lag1", "lag5", "ma5", "ma20"]),  # short_term
        (10, ["lag5", "lag20", "ma20", "ma100"]), # medium_term
        (20, ["lag5", "lag20", "ma20", "ma100"]), # medium_term
        (30, ["lag20", "lag100", "ma100"]), # long_term
    ])
    def test_select_features(self, predictor, forecast_days, expected_features):
        """Test feature selection based on forecast days"""
        features = predictor._select_features(forecast_days)
        assert features == expected_features
    
    def test_generate_features_for_prediction(self, predictor, mock_historical_data):
        """Test feature generation for prediction"""
        # Create data for prediction date
        pred_date = (mock_historical_data["date"].max() + timedelta(days=1)).strftime("%Y-%m-%d")
        pred_df = pl.DataFrame({"date": [pred_date]})
        
        # Generate features for short-term prediction
        features = predictor.generate_features_for_prediction(
            mock_historical_data,
            pred_df,
            forecast_days=5
        )
        
        # Check output
        assert isinstance(features, pl.DataFrame)
        assert set(features.columns) == set(["lag1", "lag5", "ma5", "ma20"])
        assert features.shape[0] == 1
    
    def test_generate_features_missing_price(self, predictor):
        """Test feature generation with missing price column"""
        df = pl.DataFrame({"date": [date(2023, 1, 1)]})
        pred_df = pl.DataFrame({"date": [date(2023, 1, 2)]})
        
        with pytest.raises(ValueError, match="Historical DataFrame must contain 'price' column"):
            predictor.generate_features_for_prediction(df, pred_df)
    
    @patch('app.ml.predictor.joblib.load')
    def test_predict_price_success(self, mock_load, predictor, mock_historical_data, mock_model_data):
        """Test successful price prediction for a single date"""
        # Setup mocks
        mock_load.return_value = mock_model_data
        
        # Make prediction
        result = predictor.predict_price(
            "test_models/AAPL_linear_5day_20230501.joblib",
            mock_historical_data,
            "2023-05-02"
        )
        
        # Check result
        assert result["status"] == "success"
        assert result["ticker"] == "AAPL"
        assert result["date"] == "2023-05-02"
        assert result["predicted_price"] == 105.5
        assert result["forecast_days"] == 5
        assert "features_used" in result
    
    @patch('app.ml.predictor.joblib.load')
    def test_predict_price_error(self, mock_load, predictor, mock_historical_data):
        """Test error handling in price prediction"""
        # Setup mock to raise exception
        mock_load.side_effect = Exception("Failed to load model")
        
        # Make prediction
        result = predictor.predict_price(
            "test_models/AAPL_linear_5day_20230501.joblib",
            mock_historical_data,
            "2023-05-02"
        )
        
        # Check result
        assert result["status"] == "error"
        assert "Failed to load model" in result["error"]
    
    def test_find_latest_model_success(self, predictor):
        """Test finding the latest model file"""
        with patch('pathlib.Path.glob') as mock_glob:
            # Setup mock to return list of model files
            mock_file1 = MagicMock(spec=Path)
            mock_file1.name = "AAPL_linear_5day_20230501.joblib"
            mock_file2 = MagicMock(spec=Path)
            mock_file2.name = "AAPL_linear_5day_20230601.joblib"
            mock_glob.return_value = [mock_file1, mock_file2]
            
            # Find latest model
            result = predictor.find_latest_model("AAPL", 5)
            
            # Check result is the latest model
            assert result == str(mock_file2)
    
    def test_find_latest_model_not_found(self, predictor):
        """Test error when no model file is found"""
        with patch('pathlib.Path.glob') as mock_glob:
            # Setup mock to return empty list
            mock_glob.return_value = []
            
            # Try to find model
            with pytest.raises(FileNotFoundError):
                predictor.find_latest_model("UNKNOWN", 5)
    
    @patch('app.ml.predictor.StockPricePredictor.find_latest_model')
    @patch('app.ml.predictor.joblib.load')
    @patch('app.ml.predictor.get_latest_date')
    @patch('app.ml.predictor.load_ticker_data')
    @patch('app.ml.predictor.get_next_business_days')
    def test_predict_prices_success(self, mock_next_days, mock_load_data, 
                                   mock_latest_date, mock_load, mock_find_model, 
                                   predictor, mock_historical_data, mock_model_data):
        """Test successful prediction of multiple prices"""
        # Setup mocks
        mock_find_model.return_value = "test_models/AAPL_linear_5day_20230601.joblib"
        mock_load.return_value = mock_model_data
        mock_latest_date.return_value = "2023-06-01"
        mock_load_data.return_value = mock_historical_data
        
        # Mock future dates
        future_dates = [date(2023, 6, 2), date(2023, 6, 5)]
        mock_next_days.return_value = future_dates
        
        # Make predictions
        result = predictor.predict_prices("AAPL", 5)
        
        # Check result
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 2  # Two prediction dates
        assert "date" in result.columns
        assert "predicted_price" in result.columns
        assert result["date"][0] == future_dates[0]
    
    @patch('app.ml.predictor.StockPricePredictor.find_latest_model')
    def test_predict_prices_no_model(self, mock_find_model, predictor):
        """Test error when no model is found"""
        # Setup mock to raise exception
        mock_find_model.side_effect = FileNotFoundError("No model found")
        
        # Make predictions
        result = predictor.predict_prices("UNKNOWN", 5)
        
        # Check result
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "No model found" in result["error"]
    
    def test_predict_prices_invalid_inputs(self, predictor):
        """Test error with invalid inputs"""
        # Make predictions with invalid ticker
        with pytest.raises(ValueError, match="Ticker must be a non-empty string."):
            predictor.predict_prices("", 5)

        # Make predictions with invalid forecast days
        with pytest.raises(ValueError, match="Forecast days must be a positive integer."):
            predictor.predict_prices("AAPL", 0)
            
    @patch('app.ml.predictor.StockPricePredictor.find_latest_model')
    @patch('app.ml.predictor.joblib.load')
    @patch('app.ml.predictor.get_latest_date')
    @patch('app.ml.predictor.load_ticker_data')
    def test_predict_prices_unexpected_error(self, mock_load_data, mock_latest_date, 
                                           mock_load, mock_find_model, predictor):
        """Test handling of unexpected errors"""
        # Setup mocks for the normal flow
        mock_find_model.return_value = "test_models/AAPL_linear_5day_20230601.joblib"
        mock_load.return_value = {"model": MagicMock(), "metadata": {"forecast_days": 5}}
        mock_latest_date.return_value = "2023-06-01"
        
        # But make load_ticker_data raise an unexpected exception
        mock_load_data.side_effect = Exception("Unexpected error")
        
        # Make predictions and expect the exception to be re-raised
        with pytest.raises(Exception, match="Unexpected error"):
            predictor.predict_prices("AAPL", 5)
