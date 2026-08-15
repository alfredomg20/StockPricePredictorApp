from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import pytz
from sklearn.linear_model import LinearRegression

from app.exceptions import ModelTrainingError
from app.ml.trainer import StockPriceTrainer


# Mock data for stock data
@pytest.fixture
def mock_stock_data():
    dates = [(datetime(2023, 1, 1).astimezone(pytz.timezone('UTC')) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(100)]
    
    return pl.DataFrame({
        "date": dates,
        "price": np.random.normal(100, 10, 100),
        "lag1": np.random.normal(100, 10, 100),
        "lag5": np.random.normal(100, 10, 100),
        "lag20": np.random.normal(100, 10, 100),
        "lag100": np.random.normal(100, 10, 100),
        "ma5": np.random.normal(100, 5, 100),
        "ma20": np.random.normal(100, 5, 100),
        "ma100": np.random.normal(100, 5, 100)
    })

@pytest.fixture
def trainer():
    with patch('pathlib.Path.mkdir') as _:
        return StockPriceTrainer(model_dir="test_models", tz=pytz.timezone('UTC'))

class TestStockPriceTrainer:
    def test_init_creates_directory(self):
        """Test that constructor creates model directory"""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            _ = StockPriceTrainer(model_dir="test_models", tz=pytz.timezone('UTC'))
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    def test_validate_model_inputs_valid(self, trainer, mock_stock_data):
        """Test validation with valid inputs"""
        # Should not raise exceptions
        trainer._validate_inputs("AAPL", 7, mock_stock_data)
    
    @pytest.mark.parametrize("ticker", ["", " ", None, "A"*11])
    def test_validate_model_inputs_invalid_ticker(self, trainer, mock_stock_data, ticker):
        """Test validation with invalid ticker"""
        with pytest.raises(ValueError):
            trainer._validate_inputs(ticker, 7, mock_stock_data)
    
    def test_validate_model_inputs_invalid_dataframe(self, trainer):
        """Test validation with invalid dataframe"""
        with pytest.raises(ValueError):
            trainer._validate_inputs("AAPL", 7, pl.DataFrame())
    
    @pytest.mark.parametrize("forecast_days", [0, -1, "7", 366])
    def test_validate_model_inputs_invalid_forecast_days(self, trainer, mock_stock_data, forecast_days):
        """Test validation with invalid forecast days"""
        with pytest.raises(ValueError):
            trainer._validate_inputs("AAPL", forecast_days, mock_stock_data)
    
    @pytest.mark.parametrize("test_size,should_raise,expected_exception", [
        (0.2, False, None),
        (-0.1, True, ValueError),
        (0, True, ValueError),
        (1, True, ValueError),
        (1.1, True, ValueError),
        ("0.2", True, TypeError),
    ])
    def test_validate_model_params_test_size(self, trainer, test_size, should_raise, expected_exception):
        """Test validation of test_size parameter"""
        X = pl.DataFrame({"feature1": [1, 2, 3]})
        y = pl.Series("target", [4, 5, 6])
        
        if should_raise:
            with pytest.raises(expected_exception):
                trainer._validate_model_params(test_size, X, y)
        else:
            # Should not raise exception
            trainer._validate_model_params(test_size, X, y)
    
    def test_validate_model_params_invalid_X_y(self, trainer):
        """Test validation with invalid X and y"""
        # Empty X
        with pytest.raises(ValueError):
            trainer._validate_model_params(0.2, pl.DataFrame(), pl.Series("y", [1, 2, 3]))
        
        # Empty y
        with pytest.raises(ValueError):
            trainer._validate_model_params(0.2, pl.DataFrame({"x": [1, 2, 3]}), pl.Series("y", []))
        
        # Mismatched lengths
        with pytest.raises(ValueError):
            trainer._validate_model_params(0.2, pl.DataFrame({"x": [1, 2, 3]}), pl.Series("y", [1, 2]))
    
    @pytest.mark.parametrize("forecast_days,expected_feature_count", [
        (1, 4),  # short_term
        (5, 4),  # short_term
        (10, 4), # medium_term
        (20, 4), # medium_term
        (30, 3), # long_term
    ])
    def test_select_features(self, trainer, forecast_days, expected_feature_count):
        """Test feature selection based on forecast days"""
        features = trainer._select_features(forecast_days)
        assert len(features) == expected_feature_count
        
        # Verify correct feature set based on forecast horizon
        if forecast_days <= 5:
            assert "lag1" in features
            assert "ma5" in features
        elif forecast_days <= 20:
            assert "lag5" in features
            assert "ma20" in features
        else:
            assert "lag20" in features
            assert "ma100" in features
    
    def test_prepare_features_target(self, trainer, mock_stock_data):
        """Test feature and target preparation"""
        X, y = trainer.prepare_features_target(mock_stock_data, forecast_days=5)
        
        # Check X contains date and selected features
        assert "date" in X.columns
        assert all(f in X.columns for f in ["lag1", "lag5", "ma5", "ma20"])
        
        # Check y is the target (price shifted by forecast_days)
        assert len(y) == len(X)
        assert len(y) < len(mock_stock_data)  # Should be smaller due to shifting
        
    @patch('app.ml.trainer.LinearRegression', autospec=True)
    @patch('app.ml.trainer.mean_absolute_error')
    @patch('app.ml.trainer.r2_score')
    @patch('app.ml.trainer.mean_absolute_percentage_error')
    @patch('app.ml.trainer.max_error')
    def test_train_linear_model(self, mock_max_ae, mock_mape, mock_r2, mock_mae, mock_linear_regression, trainer, mock_stock_data):
        """Test training of linear model"""
        # Setup mocks
        mock_mae.return_value = 1.5
        mock_r2.return_value = 0.85
        mock_mape.return_value = 0.1
        mock_max_ae.return_value = 2.0
        
        # Prepare data
        X, y = trainer.prepare_features_target(mock_stock_data, forecast_days=5)
        
        # Create test dates that match our mock data
        test_dates = [(datetime(2023, 1, 1).astimezone(pytz.timezone('UTC')) + timedelta(days=i)).strftime("%Y-%m-%d") 
                    for i in range(len(X))]
        
        # Configure mock model
        mock_model = mock_linear_regression.return_value
        mock_model.coef_ = np.array([0.1, 0.2, 0.3, 0.4])  # Assuming 4 features
        mock_model.intercept_ = 0.5
        mock_model.predict.return_value = np.array([100] * 20)  # Sample predictions
        
        # Mock the dates array that will be used
        with patch.object(X, 'select') as mock_select:
            # Return a dataframe with our test dates when date column is selected
            mock_select.return_value = pl.DataFrame({"date": test_dates})
            
            # Train model
            model, metrics = trainer.train_linear_model(X, y, test_size=0.2)
        
        # Verify the mock was used
        mock_linear_regression.assert_called_once()
        mock_model.fit.assert_called_once()
        
        # Check metrics
        assert metrics["mae"] == 1.5
        assert metrics["r2"] == 0.85
        assert metrics["mape"] == 0.1
        assert metrics["max_ae"] == 2.0
        assert metrics["coefficients"] == {"lag1": 0.1, "lag5": 0.2, "ma5": 0.3, "ma20": 0.4}
        assert metrics["intercept"] == 0.5
        assert isinstance(metrics["last_train_date"], str)
        parsed = datetime.strptime(metrics["last_train_date"], "%Y-%m-%d").replace(tzinfo=pytz.timezone('UTC')).date()
        assert isinstance(parsed, date)

        # Assert model object is returned correctly
        assert model is mock_model
        assert list(model.coef_) == [0.1, 0.2, 0.3, 0.4]
        assert model.intercept_ == 0.5
        
    @patch('app.ml.trainer.StockPriceTrainer.prepare_features_target')
    @patch('app.ml.trainer.StockPriceTrainer.train_linear_model')
    @patch('app.ml.trainer.StockPriceTrainer.save_model')
    def test_training_pipeline_success(self, mock_save, mock_train, mock_prepare, trainer, mock_stock_data):
        """Test successful training pipeline execution"""
        # Setup mocks
        mock_X = pl.DataFrame({"date": ["2023-01-01"], "feature1": [1]})
        mock_y = pl.Series("target", [100])
        mock_prepare.return_value = (mock_X, mock_y)
        
        mock_model = LinearRegression()
        mock_metrics = {"mae": 1.5, "r2": 0.85, "mape": 0.1, "max_ae": 2.0}
        mock_train.return_value = (mock_model, mock_metrics)
        
        mock_path = Path("/test/path/model.joblib")
        mock_save.return_value = mock_path
        
        # Run pipeline
        result = trainer.training_pipeline("AAPL", mock_stock_data, forecast_days=7)
        
        # Check result structure
        assert result["status"] == "success"
        assert result["ticker"] == "AAPL"
        assert result["forecast_days"] == 7
        assert result["model_path"] == str(mock_path)
        assert result["metrics"] == mock_metrics
        
        # Verify calls
        mock_prepare.assert_called_once_with(mock_stock_data, 7)
        mock_train.assert_called_once_with(mock_X, mock_y)
        mock_save.assert_called_once_with(mock_model, "AAPL", 7, mock_metrics)
    
    def test_training_pipeline_validation_error(self, trainer, mock_stock_data):
        """Test that validation errors raise ValueError before entering try block"""
        with patch.object(trainer, '_validate_inputs') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid ticker")
            
            with pytest.raises(ValueError) as exc_info:
                trainer.training_pipeline("INVALID", mock_stock_data, 7)
                
            assert "Invalid ticker" in str(exc_info.value)
    
    @patch('app.ml.trainer.StockPriceTrainer.prepare_features_target')
    def test_training_pipeline_processing_error(self, mock_prepare, trainer, mock_stock_data):
        """Test training pipeline with processing error re-raises ValueError"""
        mock_prepare.side_effect = ValueError("Data processing failed")
        
        with pytest.raises(ValueError, match="Data processing failed"):
            trainer.training_pipeline("AAPL", mock_stock_data, 7)
    
    @patch('app.ml.trainer.StockPriceTrainer.train_linear_model')
    def test_training_pipeline_unexpected_error(self, mock_train, trainer, mock_stock_data):
        """Test training pipeline wraps unexpected errors in ModelTrainingError"""
        mock_train.side_effect = RuntimeError("Unexpected training error")
        
        with pytest.raises(ModelTrainingError) as excinfo:
            trainer.training_pipeline("AAPL", mock_stock_data, 7)
        
        assert excinfo.value.ticker == "AAPL"
        assert "Unexpected training error" in excinfo.value.details