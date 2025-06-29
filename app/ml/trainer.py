from datetime import datetime
from typing import Tuple
import polars as pl
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, max_error, r2_score
from pathlib import Path
from app.api.models import invalidate_models_cache
from app.api.predict import invalidate_prediction_cache
from app.config import logger, timezone
from app.ml.data_loader import invalidate_latest_date_cache, invalidate_ticker_data_cache
from app.utils.time_utils import get_last_business_day, is_business_day
from app.utils.validation_utils import validate_ticker, validate_forecast_days, validate_dataframe, validate_columns

class StockPriceTrainer:
    """Handle training of stock price prediction models using linear regression."""
    def __init__(self, model_dir: str = "app/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def _validate_inputs(self, ticker: str, forecast_days: int, df: pl.DataFrame | None = None) -> None:
        """
        Validate inputs for model training.
        
        Args:
            ticker: Stock ticker symbol
            forecast_days: Number of days to forecast
            df: DataFrame with stock data (optional, if provided)
        Raises:
            ValueError: If any input is invalid
        """
        validate_ticker(ticker)
        validate_forecast_days(forecast_days)
        if df is not None:
            validate_dataframe(df)
            required_columns = ["date", "price", "lag1", "lag5", "lag20", "lag100", "ma5", "ma20", "ma100"]
            validate_columns(df, required_columns)
        
    def _validate_model_params(self, test_size: float, X: pl.DataFrame, y: pl.Series) -> None:
        """
        Validate model parameters for training.
        Args:
            test_size: Proportion of the dataset to include in the test split
            X: Feature matrix
            y: Target vector
        
        Raises:
            ValueError: If test_size, X, or y are invalid
        """
        if not isinstance(test_size, (float, int)):
            raise ValueError(f"Invalid test_size type='{type(test_size)}'. Must be float or int")
        if not (0 < test_size < 1):
            raise ValueError(f"Invalid test_size={test_size}. Must be 0 < x < 1")
        if X.is_empty() or y.is_empty():
            raise ValueError("X and y cannot be empty")
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X has {len(X)} rows, y has {len(y)} rows. They must be the same length.")

    def _invalidate_caches(self, ticker: str, forecast_days: int) -> None:
        """
        Invalidate caches related to the model and predictions.
        
        Args:
            ticker: Stock ticker symbol
            forecast_days: Number of days to forecast
        """
        invalidate_models_cache()
        invalidate_latest_date_cache(ticker)
        invalidate_ticker_data_cache(ticker)
        invalidate_prediction_cache(ticker, forecast_days)
    
    def _select_features(self, forecast_days: int) -> list[str]:
        """
        Select features based on the forecast days.
        
        Args:
            forecast_days: Number of days to forecast
            
        Returns:
            List of feature names
        """
        feature_sets = {
        'short_term': ["lag1", "lag5", "ma5", "ma20"],
        'medium_term': ["lag5", "lag20", "ma20", "ma100"],
        'long_term': ["lag20", "lag100", "ma100"]
        }
        if forecast_days <= 5:
            return feature_sets['short_term']
        elif forecast_days <= 20:
            return feature_sets['medium_term']
        else:
            return feature_sets['long_term']

    def _get_latest_model_path(self, ticker: str, forecast_days: int) -> Path | None:
        """
        Get the path to the latest model for a given ticker and forecast days.
        
        Args:
            ticker: Stock ticker symbol
            forecast_days: Number of days to forecast
            
        Returns:
            Path: Path to the latest model file, or None if no model exists
        """
        # Search for model files matching the pattern
        pattern = f"{ticker}_linear_{forecast_days}day_*.joblib"
        model_files = list(self.model_dir.glob(pattern))
        if not model_files:
            return None
        # Sort by modification time and return the latest one
        latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
        return latest_model

    def needs_training(self, ticker: str, forecast_days: int, validate_inputs: bool = True) -> bool:
        """
        Evaluate if a new model needs to be trained based on the last trained time and current market conditions.
        
        Args:
            ticker: Stock ticker symbol
            forecast_days: Number of days to forecast
            validate_inputs: Whether to validate inputs before checking (not required for internal use)
        Returns:
            bool: True if a new model needs to be trained, False otherwise
        """
        # Validate inputs if required
        if validate_inputs:
            self._validate_inputs(ticker, forecast_days)
        
        # Get the latest model and its last trained time
        latest_model_path = self._get_latest_model_path(ticker, forecast_days)
        if not latest_model_path:
            return True
        last_trained_time_str = latest_model_path.stem.split("_")[-1]
        last_trained_time = datetime.strptime(last_trained_time_str, "%Y%m%d%H%M%S")

        # Evaluate if the model needs training based on last trained time and last market close time
        now = datetime.now(timezone)
        now = now.replace(tzinfo=None)
        today = now.date()
        today_is_business_day = is_business_day(today)
        if today_is_business_day:
            market_close_time = now.replace(hour=16, minute=30, second=0)
            return last_trained_time < market_close_time < now
        last_business_day = get_last_business_day(today)
        market_close_time = datetime(last_business_day.year, last_business_day.month, last_business_day.day, 16, 30, 0)    
        return last_trained_time < market_close_time
    
    def prepare_features_target(
        self, 
        df: pl.DataFrame,
        forecast_days: int = 1
    ) -> Tuple[pl.DataFrame, pl.Series]:
        """
        Prepare features and target with horizon-aware feature selection.
        
        Args:
            df: DataFrame with columns ['date', 'price', 'lag1', 'lag5', 'lag20', 'lag100', 'ma5', 'ma20', 'ma100']
            forecast_days: Days ahead to predict
            
        Returns:
            X: Feature matrix
            y: Target vector
        """

        # Create target variable for forecast future prices
        df = df.with_columns(
            pl.col("price").shift(-forecast_days).alias("target")
        )
        
        # Drop rows with missing values
        df = df.drop_nulls()
        
        features = self._select_features(forecast_days)
        # Include 'date' for time-series context
        X = df.select(["date"] + features)
        y = df["target"]
        
        return X, y

    def train_linear_model(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        test_size: float = 0.2
    ) -> Tuple[LinearRegression, dict[str, float | str]]:
        """
        Train linear regression model with time-series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for test split
            
        Returns:
            model: Trained LinearRegression
            metrics: Evaluation metrics
        """
        # Validate model parameters
        self._validate_model_params(test_size, X, y)
        
        # Save dates for later use
        dates = X["date"].to_numpy()
        
        # Extract feature columns
        feature_cols = [col for col in X.columns if col not in ["date", "target"]]
        X_features = X.select(feature_cols)
        
        # Convert to numpy arrays for sklearn
        X_np = X_features.to_numpy()
        y_np = y.to_numpy()
        
        # Time-series aware train-test split
        tscv = TimeSeriesSplit(n_splits=int(1/test_size))
        for train_index, test_index in tscv.split(X_np):
            X_train, X_test = X_np[train_index], X_np[test_index]
            y_train, y_test = y_np[train_index], y_np[test_index]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        max_ae = max_error(y_test, y_pred)

        metrics = {
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "max_ae": max_ae,
            "features": feature_cols,
            "coefficients": dict(zip(feature_cols, model.coef_)),
            "intercept": float(model.intercept_),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "last_train_date": datetime.strptime(str(dates[train_index[-1]]), "%Y-%m-%d"),
        }
        
        return model, metrics
    
    def save_model(
        self,
        model: LinearRegression,
        ticker: str,
        forecast_days: int,
        metrics: dict[str, float]
    ) -> Path:
        """
        Save linear regression model with metadata.
        Args:
            model: Trained LinearRegression model
            ticker: Stock ticker symbol
            forecast_days: Number of days to forecast
            metrics: Evaluation metrics
        Returns:
            Path: Path to the saved model file
        """
        model_data = {
            "model": model,
            "metadata": {
                "ticker": ticker,
                "forecast_days": forecast_days,
                "model_type": "LinearRegression",
                **metrics,
            }
        }
        current_time_str = datetime.now(timezone).strftime("%Y%m%d%H%M%S")
        filename = f"{ticker}_linear_{forecast_days}day_{current_time_str}.joblib"
        path = self.model_dir / filename
        joblib.dump(model_data, path)
        
        return path

    def training_pipeline(
        self,
        ticker: str,
        df: pl.DataFrame,
        forecast_days: int = 1
    ) -> dict[str, any]:
        """
        Complete training workflow for linear regression:
        1. Feature engineering
        2. Model training
        3. Save model

        Args:
            ticker (str): Stock ticker symbol.
            df (pl.DataFrame): DataFrame with stock data containing columns ['date', 'price', 'lag1', 'lag5', 'lag20', 'lag100', 'ma5', 'ma20', 'ma100'].
            forecast_days (int): Number of days to forecast.
        Returns:
            Dict[str, any]: Result of the training process including model path and metrics.
        """
        # Validate inputs
        self._validate_inputs(ticker, forecast_days, df)
        try:
            # Check if model needs training
            logger.info("Checking if model needs training...")
            if not self.needs_training(ticker, forecast_days, validate_inputs=False):
                logger.info(f"Model for {ticker} with {forecast_days} days already trained and up-to-date.")
                latest_model_path = self._get_latest_model_path(ticker, forecast_days)
                return {
                    "status": "skipped",
                    "ticker": ticker,
                    "forecast_days": forecast_days,
                    "model_path": str(latest_model_path),
                    "message": "Model is already trained and up-to-date."
                }
            
            # Delete caches for this ticker and forecast days
            self._invalidate_caches(ticker, forecast_days)

            # Prepare data for training
            logger.info(f"Starting training for {ticker} with {forecast_days} days forecast...")
            logger.info("Preparing features and target...")
            X, y = self.prepare_features_target(df, forecast_days)
            logger.info("Features and target prepared")
            
            # Train model with selected features and target
            logger.info("Training linear regression model...")
            model, metrics = self.train_linear_model(X, y)
            logger.info("Model trained successfully")
            
            # Persist model to disk
            logger.info("Saving model...")
            model_path = self.save_model(model, ticker, forecast_days, metrics)
            logger.info(f"Model saved successfully to {model_path}")
            
            return {
                "status": "success",
                "ticker": ticker,
                "forecast_days": forecast_days,
                "model_path": str(model_path),
                "metrics": metrics
            }
        except ValueError as ve:
            logger.error(f"Training failed for {ticker} with {forecast_days} days: {str(ve)}")
            return {
                "status": "error",
                "ticker": ticker,
                "forecast_days": forecast_days,
                "error": str(ve)
            }
        except Exception as e:
            logger.critical(f"Unexpected error during training for {ticker} with {forecast_days} days: {str(e)}")
            raise e
        
