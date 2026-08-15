import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import polars as pl
import pytz
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit

from app.api.models import invalidate_models_cache
from app.api.predict import invalidate_prediction_cache
from app.exceptions import ModelTrainingError
from app.ml.data_loader import (
    invalidate_latest_date_cache,
    invalidate_ticker_data_cache,
)
from app.utils.time_utils import get_last_business_day, is_business_day
from app.utils.validation_utils import (
    validate_columns,
    validate_dataframe,
    validate_forecast_days,
    validate_ticker,
)

logger = logging.getLogger('app')

class StockPriceTrainer:
    """Handle training of stock price prediction models using linear regression."""

    def __init__(self, model_dir: str | Path, tz: pytz.BaseTzInfo):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tz = tz

    def _validate_inputs(self, ticker: str, forecast_days: int, df: pl.DataFrame | None = None) -> None:
        """Validate inputs for model training."""
        validate_ticker(ticker)
        validate_forecast_days(forecast_days)
        if df is not None:
            validate_dataframe(df)
            required_columns = ["date", "price", "lag1", "lag5", "lag20", "lag100", "ma5", "ma20", "ma100"]
            validate_columns(df, required_columns)

    def _validate_model_params(self, test_size: float, X: pl.DataFrame, y: pl.Series) -> None:
        """Validate model parameters for training."""
        if not isinstance(test_size, (float, int)):
            raise TypeError(f"Invalid test_size type='{type(test_size)}'. Must be float or int")
        if not (0 < test_size < 1):
            raise ValueError(f"Invalid test_size={test_size}. Must be 0 < x < 1")
        if X.is_empty() or y.is_empty():
            raise ValueError("X and y cannot be empty")
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X has {len(X)} rows, y has {len(y)} rows.")

    def _invalidate_caches(self, ticker: str, forecast_days: int) -> None:
        """Invalidate caches related to the model and predictions."""
        invalidate_models_cache()
        invalidate_latest_date_cache(ticker)
        invalidate_ticker_data_cache(ticker)
        invalidate_prediction_cache(ticker, forecast_days)

    def _select_features(self, forecast_days: int) -> list[str]:
        """Select features based on the forecast days."""
        feature_sets = {
            "short_term": ["lag1", "lag5", "ma5", "ma20"],
            "medium_term": ["lag5", "lag20", "ma20", "ma100"],
            "long_term": ["lag20", "lag100", "ma100"],
        }
        if forecast_days <= 5:
            return feature_sets["short_term"]
        elif forecast_days <= 20:
            return feature_sets["medium_term"]
        else:
            return feature_sets["long_term"]

    def _get_latest_model_path(self, ticker: str, forecast_days: int) -> Path | None:
        """Get the path to the latest model for a given ticker and forecast days."""
        pattern = f"{ticker}_linear_{forecast_days}day_*.joblib"
        model_files = list(self.model_dir.glob(pattern))
        if not model_files:
            return None
        return max(model_files, key=lambda f: f.stat().st_mtime)

    def needs_training(self, ticker: str, forecast_days: int, validate_inputs: bool = True) -> bool:
        """Evaluate if a new model needs to be trained based on timestamps."""
        if validate_inputs:
            self._validate_inputs(ticker, forecast_days)

        latest_model_path = self._get_latest_model_path(ticker, forecast_days)
        if not latest_model_path:
            return True

        try:
            last_trained_time_str = latest_model_path.stem.split("_")[-1]
            last_trained_time = datetime.strptime(last_trained_time_str, "%Y%m%d%H%M%S").astimezone(self.tz)
        except ValueError:
            return True

        now = datetime.now(tz=self.tz)
        today = now.date()

        if is_business_day(today):
            market_close_time = now.replace(hour=16, minute=30, second=0, microsecond=0)
            return last_trained_time < market_close_time < now

        last_business_day = get_last_business_day(today)
        market_close_time = datetime(last_business_day.year, last_business_day.month, last_business_day.day, 16, 30, 0).astimezone(self.tz)
        return last_trained_time < market_close_time

    def prepare_features_target(
        self, df: pl.DataFrame, forecast_days: int = 1
    ) -> tuple[pl.DataFrame, pl.Series]:
        """Prepare features and target with horizon-aware feature selection."""
        df = df.with_columns(pl.col("price").shift(-forecast_days).alias("target"))
        df = df.drop_nulls()

        features = self._select_features(forecast_days)
        X = df.select(["date"] + features)
        y = df["target"]

        return X, y

    def train_linear_model(
        self, X: pl.DataFrame, y: pl.Series, test_size: float = 0.2
    ) -> tuple[LinearRegression, dict[str, Any]]:
        """Train linear regression model with time-series cross-validation."""
        self._validate_model_params(test_size, X, y)

        last_date_val = X["date"].tail(1).item()
        if isinstance(last_date_val, datetime):
            last_train_date = last_date_val
        elif hasattr(last_date_val, "strftime"):
            last_train_date = datetime.combine(last_date_val, datetime.min.time())
        else:
            last_train_date = datetime.strptime(str(last_date_val)[:10], "%Y-%m-%d").astimezone(self.tz)

        feature_cols = [col for col in X.columns if col not in ["date", "target"]]
        X_features = X.select(feature_cols)

        X_np = X_features.to_numpy()
        y_np = y.to_numpy()

        n_splits = max(2, int(1 / test_size))
        
        if len(X_np) <= n_splits:
            raise ValueError(f"Not enough data points ({len(X_np)}) to perform TimeSeriesSplit with {n_splits} splits.")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for train_index, test_index in tscv.split(X_np):
            X_train, X_test = X_np[train_index], X_np[test_index]
            y_train, y_test = y_np[train_index], y_np[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics: dict[str, Any] = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_test, y_pred)),
            "max_ae": float(max_error(y_test, y_pred)),
            "features": feature_cols,
            "coefficients": dict(zip(feature_cols, model.coef_.tolist())),
            "intercept": float(model.intercept_),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "last_train_date": last_train_date.strftime("%Y-%m-%d"),
        }

        return model, metrics

    def save_model(
        self,
        model: LinearRegression,
        ticker: str,
        forecast_days: int,
        metrics: dict[str, Any],
    ) -> Path:
        """Save linear regression model atomically with metadata."""
        model_data = {
            "model": model,
            "metadata": {
                "ticker": ticker,
                "forecast_days": forecast_days,
                "model_type": "LinearRegression",
                **metrics,
            },
        }
        current_time_str = datetime.now(self.tz).strftime("%Y%m%d%H%M%S")
        filename = f"{ticker}_linear_{forecast_days}day_{current_time_str}.joblib"
        final_path = self.model_dir / filename
        temp_path = self.model_dir / f"{filename}.tmp"

        joblib.dump(model_data, temp_path)
        temp_path.replace(final_path)

        return final_path

    def training_pipeline(
        self, ticker: str, df: pl.DataFrame, forecast_days: int = 1
    ) -> dict[str, Any]:
        """Complete training workflow for linear regression."""
        self._validate_inputs(ticker, forecast_days, df)
        
        try:
            logger.info("Checking if model needs training...")
            if not self.needs_training(ticker, forecast_days, validate_inputs=False):
                logger.info(f"Model for {ticker} ({forecast_days} days) is already up-to-date.")
                latest_model_path = self._get_latest_model_path(ticker, forecast_days)
                return {
                    "status": "skipped",
                    "ticker": ticker,
                    "forecast_days": forecast_days,
                    "model_path": str(latest_model_path) if latest_model_path else "",
                    "message": "Model is already trained and up-to-date.",
                }

            self._invalidate_caches(ticker, forecast_days)

            logger.info(f"Starting training for {ticker} ({forecast_days} days forecast)...")
            X, y = self.prepare_features_target(df, forecast_days)

            model, metrics = self.train_linear_model(X, y)

            model_path = self.save_model(model, ticker, forecast_days, metrics)
            logger.info(f"Model saved successfully to {model_path}")

            return {
                "status": "success",
                "ticker": ticker,
                "forecast_days": forecast_days,
                "model_path": str(model_path),
                "metrics": metrics,
            }

        except ValueError:
            raise
        except Exception as e:
            raise ModelTrainingError(ticker=ticker, details=str(e)) from e