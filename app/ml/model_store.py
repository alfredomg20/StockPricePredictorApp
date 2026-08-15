import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pytz

from app.exceptions import ModelCorruptedError, ModelNotFoundError
from app.utils.validation_utils import (
    validate_datetime,
    validate_forecast_days,
    validate_ticker,
)

logger = logging.getLogger('app')

class ModelStore:
    """Manage storage and retrieval of trained models."""

    def __init__(self, model_dir: Path | str, tz: pytz.BaseTzInfo):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tz = tz

    def _validate_model_params(self, ticker: str, forecast_days: int, last_trained_time: str) -> None:
        """Validate model parameters for retrieval or deletion."""
        validate_ticker(ticker)
        validate_forecast_days(forecast_days)
        validate_datetime(last_trained_time)

    def _build_model_filename(self, ticker: str, forecast_days: int, last_trained_time: str) -> tuple[str, Path]:
        """Helper to generate the standardized version string and file Path."""
        self._validate_model_params(ticker, forecast_days, last_trained_time)
        
        last_train_dt = datetime.strptime(last_trained_time, "%Y-%m-%d-%H:%M:%S").astimezone(self.tz)
        last_train_time_str = last_train_dt.strftime("%Y%m%d%H%M%S")
        
        version = f"{ticker}_linear_{forecast_days}day_{last_train_time_str}"
        model_path = self.model_dir / f"{version}.joblib"
        return version, model_path

    def get_all_models(self) -> dict[str, Any]:
        """Retrieve all models stored in the model directory."""
        models = []
        
        for model_file in self.model_dir.glob("*.joblib"):
            try:
                model_data = joblib.load(model_file)
                meta = model_data["metadata"]
                
                models.append({
                    "ticker": meta["ticker"],
                    "model_type": meta["model_type"],
                    "version": model_file.stem,
                    "metrics": {
                        "mae": meta["mae"],
                        "r2": meta["r2"],
                        "mape": meta["mape"],
                        "max_ae": meta["max_ae"],
                        "features": meta["features"],
                        "coefficients": meta["coefficients"],
                        "intercept": meta["intercept"],
                        "train_samples": meta["train_samples"],
                        "test_samples": meta["test_samples"],
                        "last_train_date": meta["last_train_date"],
                    },
                })
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping corrupt or invalid model file {model_file.name}: {e}")
                continue

        if not models:
            logger.info("No models found in the directory.")
            return {"models": [], "count": 0}

        return {"models": models, "count": len(models)}

    def get_model(self, ticker: str, forecast_days: int, last_trained_time: str) -> dict[str, Any]:
        """Retrieve a specific model using filepath parameters.

        Raises:
            ModelNotFoundError: File does not exist in storage.
            ModelCorruptedError: File exists but fails joblib deserialization or structure reading.
        """
        version, model_path = self._build_model_filename(ticker, forecast_days, last_trained_time)

        if not model_path.exists():
            raise ModelNotFoundError(ticker=ticker, forecast_days=forecast_days)

        try:
            model_data = joblib.load(model_path)
            meta = model_data["metadata"]

            return {
                "ticker": meta["ticker"],
                "model_type": meta["model_type"],
                "version": version,
                "metrics": {
                    "mae": meta["mae"],
                    "r2": meta["r2"],
                    "mape": meta["mape"],
                    "max_ae": meta["max_ae"],
                    "features": meta["features"],
                    "coefficients": meta["coefficients"],
                    "intercept": meta["intercept"],
                    "train_samples": meta["train_samples"],
                    "test_samples": meta["test_samples"],
                    "last_train_date": meta["last_train_date"],
                },
                "model": model_data["model"],
            }
        except (KeyError, TypeError, Exception) as e:
            raise ModelCorruptedError(ticker=ticker, forecast_days=forecast_days) from e

    def delete_model(self, ticker: str, forecast_days: int, last_trained_time: str) -> dict[str, Any]:
        """Delete a specific model using filepath parameters.

        Raises:
            ModelNotFoundError: File does not exist.
        """
        version, model_path = self._build_model_filename(ticker, forecast_days, last_trained_time)

        if not model_path.exists():
            raise ModelNotFoundError(ticker=ticker, forecast_days=forecast_days)

        try:
            model_path.unlink()
            logger.info(f"Successfully deleted model version {version}")
            return {"status": "success", "version": version}
        except OSError as e:
            logger.error(f"IO/OS Error deleting model file {version}: {e}")
            raise