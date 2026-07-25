import joblib
from datetime import datetime
from pathlib import Path
from app.config import logger
from app.utils.validation_utils import validate_datetime, validate_forecast_days, validate_ticker

class ModelStore:
    """Manage storage and retrieval of trained models."""
    def __init__(self, model_dir: Path | str):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _validate_model_params(self, ticker: str, forecast_days: int, last_trained_time: str) -> None:
        """
        Validate model parameters for retrieval or deletion.
        
        Args:
            ticker (str): Stock ticker symbol.
            forecast_days (int): Number of days to forecast.
            last_trained_time (str): Last trained time in 'YYYY-MM-DD-HH:MM:SS' format.
        
        Raises:
            ValueError: If any parameter is invalid.
        """
        validate_ticker(ticker)
        validate_forecast_days(forecast_days)
        validate_datetime(last_trained_time)

    def get_all_models(self) -> dict[str, any]:
        """
        Retrieve all models stored in the model directory.

        Returns:
            dict: A dictionary containing a list of models and the count of models.
        """
        try:
            models = []
            for model_file in self.model_dir.glob("*.joblib"):
                try:
                    model_data = joblib.load(model_file)
                    models.append({
                        "ticker": model_data["metadata"]["ticker"],
                        "model_type": model_data["metadata"]["model_type"],
                        "version": model_file.stem,
                        "metrics": {
                            "mae": model_data["metadata"]["mae"],
                            "r2": model_data["metadata"]["r2"],
                            "mape": model_data["metadata"]["mape"],
                            "max_ae": model_data["metadata"]["max_ae"],
                            "features": model_data["metadata"]["features"],
                            "coefficients": model_data["metadata"]["coefficients"],
                            "intercept": model_data["metadata"]["intercept"],
                            "train_samples": model_data["metadata"]["train_samples"],
                            "test_samples": model_data["metadata"]["test_samples"],
                            "last_train_date": model_data["metadata"]["last_train_date"]
                        }
                    })
                except Exception as e:
                    logger.warning(f"Corrupt model file {model_file}: {e}")
                    continue
        
            if not models:
                logger.info("No models found in the directory")
                return {"models": [], "count": 0}
                    
            return {
                "models": models,
                "count": len(models)
            }
        except Exception as e:
            logger.error(f"Failed listing models: {e}")
            raise e
    
    def get_model(self, ticker: str, forecast_days: int, last_trained_time: str) -> dict[str, any]:
        """
        Retrieve a specific model using filepath parameters.

        Args:
            ticker (str): Stock ticker symbol.
            forecast_days (int): Number of days to forecast.
            last_trained_time (str): Last trained time in 'YYYY-MM-DD-HH:MM:SS' format.
        
        Returns:
            dict: A dictionary containing model metadata and the model itself.
        """
        # Validate inputs
        self._validate_model_params(ticker, forecast_days, last_trained_time)

        # Get model path using parameters
        last_train_time_str = datetime.strptime(last_trained_time, "%Y-%m-%d-%H:%M:%S").strftime("%Y%m%d%H%M%S")
        version = f"{ticker}_linear_{forecast_days}day_{last_train_time_str}"
        model_path = self.model_dir / f"{version}.joblib"

        if not model_path.exists():
            logger.error(f"Model version {version} not found")
            raise FileNotFoundError(f"Model version {version} does not exist.")
        try:
            model_data = joblib.load(model_path)
            return {
                "ticker": model_data["metadata"]["ticker"],
                "model_type": model_data["metadata"]["model_type"],
                "version": version,
                "metrics": {
                    "mae": model_data["metadata"]["mae"],
                    "r2": model_data["metadata"]["r2"],
                    "mape": model_data["metadata"]["mape"],
                    "max_ae": model_data["metadata"]["max_ae"],
                    "features": model_data["metadata"]["features"],
                    "coefficients": model_data["metadata"]["coefficients"],
                    "intercept": model_data["metadata"]["intercept"],
                    "train_samples": model_data["metadata"]["train_samples"],
                    "test_samples": model_data["metadata"]["test_samples"],
                    "last_train_date": model_data["metadata"]["last_train_date"]
                },
                "model": model_data["model"]
            }
        except Exception as e:
            logger.error(f"Failed to load model {version}: {str(e)}")
            raise e
        
    def delete_model(self, ticker: str, forecast_days: int, last_trained_time: str) -> dict[str, any]:
        """
        Delete a specific model using filepath parameters.

        Args:
            ticker (str): Stock ticker symbol.
            forecast_days (int): Number of days to forecast.
            last_trained_time (str): Last trained time in 'YYYY-MM-DD-HH:MM:SS' format.
        
        Returns:
            dict: A dictionary containing the status of the deletion and the version of the deleted model.
        """
        # Validate inputs
        self._validate_model_params(ticker, forecast_days, last_trained_time)

        # Get model path using parameters
        last_train_time_str = datetime.strptime(last_trained_time, "%Y-%m-%d-%H:%M:%S").strftime("%Y%m%d%H%M%S")
        version = f"{ticker}_linear_{forecast_days}day_{last_train_time_str}"
        model_path = self.model_dir / f"{version}.joblib"
        
        if not model_path.exists():
            logger.error(f"Model version {version} not found for deletion")
            raise FileNotFoundError(f"Model version {version} does not exist.")
        try:
            model_path.unlink()
            logger.info(f"Deleted model: {version}")
            return {"status": "success", "version": version}
        except Exception as e:
            logger.error(f"Failed to delete {version}: {str(e)}")
            raise e

