from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import polars as pl

from app.config import MODELS_DIR
from app.exceptions import ModelCorruptedError, ModelNotFoundError
from app.ml.data_loader import get_latest_date, load_ticker_data
from app.utils.time_utils import get_next_business_days
from app.utils.validation_utils import validate_forecast_days, validate_ticker


class StockPricePredictor:
    """Handle stock price predictions using trained models."""

    def __init__(self, model_dir: str | Path = MODELS_DIR):
        self.model_dir = Path(model_dir)

    def _validate_prediction_inputs(self, ticker: str, forecast_days: int) -> None:
        """Validate inputs for prediction."""
        validate_ticker(ticker)
        validate_forecast_days(forecast_days)

    def _load_model_safely(self, model_path: Path | str, ticker: str, forecast_days: int) -> dict[str, Any]:
        """Helper to safely load model files and handle corruption errors."""
        path = Path(model_path)
        if not path.exists():
            raise ModelNotFoundError(ticker=ticker, forecast_days=forecast_days)

        try:
            model_data = joblib.load(path)
            _ = model_data["model"]
            _ = model_data["metadata"]
            return model_data
        except (KeyError, TypeError, Exception) as e:
            raise ModelCorruptedError(ticker=ticker, forecast_days=forecast_days) from e

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

    def generate_features_for_prediction(
        self, df: pl.DataFrame, latest_data: pl.DataFrame, forecast_days: int = 1
    ) -> pl.DataFrame:
        """Generate features for prediction based on the latest available data."""
        if "price" not in df.columns:
            raise ValueError("Historical DataFrame must contain 'price' column")

        required_features = self._select_features(forecast_days)
        latest_historical_date = df["date"].max()

        latest_prices = (
            df.filter(pl.col("date") <= latest_historical_date)
            .sort("date", descending=False)
            .tail(100)
        )

        # Calculate lag features
        if "lag1" in required_features:
            latest_data = latest_data.with_columns(
                pl.lit(latest_prices["price"].tail(1).item()).alias("lag1")
            )
        if "lag5" in required_features:
            latest_data = latest_data.with_columns(
                pl.lit(latest_prices["price"].tail(5).head(1).item()).alias("lag5")
            )
        if "lag20" in required_features:
            latest_data = latest_data.with_columns(
                pl.lit(latest_prices["price"].tail(20).head(1).item()).alias("lag20")
            )
        if "lag100" in required_features:
            latest_data = latest_data.with_columns(
                pl.lit(latest_prices["price"].tail(100).head(1).item()).alias("lag100")
            )

        # Calculate moving average features
        if "ma5" in required_features:
            ma5_value = latest_prices["price"].tail(5).mean()
            latest_data = latest_data.with_columns(pl.lit(ma5_value).alias("ma5"))
        if "ma20" in required_features:
            ma20_value = latest_prices["price"].tail(20).mean()
            latest_data = latest_data.with_columns(pl.lit(ma20_value).alias("ma20"))
        if "ma100" in required_features:
            ma100_value = latest_prices["price"].tail(100).mean()
            latest_data = latest_data.with_columns(pl.lit(ma100_value).alias("ma100"))

        return latest_data.select(required_features)

    def find_latest_model(self, ticker: str, forecast_days: int) -> str:
        """Find the latest model file for a given ticker and forecast days."""
        model_pattern = f"{ticker}_linear_{forecast_days}day_*.joblib"
        model_files = list(self.model_dir.glob(model_pattern))

        if not model_files:
            raise ModelNotFoundError(ticker=ticker, forecast_days=forecast_days)

        latest_model = max(model_files, key=lambda x: x.name.split("_")[-1].split(".")[0])
        return str(latest_model)

    def predict_price(
        self, model_path: str, historical_data: pl.DataFrame, prediction_date: str
    ) -> dict[str, Any]:
        """Predict stock price using a trained model for a single date."""
        temp_path = Path(model_path)
        ticker_guess = temp_path.name.split("_")[0] if "_" in temp_path.name else "UNKNOWN"
        
        model_data = self._load_model_safely(model_path, ticker=ticker_guess, forecast_days=1)
        model = model_data["model"]
        metadata = model_data["metadata"]

        pred_df = pl.DataFrame({"date": [prediction_date]})
        forecast_days = metadata.get("forecast_days", 1)

        features = self.generate_features_for_prediction(
            historical_data, pred_df, forecast_days
        )

        prediction = model.predict(features.to_numpy())[0]

        return {
            "status": "success",
            "ticker": metadata["ticker"],
            "date": prediction_date,
            "predicted_price": float(prediction),
            "forecast_days": forecast_days,
            "features_used": metadata["features"],
        }

    def predict_prices(
        self,
        ticker: str,
        forecast_days: int = 1,
    ) -> pl.DataFrame:
        """Predict stock prices for multiple future days."""
        self._validate_prediction_inputs(ticker, forecast_days)

        # Encuentra el modelo más reciente y lo carga de forma segura
        model_path = self.find_latest_model(ticker, forecast_days)
        model_data = self._load_model_safely(model_path, ticker=ticker, forecast_days=forecast_days)
        
        model = model_data["model"]
        metadata = model_data["metadata"]

        latest_date_str = get_latest_date(ticker)
        latest_date = date.fromisoformat(latest_date_str)
        
        future_dates = get_next_business_days(latest_date, metadata["forecast_days"])
        start_date = latest_date - timedelta(days=180)

        # Carga datos históricos (lanzará DataFetchError o InsufficientDataError si algo falla)
        historical_data = load_ticker_data(ticker, start_date.strftime("%Y-%m-%d"), latest_date_str)
        working_data = historical_data.clone()
        predictions = []

        for future_date in future_dates:
            pred_df = pl.DataFrame({"date": [future_date]})
            features = self.generate_features_for_prediction(
                working_data, pred_df, metadata["forecast_days"]
            )

            price = float(model.predict(features.to_numpy())[0])

            predictions.append({"date": future_date, "predicted_price": price})

            new_row = pl.DataFrame({"date": [future_date], "price": [price]})

            for col in working_data.columns:
                if col not in new_row.columns and col.startswith(("lag", "ma")):
                    new_row = new_row.with_columns(pl.lit(0.0).alias(col))

            working_data = pl.concat([working_data, new_row])

        return pl.DataFrame(predictions)