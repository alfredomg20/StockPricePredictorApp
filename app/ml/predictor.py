from datetime import date, timedelta
from typing import Union
import joblib
import polars as pl
from pathlib import Path
from app.config import logger
from app.ml.data_loader import load_ticker_data, get_latest_date
from app.utils.validation_utils import validate_ticker, validate_forecast_days
from app.utils.time_utils import get_next_business_days

class StockPricePredictor:
    """Handle stock price predictions using trained models."""
    def __init__(self, model_dir: str = "app/models"):
        self.model_dir = Path(model_dir)
    
    def _validate_prediction_inputs(self, ticker: str, forecast_days: int) -> None:
        """
        Validate inputs for prediction.
        
        Args:
            ticker: Stock ticker symbol
            forecast_days: Number of days to forecast
            
        Raises:
            ValueError: If inputs are invalid
        """
        validate_ticker(ticker)
        validate_forecast_days(forecast_days)
    
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
    
    def generate_features_for_prediction(
        self, 
        df: pl.DataFrame,
        latest_data: pl.DataFrame,
        forecast_days: int = 1
    ) -> pl.DataFrame:
        """
        Generate features for prediction based on the latest available data.
        
        Args:
            df: Historical DataFrame with price data
            latest_data: DataFrame containing the date(s) for prediction
            forecast_days: Days ahead to predict
            
        Returns:
            DataFrame with features ready for prediction
        """
        # Ensure we have the historical data with price column
        if "price" not in df.columns:
            raise ValueError("Historical DataFrame must contain 'price' column")
            
        # Get the required features based on forecast horizon
        required_features = self._select_features(forecast_days)
        
        # Get the most recent date from the historical data
        latest_historical_date = df["date"].max()
        
        # Extract the most recent prices needed for feature calculation
        latest_prices = df.filter(
            pl.col("date") <= latest_historical_date
        ).sort("date", descending=False).tail(100)
        
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
            latest_data = latest_data.with_columns(
                pl.lit(ma5_value).alias("ma5")
            )
        if "ma20" in required_features:
            ma20_value = latest_prices["price"].tail(20).mean()
            latest_data = latest_data.with_columns(
                pl.lit(ma20_value).alias("ma20")
            )
        if "ma100" in required_features:
            ma100_value = latest_prices["price"].tail(100).mean()
            latest_data = latest_data.with_columns(
                pl.lit(ma100_value).alias("ma100")
            )
        
        # Select only the required features
        prediction_features = latest_data.select(required_features)
        
        return prediction_features
    
    def predict_price(
        self,
        model_path: str,
        historical_data: pl.DataFrame,
        prediction_date: str
    ) -> dict[str, any]:
        """
        Predict stock price using a trained model for a single date.
        
        Args:
            model_path: Path to the saved model
            historical_data: DataFrame with historical price data
            prediction_date: Date for which to predict the price
            
        Returns:
            Dict with prediction result
        """
        try:
            # Load model
            model_data = joblib.load(model_path)
            model = model_data["model"]
            metadata = model_data["metadata"]
            
            # Create a DataFrame with the prediction date
            pred_df = pl.DataFrame({
                "date": [prediction_date]
            })
            
            # Generate features needed for prediction
            forecast_days = metadata["forecast_days"]
            features = self.generate_features_for_prediction(
                historical_data, 
                pred_df,
                forecast_days
            )
            
            # Make prediction
            prediction = model.predict(features.to_numpy())[0]
            
            return {
                "status": "success",
                "ticker": metadata["ticker"],
                "date": prediction_date,
                "predicted_price": float(prediction),
                "forecast_days": metadata["forecast_days"],
                "features_used": metadata["features"]
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def find_latest_model(self, ticker: str, forecast_days: int) -> str:
        """
        Find the latest model file for a given ticker and forecast days.
        
        Args:
            ticker: Stock ticker symbol
            forecast_days: Number of days to forecast
            
        Returns:
            Path to the latest model file
        """
        model_pattern = f"{ticker}_linear_{forecast_days}day_*.joblib"
        model_files = list(self.model_dir.glob(model_pattern))
        
        if not model_files:
            raise FileNotFoundError(f"No model found for {ticker} with {forecast_days} day forecast")
        
        # Sort by date in filename as YYYYMMDD format
        latest_model = sorted(model_files, key=lambda x: x.name.split('_')[-1].split('.')[0], reverse=True)[0]
        return str(latest_model)
    
    def predict_prices(
        self, 
        ticker: str, 
        forecast_days: int = 1,
    ) -> Union[pl.DataFrame, dict[str, any]]:
        """
        Predict stock prices for multiple future days.
        
        Args:
            ticker: Stock ticker symbol
            forecast_days: Number of days to forecast
        Returns:
            DataFrame with dates and predicted prices
        """
        # Validate inputs
        self._validate_prediction_inputs(ticker, forecast_days)
        try:
            # Find the latest model
            model_path = self.find_latest_model(ticker, forecast_days)
            # Load the model to get metadata
            model_data = joblib.load(model_path)
            model = model_data["model"]
            metadata = model_data["metadata"]
            
            # Get the latest date from historical data
            latest_date_str = get_latest_date(ticker)
            latest_date = date.fromisoformat(latest_date_str)
            # Generate future dates
            future_dates = get_next_business_days(latest_date, metadata["forecast_days"])
            # Set start date 6 months before (considering features need at least 100 business days)
            start_date = latest_date - timedelta(days=180)

            # Initialize with historical data for feature generation
            historical_data = load_ticker_data(ticker, start_date, latest_date)
            working_data = historical_data.clone()
            predictions = []
            
            # For each future date, predict price and update working data
            for future_date in future_dates:
                # Create a DataFrame for the prediction date
                pred_df = pl.DataFrame({"date": [future_date]})
                
                # Generate features for prediction
                features = self.generate_features_for_prediction(
                    working_data,
                    pred_df,
                    metadata["forecast_days"]
                )
                
                # Make prediction
                price = float(model.predict(features.to_numpy())[0])
                
                # Store prediction
                predictions.append({
                    "date": future_date,
                    "predicted_price": price
                })
                
                # Add the prediction to working data for next iteration
                new_row = pl.DataFrame({
                    "date": [future_date],
                    "price": [price]
                })
                
                # Add other necessary columns with default values
                for col in working_data.columns:
                    if col not in new_row.columns:
                        if col.startswith("lag") or col.startswith("ma"):
                            new_row = new_row.with_columns(pl.lit(0.0).alias(col))
                
                # Append to working data
                working_data = pl.concat([working_data, new_row])
            
            # Create result DataFrame
            result_df = pl.DataFrame(predictions)
            return result_df
            
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Stock price prediction failed: {str(e)}")
            return {
                "status": "error",
                "ticker": ticker,
                "error": str(e)
            }
        except Exception as e:
            logger.critical(f"Unexpected error during prediction for {ticker}: {str(e)}")
            raise e