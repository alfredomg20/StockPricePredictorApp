import os
from datetime import date, datetime
import polars as pl

def validate_env_variables(required_vars: list[str]) -> list[str]:
    """
    Validates that all required environment variables are set.

    Args:
        required_vars (list[str]): List of required environment variable names.
    
    Returns:
        list[str]: List of missing environment variable names.
    """
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    return missing_vars

def validate_ticker(ticker: str) -> None:
    """
    Validates the ticker symbol.
    
    Args:
        ticker (str): Stock ticker symbol.
    
    Raises:
        ValueError: If the ticker is not a valid string or does not meet the criteria.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker must be a non-empty string.")
    
    if len(ticker) > 10:
        raise ValueError("Ticker symbol is too long. Maximum length is 10 characters.")
    
def validate_date(date_value: str | date) -> None:
    """
    Validates the date is in 'YYYY-MM-DD' format.
    
    Args:
        date_value (str | date): Date string or date object to validate.
    
    Raises:
        ValueError: If the date is invalid or not in the correct format.
    """
    # Instant validation if date is a date object
    if isinstance(date_value, date):
        return
    # Parsing validation if date is in string format
    try:
        datetime.strptime(date_value, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Date must be in 'YYYY-MM-DD' format.")
    
def validate_date_range(start_date: str | date, end_date: str | date) -> None:
    """
    Validates the date range.
    
    Args:
        start_date (str | date): Start date in 'YYYY-MM-DD' format.
        end_date (str | date): End date in 'YYYY-MM-DD' format.
    
    Raises:
        ValueError: If the date range is invalid.
    """
    validate_date(start_date)
    validate_date(end_date)
    
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    if start_date > end_date:
        raise ValueError("Start date must be before or equal to end date.")

def validate_datetime(datetime_value: str | datetime) -> None:
    """
    Validates the datetime is in 'YYYY-MM-DD-HH:MM:SS' format.
    
    Args:
        datetime_value (str | datetime): Datetime string or datetime object to validate.
    
    Raises:
        ValueError: If the datetime is invalid or not in the correct format.
    """
    # Instant validation if datetime is a datetime object
    if isinstance(datetime_value, datetime):
        return
    # Parsing validation if datetime is in string format
    try:
        datetime.strptime(datetime_value, "%Y-%m-%d-%H:%M:%S")
    except ValueError:
        raise ValueError("Datetime must be in 'YYYY-MM-DD HH:MM:SS' format.")    
    
def validate_forecast_days(forecast_days: int) -> None:
    """
    Validates the number of forecast days.
    
    Args:
        forecast_days (int): Number of days to forecast.
    
    Raises:
        ValueError: If the number of forecast days is not a between 1 and 365.
    """
    if not isinstance(forecast_days, int) or forecast_days <= 0:
        raise ValueError("Forecast days must be a positive integer.")
    
    if forecast_days > 365:
        raise ValueError("Forecast days cannot exceed 365 days.")

def validate_dataframe(df: pl.DataFrame):
    """
    Validates that the DataFrame is valid.

    Args:
        df (pl.DataFrame): DataFrame to validate.
    
    Raises:
        ValueError: If the DataFrame is not a Polars DataFrame or is empty.
    """
    if type(df) is not pl.DataFrame:
        raise ValueError("Data must be a Polars DataFrame.")
    if df.is_empty():
        raise ValueError("DataFrame cannot be empty.")

def validate_columns(df: pl.DataFrame, required_columns: list[str]) -> None:
    """
    Validates that the DataFrame contains the required columns.

    Args:
        df (pl.DataFrame): DataFrame to validate.
        required_columns (list[str]): List of required column names.
    
    Raises:
        ValueError: If any required column is missing.
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing_columns)}")