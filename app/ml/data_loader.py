import polars as pl
from cachetools import TTLCache

from app.exceptions import DataFetchError, InsufficientDataError
from app.schemas.config import GCloudConfigSchema
from app.utils.google_cloud_utils import get_bigquery_client
from app.utils.validation_utils import validate_date_range, validate_ticker

# Configure caching
time_to_live = 3600  # 1 hour lifetime
cached_ticker_data = TTLCache(maxsize=1000, ttl=time_to_live)
cached_latest_date = TTLCache(maxsize=100, ttl=time_to_live)

def invalidate_ticker_data_cache(ticker: str):
    """
    Deletes all cached data for a specific ticker.
    Args:
        ticker (str): Stock ticker symbol.
    """
    keys_to_delete = [k for k in cached_ticker_data if k.startswith(f"{ticker}:")]
    for k in keys_to_delete:
        del cached_ticker_data[k]

def invalidate_latest_date_cache(ticker: str):
    """
    Deletes the cached latest date for a specific ticker.
    Args:
        ticker (str): Stock ticker symbol.
    """
    if ticker in cached_latest_date:
        del cached_latest_date[ticker]

def load_ticker_data(ticker: str, start_date: str, end_date: str, gcloud_config: GCloudConfigSchema) -> pl.DataFrame:
    """
    Loads stock data for train model for selected ticker and date range from BigQuery.
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        gcloud_config (GCloudConfigSchema): Configuration dictionary for Google Cloud.
    Returns:
        polars.DataFrame: DataFrame containing stock data with features for model training.
    """
    # Return cached data if available
    cache_key = f"{ticker}:{start_date}:{end_date}"
    if cache_key in cached_ticker_data:
        return cached_ticker_data[cache_key]
    
    # Validate inputs
    validate_ticker(ticker)
    validate_date_range(start_date, end_date)
    
    # Construct the BigQuery SQL query
    query = f"""
        WITH base_data AS (
        SELECT 
            DATE(date) AS date,
            close,
            LAG(close, 1) OVER (ORDER BY date) AS lag1,
            LAG(close, 5) OVER (ORDER BY date) AS lag5,
            LAG(close, 20) OVER (ORDER BY date) AS lag20,
            LAG(close, 100) OVER (ORDER BY date) AS lag100,
            AVG(close) OVER (ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS ma5,
            AVG(close) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20,
            AVG(close) OVER (ORDER BY date ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) AS ma100
        FROM `{gcloud_config.project_id}.{gcloud_config.dataset_id}.{gcloud_config.stocks_table_id}`
        WHERE ticker = '{ticker}'
            AND date >= '{start_date}'
            AND date <= '{end_date}'
        )

        SELECT
        date,
        close AS price,  -- Target variable (next day's price)
        lag1,
        lag5,
        lag20,
        lag100,
        ma5,
        ma20,
        ma100
        FROM base_data
        WHERE 
        lag100 IS NOT NULL  -- Ensures all features have values
        AND ma100 IS NOT NULL
        ORDER BY date
    """
    # Execute the query and load data into a Polars DataFrame
    try:
        client = get_bigquery_client(gcloud_config.credentials.dict(), gcloud_config.project_id)
        query_job = client.query(query)
        df = pl.from_arrow(query_job.to_arrow())
    except Exception as e:
        raise DataFetchError(ticker=ticker, original_error=str(e)) from e

    if df.is_empty() or df.height < 10:
        raise InsufficientDataError(ticker=ticker, required_samples=10, actual_samples=df.height if not df.is_empty() else 0)

    cached_ticker_data[cache_key] = df
    return df

def get_latest_date(ticker: str, gcloud_config: GCloudConfigSchema) -> str:
    """
    Retrieves the latest date for a given ticker from BigQuery.
    Args:
        ticker (str): Stock ticker symbol.
        gcloud_config (GCloudConfigSchema): Configuration dictionary for Google Cloud.
    Returns:
        str: Latest date in 'YYYY-MM-DD' format.
    """
    # Return cached latest date if available
    if ticker in cached_latest_date:
        return cached_latest_date[ticker]
    
    # Validate input
    validate_ticker(ticker)

    # Construct the BigQuery SQL query
    query = f"""
        SELECT MAX(date) AS latest_date
        FROM `{gcloud_config.project_id}.{gcloud_config.dataset_id}.{gcloud_config.stocks_table_id}`
        WHERE ticker = '{ticker}'
    """
    # Execute the query and retrieve the latest date
    try:
        client = get_bigquery_client(gcloud_config.credentials.dict(), gcloud_config.project_id)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()
    except Exception as e:
        raise DataFetchError(ticker=ticker, original_error=str(e)) from e
    
    if df.empty or df['latest_date'].iloc[0] is None:
        raise InsufficientDataError(ticker=ticker, required_samples=1, actual_samples=0)
    latest_date = df['latest_date'].iloc[0]
    latest_date_str = latest_date.strftime('%Y-%m-%d')

    cached_latest_date[ticker] = latest_date_str
    return latest_date_str
