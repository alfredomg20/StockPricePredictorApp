from cachetools import TTLCache
import polars as pl
from app.config import CREDENTIALS_DICT, PROJECT_ID, DATASET_ID, STOCKS_TABLE_ID
from app.utils.google_cloud_utils import get_bigquery_client
from app.utils.validation_utils import validate_ticker, validate_date_range

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
    keys_to_delete = [k for k in cached_ticker_data.keys() if k.startswith(f"{ticker}:")]
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

def load_ticker_data(ticker: str, start_date: str, end_date: str) -> pl.DataFrame:
    """
    Loads stock data for train model for selected ticker and date range from BigQuery.
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
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
        FROM `{PROJECT_ID}.{DATASET_ID}.{STOCKS_TABLE_ID}`
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
        client = get_bigquery_client(CREDENTIALS_DICT, PROJECT_ID)
        query_job = client.query(query)
        df = pl.from_arrow(query_job.to_arrow())
    except Exception as e:
        raise RuntimeError(f"Failed to load data for ticker {ticker} from BigQuery: {e}")
    if df.is_empty():
        raise ValueError(f"No data found for ticker {ticker} in the specified date range.")
    # Cache the DataFrame
    cached_ticker_data[cache_key] = df
    return df

def get_latest_date(ticker: str) -> str:
    """
    Retrieves the latest date for a given ticker from BigQuery.
    Args:
        ticker (str): Stock ticker symbol.
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
        FROM `{PROJECT_ID}.{DATASET_ID}.{STOCKS_TABLE_ID}`
        WHERE ticker = '{ticker}'
    """
    # Execute the query and retrieve the latest date
    try:
        client = get_bigquery_client(CREDENTIALS_DICT, PROJECT_ID)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()
        if df.empty:
            raise ValueError("No data found for the specified ticker.")
        latest_date = df['latest_date'].iloc[0]
        latest_date_str = latest_date.strftime('%Y-%m-%d')
        # Cache the latest date
        cached_latest_date[ticker] = latest_date_str
        return latest_date_str
    except ValueError as ve:
        raise ValueError(f"No data found for ticker {ticker}: {ve}")
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve latest date for ticker {ticker}: {e}")
    
