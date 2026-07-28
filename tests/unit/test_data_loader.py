from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from google.cloud.bigquery import QueryJob

from app.config import timezone
from app.exceptions import DataFetchError, InsufficientDataError
from app.ml.data_loader import (
    get_latest_date,
    invalidate_latest_date_cache,
    load_ticker_data,
)

# Mock data for successful queries
MOCK_STOCK_DATA = [
    {
        "date": datetime(2023, 1, i+1).astimezone(timezone),
        "price": 150.0 + i,
        "lag1": 149.5 + i,
        "lag5": 148.0 + i,
        "lag20": 145.0 + i,
        "lag100": 140.0 + i,
        "ma5": 149.0 + i,
        "ma20": 147.0 + i,
        "ma100": 142.0 + i,
    }
    for i in range(10)
]

@pytest.fixture
def mock_bigquery_client():
    with patch("app.ml.data_loader.get_bigquery_client") as mock_client:
        yield mock_client

@pytest.fixture
def mock_query_job():
    job = MagicMock(spec=QueryJob)
    job.to_arrow.return_value = pl.DataFrame(MOCK_STOCK_DATA).to_arrow()
    return job

@pytest.fixture
def mock_empty_query_job():
    job = MagicMock(spec=QueryJob)
    job.to_arrow.return_value = pl.DataFrame().to_arrow()
    return job

class TestLoadTickerData:
    @classmethod
    def setup_class(cls):
        """Desactivate cache for tests using patching"""
        patcher = patch("app.ml.data_loader.cached_ticker_data")
        cls.mock_cache = patcher.start()
        cls.mock_cache.get.return_value = None
        cls.patcher = patcher

    @classmethod
    def teardown_class(cls):
        """Stop patcher after tests"""
        cls.patcher.stop()

    def test_load_valid_data(self, mock_bigquery_client, mock_query_job):
        """Test loading data with valid inputs"""
        mock_bigquery_client.return_value.query.return_value = mock_query_job
        
        df = load_ticker_data("AAPL", "2023-01-01", "2023-01-10")
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10
        assert "price" in df.columns
        assert "ma100" in df.columns

    def test_empty_data_raises_insufficient_data_error(self, mock_bigquery_client, mock_empty_query_job):
        """Test that empty data raises InsufficientDataError"""
        mock_bigquery_client.return_value.query.return_value = mock_empty_query_job
        
        with pytest.raises(InsufficientDataError) as excinfo:
            load_ticker_data("UNKNOWN", "2023-01-01", "2023-01-02")
        
        assert "Insufficient stock data" in str(excinfo.value)
        assert excinfo.value.ticker == "UNKNOWN"

    @pytest.mark.parametrize("ticker", ["", " ", None, "A"*11])
    def test_invalid_ticker(self, ticker, mock_bigquery_client):
        """Test invalid ticker values"""
        with pytest.raises(ValueError):
            load_ticker_data(ticker, "2023-01-01", "2023-01-02")

    @pytest.mark.parametrize("start_date,end_date", [
        ("", "2023-01-02"),  # empty start date
        ("2023-01-01", ""),  # empty end date
        ("invalid-date", "2023-01-02"), # non-date string
        ("2023-01-01", "invalid-date"), # non-date string
        ("2023/01/01", "2023/01/02"), # wrong date format  
        ("01-01-2023", "01-02-2023"), # wrong date format
        ("2023-01-02", "2023-01-01"),  # end before start
        ("2023-13-01", "2023-01-02"),  # invalid month
        ("2023-01-32", "2023-01-02"),  # invalid day
    ])
    def test_invalid_dates(self, start_date, end_date, mock_bigquery_client):
        """Test various invalid date scenarios"""
        with pytest.raises(ValueError):
            load_ticker_data("AAPL", start_date, end_date)

    def test_bigquery_failure(self, mock_bigquery_client):
        """Test BigQuery failure scenario"""
        mock_bigquery_client.return_value.query.side_effect = Exception("BigQuery connection timeout")
        
        with pytest.raises(DataFetchError) as excinfo:
            load_ticker_data("AAPL", "2023-01-01", "2023-01-02")
        
        assert "Failed to fetch stock data" in str(excinfo.value)
        assert "BigQuery connection timeout" in str(excinfo.value)

class TestGetLatestDate:    
    def test_get_latest_date_success(self, mock_bigquery_client):
        """Test successful retrieval of latest date"""
        mock_result = MagicMock()
        mock_result.to_dataframe.return_value = pl.DataFrame({
            "latest_date": [datetime(2023, 1, 2).astimezone(timezone)]
        }).to_pandas()
        
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_result
        mock_client = mock_bigquery_client.return_value
        mock_client.query.return_value = mock_query_job
        
        result = get_latest_date("AAPL")
        assert result == "2023-01-02"

    def test_get_latest_date_no_data(self, mock_bigquery_client):
        """Test when no data exists for ticker"""
        mock_result = MagicMock()
        mock_result.to_dataframe.return_value = pl.DataFrame({
            "latest_date": []
        }).to_pandas()
        
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_result
        mock_client = mock_bigquery_client.return_value
        mock_client.query.return_value = mock_query_job

        with pytest.raises(InsufficientDataError) as excinfo:
            get_latest_date("UNKNOWN")
        
        assert excinfo.value.ticker == "UNKNOWN"

    @pytest.mark.parametrize("ticker", ["", " ", None, "A"*11])
    def test_invalid_ticker(self, ticker, mock_bigquery_client):
        """Test invalid ticker values"""
        with pytest.raises(ValueError):
            get_latest_date(ticker)

    def test_bigquery_failure(self, mock_bigquery_client):
        """Test BigQuery failure scenario"""
        ticker = "AAPL"
        invalidate_latest_date_cache(ticker)

        mock_bigquery_client.return_value.query.side_effect = Exception("BigQuery connection timeout")

        with pytest.raises(DataFetchError) as excinfo:
            get_latest_date(ticker)
        
        assert "Failed to fetch stock data" in str(excinfo.value)
        assert excinfo.value.ticker == ticker