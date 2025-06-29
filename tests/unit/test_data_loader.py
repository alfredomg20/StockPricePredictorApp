import pytest
from unittest.mock import patch, MagicMock
import polars as pl
from datetime import datetime
from app.ml.data_loader import load_ticker_data, get_latest_date, invalidate_latest_date_cache
from google.cloud.bigquery import QueryJob

# Mock data for successful queries
MOCK_STOCK_DATA = [
    {
        "date": datetime(2023, 1, 1),
        "price": 150.0,
        "lag1": 149.5,
        "lag5": 148.0,
        "lag20": 145.0,
        "lag100": 140.0,
        "ma5": 149.0,
        "ma20": 147.0,
        "ma100": 142.0,
    },
    {
        "date": datetime(2023, 1, 2),
        "price": 151.0,
        "lag1": 150.0,
        "lag5": 149.0,
        "lag20": 146.0,
        "lag100": 141.0,
        "ma5": 149.5,
        "ma20": 147.5,
        "ma100": 142.5,
    },
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
        
        df = load_ticker_data("AAPL", "2023-01-01", "2023-01-02")
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert "price" in df.columns
        assert "ma100" in df.columns

    def test_empty_data_raises_value_error(self, mock_bigquery_client, mock_empty_query_job):
        """Test that empty data raises ValueError"""
        mock_bigquery_client.return_value.query.return_value = mock_empty_query_job
        
        with pytest.raises(ValueError) as excinfo:
            load_ticker_data("UNKNOWN", "2023-01-01", "2023-01-02")
        
        assert "No data found" in str(excinfo.value)

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
        mock_bigquery_client.return_value.query.side_effect = Exception("BigQuery error")
        
        with pytest.raises(RuntimeError) as excinfo:
            load_ticker_data("AAPL", "2023-01-01", "2023-01-02")
        
        assert "Failed to load data" in str(excinfo.value)

class TestGetLatestDate:    
    def test_get_latest_date_success(self, mock_bigquery_client):
        """Test successful retrieval of latest date"""
        mock_result = MagicMock()
        mock_result.to_dataframe.return_value = pl.DataFrame({
            "latest_date": [datetime(2023, 1, 2)]
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

        with pytest.raises(ValueError):
            get_latest_date("UNKNOWN")

    @pytest.mark.parametrize("ticker", ["", " ", None, "A"*11])
    def test_invalid_ticker(self, ticker, mock_bigquery_client):
        """Test invalid ticker values"""
        with pytest.raises(ValueError):
            get_latest_date(ticker)

    def test_bigquery_failure(self, mock_bigquery_client):
        """Test BigQuery failure scenario"""
        ticker = "AAPL"
        invalidate_latest_date_cache(ticker)

        mock_bigquery_client.return_value.query.side_effect = Exception("BigQuery error")

        with pytest.raises(RuntimeError) as excinfo:
            get_latest_date(ticker)
        
        assert "Failed to retrieve latest date" in str(excinfo.value)