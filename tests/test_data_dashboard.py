"""Unit tests for the data dashboard module."""

import pytest
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from stocktrader.data.data_dashboard import DataDashboard

@pytest.fixture
def dashboard():
    """Create a dashboard instance for testing."""
    return DataDashboard()

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    data = {
        'Open': np.random.randn(len(dates)),
        'High': np.random.randn(len(dates)),
        'Low': np.random.randn(len(dates)),
        'Close': np.random.randn(len(dates)),
        'Volume': np.random.randint(1000, 10000, len(dates))
    }
    return pd.DataFrame(data, index=dates)

def test_validate_dates():
    """Test date validation logic."""
    dashboard = DataDashboard()
    
    # Valid date range
    start = date(2023, 1, 1)
    end = date(2023, 12, 31)
    valid, _ = dashboard.validate_dates(start, end, "1d")
    assert valid is True
    
    # Invalid date range
    start = date(2023, 12, 31)
    end = date(2023, 1, 1)
    valid, message = dashboard.validate_dates(start, end, "1d")
    assert valid is False
    assert "Start date must be before end date" in message

    # Future end date
    start = date.today()
    end = date.today() + timedelta(days=1)
    valid, message = dashboard.validate_dates(start, end, "1d")
    assert valid is False
    assert "End date cannot be in the future" in message

def test_clean_data_directory(dashboard, tmp_path):
    """Test data directory cleaning."""
    # Create test files
    test_file = tmp_path / "test.csv"
    test_file.write_text("test")
    
    with patch('pathlib.Path.glob') as mock_glob:
        mock_glob.return_value = [test_file]
        dashboard.data_dir = tmp_path
        dashboard.clean_data_directory()
        
    assert not test_file.exists()

@patch('yfinance.download')
def test_fetch_ohlcv(mock_download, dashboard, sample_ohlcv_data):
    """Test OHLCV data fetching."""
    mock_download.return_value = sample_ohlcv_data
    
    df = dashboard.fetch_ohlcv(
        "AAPL",
        "2023-01-01",
        "2023-01-10",
        "1d"
    )
    
    assert df is not None
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    assert len(df) == len(sample_ohlcv_data)

@patch('yfinance.Ticker')
def test_is_valid_symbol(mock_ticker, dashboard):
    """Test stock symbol validation."""
    mock_info = {'regularMarketPrice': 100.0}
    mock_ticker.return_value = Mock(info=mock_info)
    
    assert dashboard.is_valid_symbol("AAPL") is True
    
    mock_ticker.return_value = Mock(info={})
    assert dashboard.is_valid_symbol("INVALID") is False

def test_save_data(dashboard, sample_ohlcv_data, tmp_path):
    """Test data saving functionality."""
    dashboard.data_dir = tmp_path
    
    path = dashboard.save_data(sample_ohlcv_data, "AAPL", "1d")
    assert path is not None
    assert path.exists()
    assert path.suffix == ".csv"
    
    # Verify saved data
    saved_df = pd.read_csv(path, index_col=0)
    pd.testing.assert_frame_equal(
        saved_df,
        sample_ohlcv_data,
        check_dtype=False
    )