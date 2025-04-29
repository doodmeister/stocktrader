"""Unit tests for the data dashboard components."""

import pytest
from datetime import date, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from stocktrader.data.data_dashboard import DataDashboard
from stocktrader.data.config import DashboardConfig
from stocktrader.data.data_validator import DataValidator

@pytest.fixture
def dashboard():
    """Create a dashboard instance for testing."""
    return DataDashboard()

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', '2023-01-10')
    return pd.DataFrame({
        'Open': np.random.randn(len(dates)),
        'High': np.random.randn(len(dates)),
        'Low': np.random.randn(len(dates)),
        'Close': np.random.randn(len(dates)),
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

def test_initialization(dashboard):
    """Test dashboard initialization."""
    assert isinstance(dashboard.config, DashboardConfig)
    assert isinstance(dashboard.validator, DataValidator)
    assert dashboard.saved_paths == []
    assert dashboard.interval == "1d"

def test_symbol_validation(dashboard):
    """Test symbol validation logic."""
    valid_symbols = "AAPL, MSFT, GOOGL"
    invalid_symbols = "AAP@L, MS&FT, 123"
    
    # Test valid symbols
    validated = dashboard.validator.validate_symbols(valid_symbols)
    assert len(validated) == 3
    assert all(s.isalpha() for s in validated)
    
    # Test invalid symbols
    validated = dashboard.validator.validate_symbols(invalid_symbols)
    assert len(validated) == 0

@patch('yfinance.download')
def test_data_download(mock_download, dashboard, sample_data):
    """Test stock data downloading."""
    mock_download.return_value = sample_data
    
    df = dashboard._download("AAPL")
    assert df is not None
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Test error handling
    mock_download.side_effect = Exception("API Error")
    df = dashboard._download("AAPL")
    assert df is None

def test_date_validation(dashboard):
    """Test date range validation."""
    today = date.today()
    
    # Valid date range
    valid_start = today - timedelta(days=30)
    assert dashboard.validator.validate_dates(valid_start, today)
    
    # Invalid date range (future date)
    future_date = today + timedelta(days=1)
    assert not dashboard.validator.validate_dates(today, future_date)
    
    # Invalid date range (start after end)
    assert not dashboard.validator.validate_dates(today, valid_start)