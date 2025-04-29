"""Unit tests for the data dashboard components.

This test suite validates the core functionality of the DataDashboard class,
including data validation, download capabilities, and date handling.
"""

import pytest
from datetime import date, timedelta
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
from pathlib import Path
import logging

from stocktrader.data.data_dashboard import DataDashboard
from stocktrader.data.config import DashboardConfig
from stocktrader.data.data_validator import DataValidator
from stocktrader.data.exceptions import ValidationError, DownloadError

# Test Constants
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
TEST_DATE_RANGE = pd.date_range('2023-01-01', '2023-01-10')
EXPECTED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

@pytest.fixture(scope="function")
def dashboard():
    """Create a fresh dashboard instance for each test.
    
    Returns:
        DataDashboard: Configured dashboard instance with test settings
    """
    dashboard = DataDashboard()
    # Ensure test directories exist and are clean
    dashboard.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    dashboard.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    yield dashboard
    # Cleanup after tests
    for path in dashboard.saved_paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

@pytest.fixture
def sample_data():
    """Create deterministic sample OHLCV data for testing.
    
    Returns:
        pd.DataFrame: Sample stock data with standard columns
    """
    np.random.seed(42)  # Ensure reproducible test data
    return pd.DataFrame({
        'Open': np.random.randn(len(TEST_DATE_RANGE)),
        'High': np.random.randn(len(TEST_DATE_RANGE)),
        'Low': np.random.randn(len(TEST_DATE_RANGE)),
        'Close': np.random.randn(len(TEST_DATE_RANGE)),
        'Volume': np.random.randint(1000, 10000, len(TEST_DATE_RANGE))
    }, index=TEST_DATE_RANGE)

class TestDashboardInitialization:
    """Test suite for dashboard initialization and configuration."""
    
    def test_initialization(self, dashboard):
        """Verify proper dashboard initialization and default values."""
        assert isinstance(dashboard.config, DashboardConfig)
        assert isinstance(dashboard.validator, DataValidator)
        assert isinstance(dashboard.saved_paths, list)
        assert dashboard.interval == "1d"
        assert dashboard.clean_old is True
        assert dashboard.auto_refresh is False

    def test_directory_setup(self, dashboard):
        """Verify required directories are created."""
        assert dashboard.config.DATA_DIR.exists()
        assert dashboard.config.MODEL_DIR.exists()

class TestSymbolValidation:
    """Test suite for stock symbol validation logic."""

    @pytest.mark.parametrize("symbols,expected_count", [
        ("AAPL, MSFT, GOOGL", 3),
        ("aapl,msft", 2),
        ("", 0),
        ("   ", 0),
    ])
    def test_valid_symbols(self, dashboard, symbols, expected_count):
        """Test validation of various valid symbol formats."""
        validated = dashboard.validator.validate_symbols(symbols)
        assert len(validated) == expected_count
        if validated:
            assert all(s.isalpha() for s in validated)

    @pytest.mark.parametrize("invalid_input", [
        "AAP@L, MS&FT, 123",
        "AAPL!!",
        "123,456",
        "A" * 10  # Too long
    ])
    def test_invalid_symbols(self, dashboard, invalid_input):
        """Test rejection of invalid symbol formats."""
        validated = dashboard.validator.validate_symbols(invalid_input)
        assert len(validated) == 0

class TestDataDownload:
    """Test suite for stock data downloading functionality."""

    @patch('yfinance.download')
    def test_successful_download(self, mock_download, dashboard, sample_data):
        """Test successful data download and processing."""
        mock_download.return_value = sample_data
        
        df = dashboard._download("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert all(col in df.columns for col in EXPECTED_COLUMNS)
        assert not df.empty
        assert df.index.is_monotonic_increasing

    @patch('yfinance.download')
    def test_download_error_handling(self, mock_download, dashboard):
        """Test handling of various download failure scenarios."""
        error_cases = [
            Exception("API Error"),
            ConnectionError("Network Error"),
            TimeoutError("Request Timeout"),
            ValueError("Invalid Response")
        ]
        
        for error in error_cases:
            mock_download.side_effect = error
            df = dashboard._download("AAPL")
            assert df is None

class TestDateValidation:
    """Test suite for date range validation."""

    def test_valid_date_ranges(self, dashboard):
        """Test acceptance of valid date ranges."""
        today = date.today()
        test_ranges = [
            (today - timedelta(days=30), today),
            (today - timedelta(days=365), today),
            (today - timedelta(days=1), today)
        ]
        
        for start, end in test_ranges:
            assert dashboard.validator.validate_dates(start, end)

    def test_invalid_date_ranges(self, dashboard):
        """Test rejection of invalid date ranges."""
        today = date.today()
        invalid_ranges = [
            (today, today + timedelta(days=1)),  # Future date
            (today, today - timedelta(days=1)),  # End before start
            (today - timedelta(days=366), today)  # Too old
        ]
        
        for start, end in invalid_ranges:
            assert not dashboard.validator.validate_dates(start, end)