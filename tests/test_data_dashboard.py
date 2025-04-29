"""Unit tests for the refactored data dashboard."""

import pytest
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from stocktrader.data.data_dashboard import (
    DataDashboard, 
    DataValidator,
    DataFetcher,
    ModelManager
)

@pytest.fixture
def dashboard():
    """Create dashboard instance for testing."""
    return DataDashboard()

@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    dates = pd.date_range('2023-01-01', '2023-01-10')
    return pd.DataFrame({
        'Open': np.random.randn(len(dates)),
        'High': np.random.randn(len(dates)),
        'Low': np.random.randn(len(dates)), 
        'Close': np.random.randn(len(dates)),
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

def test_date_validation():
    """Test date validation logic."""
    validator = DataValidator()
    
    # Valid dates
    valid, _ = validator.validate_dates(
        date(2023,1,1),
        date(2023,12,31),
        "1d"
    )
    assert valid is True
    
    # Invalid dates
    valid, msg = validator.validate_dates(
        date(2023,12,31),
        date(2023,1,1),
        "1d"  
    )
    assert valid is False
    assert "before end date" in msg

# Additional tests...