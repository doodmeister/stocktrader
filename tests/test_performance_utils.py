"""Unit tests for performance utilities."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from performance_utils import PatternDetector, DataManager, DashboardState

@pytest.fixture
def sample_dataframe():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=5),
        periods=100,
        freq='5min'
    )
    
    return pd.DataFrame({
        'open': np.random.uniform(100, 150, 100),
        'high': np.random.uniform(120, 170, 100),
        'low': np.random.uniform(90, 130, 100),
        'close': np.random.uniform(100, 150, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    }, index=dates)

def test_pattern_detector(sample_dataframe):
    """Test pattern detection functionality."""
    patterns = PatternDetector.detect_patterns(sample_dataframe)
    assert isinstance(patterns, list)
    
    # Test with empty dataframe
    empty_patterns = PatternDetector.detect_patterns(pd.DataFrame())
    assert empty_patterns == []
    
    # Test with single row
    single_row = PatternDetector.detect_patterns(sample_dataframe.iloc[[0]])
    assert single_row == []

def test_dashboard_state():
    """Test dashboard state initialization."""
    state = DashboardState()
    assert hasattr(st.session_state, 'data')
    assert hasattr(st.session_state, 'model')
    assert hasattr(st.session_state, 'symbols')
    assert isinstance(st.session_state.symbols, list)
    assert len(st.session_state.class_names) == 6