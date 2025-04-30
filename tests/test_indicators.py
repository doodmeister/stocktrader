"""Unit tests for technical indicators module."""
import pytest
import pandas as pd
import numpy as np
from utils.indicators import (
    add_rsi,
    add_macd,
    add_bollinger_bands,
    IndicatorError,
    validate_dataframe
)

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    return pd.DataFrame({
        'close': [10, 11, 12, 11, 10, 9, 10, 11, 12, 13],
        'volume': [1000] * 10
    })

def test_validate_dataframe():
    """Test DataFrame validation function."""
    valid_df = pd.DataFrame({'close': [1, 2, 3]})
    
    with pytest.raises(TypeError):
        validate_dataframe("not a dataframe", ['close'])
        
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame(), ['close'])
        
    with pytest.raises(ValueError):
        validate_dataframe(valid_df, ['missing_column'])
        
    df_with_nan = pd.DataFrame({'close': [1, np.nan, 3]})
    with pytest.raises(ValueError):
        validate_dataframe(df_with_nan, ['close'])

def test_rsi_calculation(sample_data):
    """Test RSI indicator calculation."""
    df = add_rsi(sample_data)
    
    assert 'rsi' in df.columns
    assert len(df) == len(sample_data)
    assert df['rsi'].between(0, 100).all()
    
    # Test invalid inputs
    with pytest.raises(IndicatorError):
        add_rsi(sample_data, length=-1)
        
    with pytest.raises(IndicatorError):
        add_rsi(pd.DataFrame())

def test_macd_calculation(sample_data):
    """Test MACD indicator calculation."""
    df = add_macd(sample_data)
    
    required_cols = ['macd', 'macd_signal', 'macd_hist']
    assert all(col in df.columns for col in required_cols)
    
    # Test parameter validation
    with pytest.raises(IndicatorError):
        add_macd(sample_data, fast=26, slow=12)  # Invalid fast/slow
        
    with pytest.raises(IndicatorError):
        add_macd(sample_data, signal=0)  # Invalid signal period

def test_bollinger_bands_calculation(sample_data):
    """Test Bollinger Bands calculation."""
    df = add_bollinger_bands(sample_data)
    
    required_cols = ['bb_upper', 'bb_middle', 'bb_lower']
    assert all(col in df.columns for col in required_cols)
    
    # Verify band relationships
    assert (df['bb_upper'] >= df['bb_middle']).all()
    assert (df['bb_middle'] >= df['bb_lower']).all()
    
    # Test invalid parameters
    with pytest.raises(IndicatorError):
        add_bollinger_bands(sample_data, length=0)
        
    with pytest.raises(IndicatorError):
        add_bollinger_bands(sample_data, std=-1)