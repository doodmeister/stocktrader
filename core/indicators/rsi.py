\
"""
core/indicators/rsi.py

Relative Strength Index (RSI) calculation.
"""
import numpy as np
import pandas as pd
from pandas import DataFrame # Added for type hint consistency
from utils.logger import setup_logger
from .base import IndicatorError, validate_input

logger = setup_logger(__name__)

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("pandas_ta not found - using fallback for RSI calculations")

@validate_input(['close'])
def calculate_rsi(df: DataFrame, length: int = 14, close_col: str = 'close') -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        df: DataFrame with price data
        length: Period for RSI calculation
        close_col: Name of close price column
        
    Returns:
        pd.Series: RSI values (0-100)
    """
    try:
        if length < 1:
            raise ValueError("Length must be positive")
            
        if TA_AVAILABLE:
            rsi_series = ta.rsi(df[close_col], length=length)
            if rsi_series is None:
                raise ValueError("pandas_ta.rsi returned None")
            # If rsi is a DataFrame, extract the first column as Series
            if isinstance(rsi_series, pd.DataFrame):
                if rsi_series.shape[1] == 0:
                    raise ValueError("pandas_ta.rsi returned empty DataFrame")
                rsi_series = rsi_series.iloc[:, 0]
        else:
            delta = df[close_col].diff()
            gain = delta.clip(lower=0).rolling(window=length, min_periods=1).mean()
            loss = -delta.clip(upper=0).rolling(window=length, min_periods=1).mean()
            rs = gain / np.where(loss == 0, np.inf, loss) # type: ignore
            rsi_series = 100 - (100 / (1 + rs))
        
        return pd.Series(rsi_series, index=df.index).clip(0, 100)
    except Exception as e:
        logger.error(f"RSI calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate RSI: {e}") from e
