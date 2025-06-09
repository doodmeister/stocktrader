\
"""
core/indicators/macd.py

Moving Average Convergence Divergence (MACD) calculation.
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
    logger.warning("pandas_ta not found - using fallback for MACD calculations")

@validate_input(['close'])
def calculate_macd(df: DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, 
                   close_col: str = 'close') -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD indicator.
    
    Args:
        df: DataFrame with price data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period
        close_col: Name of close price column
        
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    try:
        # Ensure all periods are valid
        if any(p < 1 for p in (fast, slow, signal)):
            raise ValueError("All periods must be positive integers.")
        if fast >= slow:
            raise ValueError("Fast period must be less than slow period.")

        # If not enough rows, return NaN series
        min_required_rows = slow + signal # pandas_ta might have different internal min_periods
        if len(df) < min_required_rows: # A basic check, pandas_ta might still return NaNs
            logger.warning(f"Insufficient data for MACD. Need at least {min_required_rows} rows, got {len(df)}. Returning NaN series.")
            nan_series = pd.Series([np.nan] * len(df), index=df.index)
            return nan_series, nan_series, nan_series

        if TA_AVAILABLE:
            macd_df = ta.macd(df[close_col], fast=fast, slow=slow, signal=signal)
            if macd_df is None or macd_df.empty:
                raise ValueError("pandas_ta.macd returned None or empty DataFrame")
            
            # Extract columns (pandas_ta returns DataFrame with specific column names)
            macd_col = f'MACD_{fast}_{slow}_{signal}'
            signal_col = f'MACDs_{fast}_{slow}_{signal}'
            hist_col = f'MACDh_{fast}_{slow}_{signal}'
            
            if not all(col in macd_df.columns for col in [macd_col, signal_col, hist_col]):
                raise KeyError(f"Expected MACD columns not found. Available: {macd_df.columns}")
                
            return macd_df[macd_col], macd_df[signal_col], macd_df[hist_col]
        else:
            exp1 = df[close_col].ewm(span=fast, adjust=False, min_periods=fast).mean()
            exp2 = df[close_col].ewm(span=slow, adjust=False, min_periods=slow).mean()
            
            # Ensure EMAs have enough non-NaN values before subtraction
            # This check might be redundant if min_periods in ewm handles it, but good for robustness
            if exp1.isna().all() or exp2.isna().all(): # Check if all are NaN after min_periods
                 logger.warning(f"EMAs for MACD resulted in all NaNs. Check input data length and quality. Fast: {len(exp1.dropna())}, Slow: {len(exp2.dropna())}")
                 nan_series = pd.Series([np.nan] * len(df), index=df.index)
                 return nan_series, nan_series, nan_series

            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram

    except Exception as e:
        logger.error(f"MACD calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate MACD: {e}") from e
