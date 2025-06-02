"""
core/technical_indicators.py

Core technical indicator calculations for financial analysis.

This module provides optimized, centralized implementations of technical indicators
with proper error handling, validation, and fallback implementations.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Union, List, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("pandas_ta not found - using fallback implementations for indicators")

class IndicatorError(Exception):
    """Custom exception for indicator calculation errors."""
    pass

def validate_indicator_data(df: DataFrame, required_columns: List[str]) -> None:
    """
    Validate DataFrame for technical indicator calculations.
    
    This function provides basic validation without circular imports.
    For advanced validation features, use core.data_validator directly.
    
    Args:
        df: Input DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        TypeError: If input is not a DataFrame
        ValueError: If validation fails
    """
    # Basic type validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for null values in required columns
    null_columns = []
    for col in required_columns:
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            null_columns.append(f"{col} ({null_count} nulls)")
    
    if null_columns:
        logger.warning(f"Null values found in columns: {', '.join(null_columns)}")
    
    # Basic OHLC validation if applicable
    ohlc_cols = ['open', 'high', 'low', 'close']
    if all(col in df.columns for col in ohlc_cols):
        invalid_rows = []
        for idx, row in df.iterrows():
            if not (row['low'] <= row['open'] <= row['high'] and 
                   row['low'] <= row['close'] <= row['high']):
                invalid_rows.append(idx)
        
        if invalid_rows:
            logger.warning(f"Invalid OHLC relationships found in {len(invalid_rows)} rows")
            if len(invalid_rows) <= 5:  # Log specific rows if not too many
                logger.warning(f"Invalid rows: {invalid_rows}")
    
    logger.debug(f"Validation passed for indicator data: {len(df)} rows, {len(df.columns)} columns")

def validate_input(required_cols):
    """
    Decorator to validate DataFrame input for indicator functions.
    Usage: @validate_input(['close'])
    """
    def decorator(func):
        def wrapper(df, *args, **kwargs):
            validate_indicator_data(df, required_cols)
            return func(df, *args, **kwargs)
        return wrapper
    return decorator

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
            rsi = ta.rsi(df[close_col], length=length)
            if rsi is None:
                raise ValueError("pandas_ta.rsi returned None")
            # If rsi is a DataFrame, extract the first column as Series
            if isinstance(rsi, pd.DataFrame):
                if rsi.shape[1] == 0:
                    raise ValueError("pandas_ta.rsi returned empty DataFrame")
                rsi = rsi.iloc[:, 0]
        else:
            delta = df[close_col].diff()
            gain = delta.clip(lower=0).rolling(window=length, min_periods=1).mean()
            loss = -delta.clip(upper=0).rolling(window=length, min_periods=1).mean()
            rs = gain / np.where(loss == 0, np.inf, loss)
            rsi = 100 - (100 / (1 + rs))
        
        return pd.Series(rsi, index=df.index).clip(0, 100)
    except Exception as e:
        logger.error(f"RSI calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate RSI: {e}") from e

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
        min_required_rows = slow + signal
        if len(df) < min_required_rows:
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
            
            if exp1.isna().any() or exp2.isna().any():
                raise ValueError("EMAs contain NaN values. Check for missing or malformed data.")
                
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram

    except Exception as e:
        logger.error(f"MACD calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate MACD: {e}") from e

@validate_input(['close'])
def calculate_bollinger_bands(df: DataFrame, length: int = 20, std: Union[int, float] = 2, 
                            close_col: str = 'close') -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with price data
        length: Period for moving average
        std: Standard deviation multiplier
        close_col: Name of close price column
        
    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    try:
        if length < 1:
            raise ValueError("Length must be positive")
        if std <= 0:
            raise ValueError("Standard deviation multiplier must be positive")
            
        if TA_AVAILABLE:
            bb = ta.bbands(df[close_col], length=length, std=std)
            if bb is None or not hasattr(bb, "columns"):
                raise ValueError("pandas_ta.bbands returned None. Check your input data for sufficient rows and NaNs")
            
            # Try all reasonable column name variants for each band
            std_variants = [str(std), f"{float(std):.1f}", f"{float(std)}"]
            def find_col(prefix):
                for std_str in std_variants:
                    col = f"{prefix}_{length}_{std_str}"
                    if col in bb.columns:
                        return col
                return None
                
            upper_col = find_col('BBU')
            middle_col = find_col('BBM')
            lower_col = find_col('BBL')
            
            if not all([upper_col, middle_col, lower_col]):
                raise KeyError(f"Bollinger Bands columns not found in DataFrame: {bb.columns}")
                
            return bb[upper_col], bb[middle_col], bb[lower_col]
        else:
            ma = df[close_col].rolling(window=length, min_periods=1).mean()
            sd = df[close_col].rolling(window=length, min_periods=1).std()
            upper = ma + (std * sd)
            lower = ma - (std * sd)
            
            return upper, ma, lower
            
    except Exception as e:
        logger.error(f"Bollinger Bands calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate Bollinger Bands: {e}") from e

@validate_input(['high', 'low', 'close'])
def calculate_atr(df: DataFrame, length: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with OHLC data
        length: Period for ATR calculation
        
    Returns:
        pd.Series: ATR values
    """
    try:
        if length < 1:
            raise ValueError("Length must be positive")
            
        if TA_AVAILABLE:
            atr = ta.atr(df['high'], df['low'], df['close'], length=length)
            if atr is None:
                atr = pd.Series([np.nan] * len(df), index=df.index)
        else:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            # Ensure all are pandas Series before concat
            tr = pd.concat([
                high_low.rename("high_low"),
                high_close.rename("high_close"),
                low_close.rename("low_close")
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=length, min_periods=1).mean()
            
        return atr
    except Exception as e:
        logger.error(f"ATR calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate ATR: {e}") from e

@validate_input(['close'])
def calculate_sma(df: DataFrame, length: int = 20, close_col: str = 'close') -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        df: DataFrame with price data
        length: Period for SMA calculation
        close_col: Name of close price column
        
    Returns:
        pd.Series: SMA values
    """
    try:
        if length < 1:
            raise ValueError("Length must be positive")
            
        return df[close_col].rolling(window=length, min_periods=1).mean()
    except Exception as e:
        logger.error(f"SMA calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate SMA: {e}") from e

@validate_input(['close'])
def calculate_ema(df: DataFrame, length: int = 20, close_col: str = 'close') -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        df: DataFrame with price data
        length: Period for EMA calculation
        close_col: Name of close price column
        
    Returns:
        pd.Series: EMA values
    """
    try:
        if length < 1:
            raise ValueError("Length must be positive")
            
        return df[close_col].ewm(span=length, adjust=False).mean()
    except Exception as e:
        logger.error(f"EMA calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate EMA: {e}") from e
