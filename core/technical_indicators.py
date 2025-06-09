"""
core/technical_indicators.py

Core technical indicator calculations for financial analysis.

This module provides optimized, centralized implementations of technical indicators
with proper error handling, validation, and fallback implementations.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Union, List
from utils.logger import setup_logger

# Import from the new sub-package
from core.indicators.base import IndicatorError, validate_input
from core.indicators.rsi import calculate_rsi
from core.indicators.macd import calculate_macd
from core.indicators.bollinger_bands import calculate_bollinger_bands


logger = setup_logger(__name__)

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("pandas_ta not found - some indicators might use fallback implementations or be unavailable if not part of the new structure")

@validate_input(['high', 'low', 'close'], check_ohlcv_coherence=True)
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
