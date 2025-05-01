"""
indicators.py

Technical indicators module for financial analysis.

Provides wrapper functions around technical analysis libraries with fallback implementations.
Handles input validation, error cases, and proper typing.
"""
import logging
from typing import Optional, Union, List, Any, Dict
import numpy as np
import pandas as pd
from pandas import DataFrame

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("pandas_ta not found - using fallback implementations for indicators")

# Configure logging
logger = logging.getLogger(__name__)

class IndicatorError(Exception):
    """Custom exception for indicator calculation errors."""
    pass


def validate_dataframe(df: DataFrame, required_columns: List[str]) -> None:
    """Validate DataFrame has required columns and no NaN values."""
    if not isinstance(df, DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
    if df.empty:
        raise ValueError("DataFrame is empty")
        
    if df[required_columns].isna().any().any():
        raise ValueError("DataFrame contains NaN values in required columns")


def add_rsi(
    df: DataFrame, 
    length: int = 14,
    close_col: str = 'close'
) -> DataFrame:
    """
    Calculate Relative Strength Index (RSI) and append as 'rsi' column.
    """
    try:
        df = df.copy()
        validate_dataframe(df, [close_col])
        
        if length < 1:
            raise ValueError("Length must be positive")
            
        if TA_AVAILABLE:
            df['rsi'] = ta.rsi(df[close_col], length=length)
        else:
            delta = df[close_col].diff()
            gain = delta.clip(lower=0).rolling(window=length, min_periods=1).mean()
            loss = -delta.clip(upper=0).rolling(window=length, min_periods=1).mean()
            
            rs = gain / np.where(loss == 0, np.inf, loss)
            df['rsi'] = 100 - (100 / (1 + rs))
            
        df['rsi'] = df['rsi'].clip(0, 100)
        return df
        
    except Exception as e:
        logger.error(f"RSI calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate RSI: {e}") from e


def add_macd(
    df: DataFrame,
    fast: int = 12,
    slow: int = 26, 
    signal: int = 9,
    close_col: str = 'close'
) -> DataFrame:
    """
    Calculate MACD indicator and append columns.
    """
    try:
        df = df.copy()
        validate_dataframe(df, [close_col])
        
        if any(p < 1 for p in (fast, slow, signal)):
            raise ValueError("All periods must be positive")
        if fast >= slow:
            raise ValueError("Fast period must be less than slow period")
            
        if TA_AVAILABLE:
            macd = ta.macd(
                df[close_col],
                fast=fast,
                slow=slow,
                signal=signal
            )
            df[['macd','macd_signal','macd_hist']] = macd
        else:
            exp1 = df[close_col].ewm(span=fast, adjust=False, min_periods=1).mean()
            exp2 = df[close_col].ewm(span=slow, adjust=False, min_periods=1).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(
                span=signal, 
                adjust=False,
                min_periods=1
            ).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
        return df
        
    except Exception as e:
        logger.error(f"MACD calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate MACD: {e}") from e


def add_bollinger_bands(
    df: DataFrame,
    length: int = 20,
    std: Union[int, float] = 2,
    close_col: str = 'close'
) -> DataFrame:
    """
    Calculate Bollinger Bands and append band columns.
    """
    try:
        df = df.copy()
        validate_dataframe(df, [close_col])
        
        if length < 1:
            raise ValueError("Length must be positive")
        if std <= 0:
            raise ValueError("Standard deviation multiplier must be positive")
            
        if TA_AVAILABLE:
            bb = ta.bbands(df[close_col], length=length, std=std)
            df[['bb_upper','bb_middle','bb_lower']] = bb[[
                f'BBU_{length}_{std}',
                f'BBM_{length}_{std}',
                f'BBL_{length}_{std}'
            ]]
        else:
            ma = df[close_col].rolling(
                window=length,
                min_periods=1
            ).mean()
            sd = df[close_col].rolling(
                window=length,
                min_periods=1
            ).std()
            
            df['bb_middle'] = ma
            df['bb_upper'] = ma + (std * sd)
            df['bb_lower'] = ma - (std * sd)
            
        return df
        
    except Exception as e:
        logger.error(f"Bollinger Bands calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate Bollinger Bands: {e}") from e


class TechnicalIndicators:
    """
    Facade for technical indicator functions. Call static methods or instantiate to use.
    """

    @staticmethod
    def add_rsi(df: DataFrame, length: int = 14, close_col: str = 'close') -> DataFrame:
        return add_rsi(df, length, close_col)

    @staticmethod
    def add_macd(df: DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, close_col: str = 'close') -> DataFrame:
        return add_macd(df, fast, slow, signal, close_col)

    @staticmethod
    def add_bollinger_bands(df: DataFrame, length: int = 20, std: Union[int, float] = 2, close_col: str = 'close') -> DataFrame:
        return add_bollinger_bands(df, length, std, close_col)
