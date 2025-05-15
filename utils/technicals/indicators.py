"""
indicators.py

Unified technical indicators module for financial analysis.

Provides wrapper functions around technical analysis libraries with fallback implementations.
Handles input validation, error cases, and proper typing.
Adds composite signal and price target logic from technical_analysis.py.
"""
from utils.logger import setup_logger
from typing import Optional, Union, List, Any, Dict
import utils.logger as logging
import numpy as np
import pandas as pd
from pandas import DataFrame

logger = setup_logger(__name__)

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("pandas_ta not found - using fallback implementations for indicators")

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

def add_rsi(df: DataFrame, length: int = 14, close_col: str = 'close') -> DataFrame:
    """Calculate Relative Strength Index (RSI) and append as 'rsi' column."""
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

def add_macd(df: DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, close_col: str = 'close') -> DataFrame:
    """Calculate MACD indicator and append columns."""
    try:
        df = df.copy()
        validate_dataframe(df, [close_col])
        if any(p < 1 for p in (fast, slow, signal)):
            raise ValueError("All periods must be positive")
        if fast >= slow:
            raise ValueError("Fast period must be less than slow period")
        if TA_AVAILABLE:
            macd = ta.macd(df[close_col], fast=fast, slow=slow, signal=signal)
            df[['macd','macd_signal','macd_hist']] = macd
        else:
            exp1 = df[close_col].ewm(span=fast, adjust=False, min_periods=1).mean()
            exp2 = df[close_col].ewm(span=slow, adjust=False, min_periods=1).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False, min_periods=1).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        return df
    except Exception as e:
        logger.error(f"MACD calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate MACD: {e}") from e

def add_bollinger_bands(df: DataFrame, length: int = 20, std: Union[int, float] = 2, close_col: str = 'close') -> DataFrame:
    """Calculate Bollinger Bands and append band columns."""
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
            ma = df[close_col].rolling(window=length, min_periods=1).mean()
            sd = df[close_col].rolling(window=length, min_periods=1).std()
            df['bb_middle'] = ma
            df['bb_upper'] = ma + (std * sd)
            df['bb_lower'] = ma - (std * sd)
        return df
    except Exception as e:
        logger.error(f"Bollinger Bands calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate Bollinger Bands: {e}") from e

def add_atr(df: DataFrame, length: int = 14) -> DataFrame:
    """Calculate Average True Range (ATR) and append as 'atr' column."""
    try:
        df = df.copy()
        validate_dataframe(df, ['high', 'low', 'close'])
        if length < 1:
            raise ValueError("Length must be positive")
        if TA_AVAILABLE:
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=length)
        else:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=length, min_periods=1).mean()
        return df
    except Exception as e:
        logger.error(f"ATR calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate ATR: {e}") from e

def add_sma(df: DataFrame, length: int = 20, close_col: str = 'close') -> DataFrame:
    """Add Simple Moving Average (SMA) column."""
    try:
        df = df.copy()
        validate_dataframe(df, [close_col])
        df[f'sma_{length}'] = df[close_col].rolling(window=length, min_periods=1).mean()
        return df
    except Exception as e:
        logger.error(f"SMA calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate SMA: {e}") from e

def add_ema(df: DataFrame, length: int = 20, close_col: str = 'close') -> DataFrame:
    """Add Exponential Moving Average (EMA) column."""
    try:
        df = df.copy()
        validate_dataframe(df, [close_col])
        df[f'ema_{length}'] = df[close_col].ewm(span=length, adjust=False).mean()
        return df
    except Exception as e:
        logger.error(f"EMA calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate EMA: {e}") from e

def add_technical_indicators(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    """
    Add multiple technical indicators to a price DataFrame.
    Supported: 'SMA', 'EMA', 'MACD', 'RSI', 'Bollinger Bands', 'ATR'
    """
    result = df.copy()
    for indicator in indicators:
        if indicator == "SMA":
            result = add_sma(result)
        elif indicator == "EMA":
            result = add_ema(result)
        elif indicator == "MACD":
            result = add_macd(result)
        elif indicator == "RSI":
            result = add_rsi(result)
        elif indicator == "Bollinger Bands":
            result = add_bollinger_bands(result)
        elif indicator == "ATR":
            result = add_atr(result)
    return result

def add_composite_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a composite signal column based on RSI, MACD, and Bollinger Bands.
    Example logic: Buy if RSI < 30 and MACD > MACD_signal and close < BB_lower.
    """
    df = df.copy()
    df['composite_signal'] = 0
    buy = (df['rsi'] < 30) & (df['macd'] > df['macd_signal']) & (df['close'] < df['bb_lower'])
    sell = (df['rsi'] > 70) & (df['macd'] < df['macd_signal']) & (df['close'] > df['bb_upper'])
    df.loc[buy, 'composite_signal'] = 1
    df.loc[sell, 'composite_signal'] = -1
    return df

def calculate_price_target(df: pd.DataFrame, atr_mult: float = 1.5) -> pd.DataFrame:
    """
    Add price target columns based on ATR.
    """
    df = df.copy()
    if 'atr' not in df.columns:
        df = add_atr(df)
    df['target_price_up'] = df['close'] + atr_mult * df['atr']
    df['target_price_down'] = df['close'] - atr_mult * df['atr']
    return df

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
    @staticmethod
    def add_atr(df: DataFrame, length: int = 14) -> DataFrame:
        return add_atr(df, length)
    @staticmethod
    def add_sma(df: DataFrame, length: int = 20, close_col: str = 'close') -> DataFrame:
        return add_sma(df, length, close_col)
    @staticmethod
    def add_ema(df: DataFrame, length: int = 20, close_col: str = 'close') -> DataFrame:
        return add_ema(df, length, close_col)
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        return add_technical_indicators(df, indicators)
    @staticmethod
    def add_composite_signal(df: pd.DataFrame) -> pd.DataFrame:
        return add_composite_signal(df)
    @staticmethod
    def calculate_price_target(df: pd.DataFrame, atr_mult: float = 1.5) -> pd.DataFrame:
        return calculate_price_target(df, atr_mult)
