"""
DEPRECATED: This module is deprecated. Use utils.technicals.analysis and core.technical_indicators instead.

indicators.py

This module has been refactored. Core indicator calculations have been moved to:
- core.technical_indicators.py (for core calculation functions)

Higher-level analysis functionality has been moved to:
- utils.technicals.analysis.py (for composite analysis and signal generation)

This file is kept for backward compatibility but will be removed in a future version.
"""

import warnings
warnings.warn(
    "utils.technicals.indicators is deprecated. Use utils.technicals.analysis and core.technical_indicators instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new modules for backward compatibility
from utils.technicals.analysis import *
from core.technical_indicators import IndicatorError

from utils.logger import setup_logger
from typing import Optional, Union, List, Any, Dict
import numpy as np
import pandas as pd
from pandas import DataFrame

# Import centralized validation system
from core.data_validator import validate_dataframe, DataFrameValidationResult, ValidationResult, get_global_validator

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
    Validate DataFrame for technical indicator calculations using centralized validation.
    
    Args:
        df: Input DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        TypeError: If input is not a DataFrame
        ValueError: If validation fails
    """
    # First run centralized validation
    try:
        logger.debug("Running centralized validation for indicator data")
        validation_result = validate_dataframe(df, required_cols=required_columns)
        
        if not validation_result.is_valid:
            error_message = "; ".join(validation_result.errors)
            logger.error(f"Centralized validation failed: {error_message}")
            raise ValueError(f"Data validation failed: {error_message}")
        
        # Log any warnings from centralized validation
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"Data warning: {warning}")
                
        logger.debug("Centralized validation passed for indicator data")
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise ValueError(f"Failed to validate data: {e}")
    
    # Additional indicator-specific validation (if needed)
    # The centralized validator already handles:
    # - DataFrame type checking
    # - Required columns existence
    # - Empty DataFrame detection
    # - NaN value checking
    # - OHLC relationships (when applicable)
    # - Statistical anomaly detection

def add_rsi(df: DataFrame, length: int = 14, close_col: str = 'close') -> DataFrame:
    """Calculate Relative Strength Index (RSI) and append as 'rsi' column."""
    try:
        df = df.copy()
        validate_indicator_data(df, [close_col])
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
    """Calculate MACD indicator and append columns. If insufficient rows, fill columns with NaN."""
    try:
        df = df.copy()
        validate_indicator_data(df, [close_col])

        # Ensure all periods are valid
        if any(p < 1 for p in (fast, slow, signal)):
            raise ValueError("All periods must be positive integers.")
        if fast >= slow:
            raise ValueError("Fast period must be less than slow period.")

        # If not enough rows, add NaN columns and return
        min_required_rows = slow + signal
        if len(df) < min_required_rows:
            df['macd'] = np.nan
            df['macd_signal'] = np.nan
            df['macd_hist'] = np.nan
            logger.warning(f"Insufficient data for MACD. Need at least {min_required_rows} rows, got {len(df)}. Returning NaN columns.")
            return df

        if TA_AVAILABLE:
            macd = ta.macd(df[close_col], fast=fast, slow=slow, signal=signal)
            df[['macd', 'macd_signal', 'macd_hist']] = macd
        else:
            exp1 = df[close_col].ewm(span=fast, adjust=False, min_periods=fast).mean()
            exp2 = df[close_col].ewm(span=slow, adjust=False, min_periods=slow).mean()
            if exp1.isna().any() or exp2.isna().any():
                raise ValueError("EMAs contain NaN values. Check for missing or malformed data.")
            df['macd'] = exp1 - exp2
            if df['macd'].isna().all():
                raise ValueError("MACD calculation failed: resulting 'macd' column is all NaN.")
            df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False, min_periods=signal).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

        for col in ['macd', 'macd_signal', 'macd_hist']:
            if df[col].isna().all():
                raise ValueError(f"{col} column is entirely NaN. MACD calculation incomplete.")

        return df

    except Exception as e:
        logger.error(f"MACD calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate MACD: {e}") from e

def add_bollinger_bands(df: DataFrame, length: int = 20, std: Union[int, float] = 2, close_col: str = 'close') -> DataFrame:
    """Calculate Bollinger Bands and append band columns."""
    try:
        df = df.copy()
        validate_indicator_data(df, [close_col])
        if length < 1:
            raise ValueError("Length must be positive")
        if std <= 0:
            raise ValueError("Standard deviation multiplier must be positive")
        if TA_AVAILABLE:
            bb = ta.bbands(df[close_col], length=length, std=std)
            if bb is None or not hasattr(bb, "columns"):
                raise ValueError("pandas_ta.bbands returned None. Check your input data for sufficient rows and NaNs.")
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
            df['bb_upper'] = bb[upper_col]
            df['bb_middle'] = bb[middle_col]
            df['bb_lower'] = bb[lower_col]
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
        validate_indicator_data(df, ['high', 'low', 'close'])
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
        validate_indicator_data(df, [close_col])
        df[f'sma_{length}'] = df[close_col].rolling(window=length, min_periods=1).mean()
        return df
    except Exception as e:
        logger.error(f"SMA calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate SMA: {e}") from e

def add_ema(df: DataFrame, length: int = 20, close_col: str = 'close') -> DataFrame:
    """Add Exponential Moving Average (EMA) column."""
    try:
        df = df.copy()
        validate_indicator_data(df, [close_col])
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

def compute_price_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for price columns."""
    return pd.DataFrame({
        "Open": [df['open'].min(), df['open'].max(), df['open'].mean(), df['open'].std()],
        "High": [df['high'].min(), df['high'].max(), df['high'].mean(), df['high'].std()],
        "Low":  [df['low'].min(),  df['low'].max(),  df['low'].mean(),  df['low'].std()],
        "Close":[df['close'].min(),df['close'].max(),df['close'].mean(),df['close'].std()]
    }, index=["Min","Max","Mean","Std"])

def compute_return_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution statistics for daily returns."""
    daily_returns = df['close'].pct_change() * 100
    return pd.DataFrame({
        "Daily Returns (%)": [
            daily_returns.min(),
            daily_returns.quantile(0.25),
            daily_returns.median(),
            daily_returns.quantile(0.75),
            daily_returns.max(),
            daily_returns.mean(),
            daily_returns.std()
        ]
    }, index=["Min","25%","Median","75%","Max","Mean","Std"])

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
