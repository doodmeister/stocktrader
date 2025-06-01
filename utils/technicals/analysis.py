"""
utils/technicals/analysis.py

High-level technical analysis functionality for trading strategies.

This module provides composite analysis, signal generation, and price targeting
using the core technical indicators from core.technical_indicators.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union, List, Dict, Tuple
from utils.logger import setup_logger
from core.technical_indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands, 
    calculate_atr, calculate_sma, calculate_ema, 
    IndicatorError, validate_indicator_data
)

logger = setup_logger(__name__)

class TechnicalAnalysis:
    """
    High-level technical analysis class for comprehensive market evaluation.
    
    Provides methods for calculating individual indicators, composite signals,
    and price targets using enterprise-grade core calculations.
    """

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.data = data.copy()

    def sma(self, period: int = 20, close_col: str = 'close') -> pd.Series:
        """Simple Moving Average."""
        try:
            return calculate_sma(self.data, length=period, close_col=close_col)
        except Exception as e:
            logger.error(f"SMA calculation failed: {e}")
            raise

    def ema(self, period: int = 20, close_col: str = 'close') -> pd.Series:
        """Exponential Moving Average."""
        try:
            return calculate_ema(self.data, length=period, close_col=close_col)
        except Exception as e:
            logger.error(f"EMA calculation failed: {e}")
            raise

    def rsi(self, period: int = 14, close_col: str = 'close') -> pd.Series:
        """Relative Strength Index."""
        try:
            return calculate_rsi(self.data, length=period, close_col=close_col)
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            raise

    def macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, 
             close_col: str = 'close') -> Tuple[pd.Series, pd.Series]:
        """Moving Average Convergence Divergence."""
        try:
            macd_line, signal_line, _ = calculate_macd(
                self.data, fast=fast_period, slow=slow_period, 
                signal=signal_period, close_col=close_col
            )
            return macd_line, signal_line
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            raise

    def bollinger_bands(self, period: int = 20, std_dev: Union[int, float] = 2, 
                       close_col: str = 'close') -> Tuple[pd.Series, pd.Series]:
        """Bollinger Bands."""
        try:
            upper, _, lower = calculate_bollinger_bands(
                self.data, length=period, std=std_dev, close_col=close_col
            )
            return upper, lower
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            raise

    def atr(self, period: int = 14) -> pd.Series:
        """Average True Range."""
        try:
            return calculate_atr(self.data, length=period)
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            raise

    def evaluate(self, market_data: Optional[pd.DataFrame] = None, rsi_period: int = 14,
                macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                bb_period: int = 20, bb_std: Union[int, float] = 2) -> Tuple[Optional[float], ...]:
        """
        Evaluate market data using technical indicators and return a composite signal.
        Returns a float in [-1, 1] (bearish to bullish).
        """
        df = market_data if market_data is not None else self.data
        min_len = max(rsi_period, macd_slow + macd_signal, bb_period)
        
        if df is None or df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            return None, None, None, None
        if len(df) < min_len:
            logger.warning(f"Not enough data for evaluation: need at least {min_len} rows, got {len(df)}")
            return None, None, None, None

        try:
            # Calculate indicators with user parameters
            rsi = calculate_rsi(df, length=rsi_period)
            macd_line, macd_sig, _ = calculate_macd(
                df, fast=macd_fast, slow=macd_slow, signal=macd_signal
            )
            bb_upper, _, bb_lower = calculate_bollinger_bands(
                df, length=bb_period, std=bb_std
            )
            close = df['close']

            # RSI Score
            rsi_score = ((rsi.iloc[-1] - 50) / 50) if not pd.isna(rsi.iloc[-1]) else 0

            # MACD Score (normalized and clamped)
            macd_diff = macd_line.iloc[-1] - macd_sig.iloc[-1]
            macd_range = max(abs(macd_line.max()), abs(macd_line.min()), 1e-6)
            macd_score = np.clip(macd_diff / macd_range, -1, 1)

            # Bollinger Bands Score (scaled)
            price = close.iloc[-1]
            if bb_upper.iloc[-1] != bb_lower.iloc[-1]:
                bb_score = np.clip((2 * (price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) - 1), -1, 1)
            else:
                bb_score = 0

            composite = np.mean([rsi_score, macd_score, bb_score])
            composite = np.clip(composite, -1, 1)

            logger.debug(f"RSI_score={rsi_score:.2f}, MACD_score={macd_score:.2f}, BB_score={bb_score:.2f}")

            return float(composite), rsi_score, macd_score, bb_score
        except Exception as e:
            logger.error(f"Error in evaluate(): {e}")
            return None, None, None, None

    def calculate_atr(self) -> Optional[float]:
        """Wrapper for ATR calculation returning single value."""
        try:
            atr_series = calculate_atr(self.data, length=3)
            if atr_series is None or pd.isna(atr_series.iloc[-1]):
                return None
            return float(atr_series.iloc[-1])
        except Exception:
            return None

    def calculate_price_target_fib(self, lookback: int = 20, extension: float = 0.618) -> float:
        """
        Fibonacci‐extension price target:
          - Finds the highest high and lowest low over the past `lookback` bars.
          - Projects the target = swing_high + ( (high–low) * extension ).
        """
        if len(self.data) < lookback:
            return float(self.data['close'].iloc[-1])

        recent = self.data.iloc[-lookback:]
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        diff = swing_high - swing_low
        return float(swing_high + diff * extension)

    def calculate_price_target(self) -> float:
        """Calculate price target using Fibonacci extension."""
        return self.calculate_price_target_fib(lookback=30, extension=0.618)

    @staticmethod
    def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all standard technical indicators and composite signal to the DataFrame.
        """
        try:
            result = df.copy()
            
            # Add individual indicators
            result['rsi'] = calculate_rsi(df)
            macd, macd_signal, macd_hist = calculate_macd(df)
            result['macd'] = macd
            result['macd_signal'] = macd_signal
            result['macd_hist'] = macd_hist
            
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df)
            result['bb_upper'] = bb_upper
            result['bb_middle'] = bb_middle
            result['bb_lower'] = bb_lower
            
            result['atr'] = calculate_atr(df)
            result['sma_20'] = calculate_sma(df)
            result['ema_20'] = calculate_ema(df)
            
            # Add composite signals and price targets
            result = add_composite_signal(result)
            result = calculate_price_target_columns(result)
            
            return result
        except Exception as e:
            logger.error(f"Failed to enrich DataFrame with technical indicators: {e}")
            return df

def add_composite_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a composite signal column based on RSI, MACD, and Bollinger Bands.
    Example logic: Buy if RSI < 30 and MACD > MACD_signal and close < BB_lower.
    """
    df = df.copy()
    df['composite_signal'] = 0
    
    # Ensure required columns exist
    required_cols = ['rsi', 'macd', 'macd_signal', 'close', 'bb_lower', 'bb_upper']
    if not all(col in df.columns for col in required_cols):
        logger.warning("Missing required columns for composite signal calculation")
        return df
    
    buy = (df['rsi'] < 30) & (df['macd'] > df['macd_signal']) & (df['close'] < df['bb_lower'])
    sell = (df['rsi'] > 70) & (df['macd'] < df['macd_signal']) & (df['close'] > df['bb_upper'])
    
    df.loc[buy, 'composite_signal'] = 1
    df.loc[sell, 'composite_signal'] = -1
    
    return df

def calculate_price_target_columns(df: pd.DataFrame, atr_mult: float = 1.5) -> pd.DataFrame:
    """
    Add price target columns based on ATR.
    """
    df = df.copy()
    
    if 'atr' not in df.columns:
        df['atr'] = calculate_atr(df)
    
    df['target_price_up'] = df['close'] + atr_mult * df['atr']
    df['target_price_down'] = df['close'] - atr_mult * df['atr']
    
    return df

def add_technical_indicators(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    """
    Add multiple technical indicators to a price DataFrame.
    Supported: 'SMA', 'EMA', 'MACD', 'RSI', 'Bollinger Bands', 'ATR'
    """
    result = df.copy()
    
    for indicator in indicators:
        try:
            if indicator == "SMA":
                result['sma_20'] = calculate_sma(result)
            elif indicator == "EMA":
                result['ema_20'] = calculate_ema(result)
            elif indicator == "MACD":
                macd, macd_signal, macd_hist = calculate_macd(result)
                result['macd'] = macd
                result['macd_signal'] = macd_signal
                result['macd_hist'] = macd_hist
            elif indicator == "RSI":
                result['rsi'] = calculate_rsi(result)
            elif indicator == "Bollinger Bands":
                bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(result)
                result['bb_upper'] = bb_upper
                result['bb_middle'] = bb_middle
                result['bb_lower'] = bb_lower
            elif indicator == "ATR":
                result['atr'] = calculate_atr(result)
            else:
                logger.warning(f"Unknown indicator: {indicator}")
        except Exception as e:
            logger.error(f"Failed to calculate {indicator}: {e}")
            
    return result

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

# Backward compatibility classes
class TechnicalIndicators:
    """
    Facade for technical indicator functions for backward compatibility.
    """
    @staticmethod
    def add_rsi(df: DataFrame, length: int = 14, close_col: str = 'close') -> DataFrame:
        result = df.copy()
        result['rsi'] = calculate_rsi(df, length, close_col)
        return result
        
    @staticmethod
    def add_macd(df: DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, close_col: str = 'close') -> DataFrame:
        result = df.copy()
        macd, macd_signal, macd_hist = calculate_macd(df, fast, slow, signal, close_col)
        result['macd'] = macd
        result['macd_signal'] = macd_signal
        result['macd_hist'] = macd_hist
        return result
        
    @staticmethod
    def add_bollinger_bands(df: DataFrame, length: int = 20, std: Union[int, float] = 2, close_col: str = 'close') -> DataFrame:
        result = df.copy()
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df, length, std, close_col)
        result['bb_upper'] = bb_upper
        result['bb_middle'] = bb_middle
        result['bb_lower'] = bb_lower
        return result
        
    @staticmethod
    def add_atr(df: DataFrame, length: int = 14) -> DataFrame:
        result = df.copy()
        result['atr'] = calculate_atr(df, length)
        return result
        
    @staticmethod
    def add_sma(df: DataFrame, length: int = 20, close_col: str = 'close') -> DataFrame:
        result = df.copy()
        result[f'sma_{length}'] = calculate_sma(df, length, close_col)
        return result
        
    @staticmethod
    def add_ema(df: DataFrame, length: int = 20, close_col: str = 'close') -> DataFrame:
        result = df.copy()
        result[f'ema_{length}'] = calculate_ema(df, length, close_col)
        return result
        
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        return add_technical_indicators(df, indicators)
        
    @staticmethod
    def add_composite_signal(df: pd.DataFrame) -> pd.DataFrame:
        return add_composite_signal(df)
        
    @staticmethod
    def calculate_price_target(df: pd.DataFrame, atr_mult: float = 1.5) -> pd.DataFrame:
        return calculate_price_target_columns(df, atr_mult)

# DataFrame-returning wrapper functions for backward compatibility
def add_rsi(df: DataFrame, length: int = 14, close_col: str = 'close') -> DataFrame:
    """Add RSI column to DataFrame and return the DataFrame."""
    result = df.copy()
    result['rsi'] = calculate_rsi(df, length, close_col)
    return result

def add_macd(df: DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, close_col: str = 'close') -> DataFrame:
    """Add MACD columns to DataFrame and return the DataFrame."""
    result = df.copy()
    macd, macd_signal, macd_hist = calculate_macd(df, fast, slow, signal, close_col)
    result['macd'] = macd
    result['macd_signal'] = macd_signal
    result['macd_hist'] = macd_hist
    return result

def add_bollinger_bands(df: DataFrame, length: int = 20, std: Union[int, float] = 2, close_col: str = 'close') -> DataFrame:
    """Add Bollinger Bands columns to DataFrame and return the DataFrame."""
    result = df.copy()
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df, length, std, close_col)
    result['bb_upper'] = bb_upper
    result['bb_middle'] = bb_middle
    result['bb_lower'] = bb_lower
    return result

def add_atr(df: DataFrame, length: int = 14) -> DataFrame:
    """Add ATR column to DataFrame and return the DataFrame."""
    result = df.copy()
    result['atr'] = calculate_atr(df, length)
    return result

def add_sma(df: DataFrame, length: int = 20, close_col: str = 'close') -> DataFrame:
    """Add SMA column to DataFrame and return the DataFrame."""
    result = df.copy()
    result[f'sma_{length}'] = calculate_sma(df, length, close_col)
    return result

def add_ema(df: DataFrame, length: int = 20, close_col: str = 'close') -> DataFrame:
    """Add EMA column to DataFrame and return the DataFrame."""
    result = df.copy()
    result[f'ema_{length}'] = calculate_ema(df, length, close_col)
    return result
