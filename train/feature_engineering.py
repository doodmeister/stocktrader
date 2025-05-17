import pandas as pd
import numpy as np

def compute_technical_features(df):
    """
    Adds technical indicator columns (e.g., RSI, MACD, Bollinger Bands, ATR, etc.)
    to the DataFrame and returns the enriched DataFrame.
    """
    # Example using TA-Lib or pandas_ta, or your own calculations

    # Ensure required columns exist
    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    # Example: Simple RSI and MACD using pandas_ta (if installed)
    try:
        import pandas_ta as ta
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        bbands = ta.bbands(df['close'])
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0']
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    except ImportError:
        # Fallback: simple moving average as an example
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()

    # Fill any new NaNs (from indicators) if desired
    df = df.fillna(0)
    return df

from patterns.pattern_utils import get_pattern_names, get_pattern_method
from typing import Optional, List

def add_candlestick_pattern_features(df: pd.DataFrame, selected_patterns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    For each registered candlestick pattern, add a binary column to the DataFrame
    indicating whether the pattern is detected at each row (using a rolling window).
    Uses pandas' rolling().apply() for efficiency.
    """
    pattern_names = selected_patterns or get_pattern_names()
    for pattern in pattern_names:
        method = get_pattern_method(pattern)
        min_rows = 3  # Default window size
        try:
            from patterns.patterns import CandlestickPatterns
            for name, _, mr in CandlestickPatterns._PATTERNS:
                if name == pattern:
                    min_rows = mr
                    break
        except Exception:
            pass

        def safe_method(window):
            try:
                return int(method(window)) if method else 0
            except Exception:
                return 0

        result = (
            df.rolling(window=min_rows, min_periods=min_rows)
              .apply(safe_method, raw=False)
              .fillna(0)
              .astype(int)
        )
        # Assign as numpy array to avoid index alignment issues
        df[pattern.replace(" ", "")] = result.to_numpy()

    return df