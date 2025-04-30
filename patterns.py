"""
patterns.py

Production-grade candlestick pattern detection module.
- Modular, extensible, and robust.
- Includes input validation, logging, and error handling.
- Designed for integration with ML and rule-based pipelines.
"""

import logging
from typing import List, Callable, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class PatternDetectionError(Exception):
    """Custom exception for pattern detection errors."""
    pass

class CandlestickPatterns:
    """
    Rule-based candlestick pattern detection.
    All methods assume a DataFrame with columns: open, high, low, close.
    """

    # Registry of (pattern name, detection function, min required rows)
    _PATTERNS: List[Tuple[str, Callable[[pd.DataFrame], bool], int]] = []

    @classmethod
    def register_pattern(cls, name: str, func: Callable[[pd.DataFrame], bool], min_rows: int = 1):
        """Register a new pattern detection function."""
        cls._PATTERNS.append((name, func, min_rows))

    @classmethod
    def get_pattern_names(cls) -> List[str]:
        """Return the list of registered pattern names."""
        return [name for name, _, _ in cls._PATTERNS]

    @classmethod
    def detect_patterns(cls, df: pd.DataFrame) -> List[str]:
        """
        Detect all registered patterns in the given DataFrame.
        Returns a list of detected pattern names.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Input is not a pandas DataFrame")
            raise PatternDetectionError("Input must be a pandas DataFrame.")

        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(df.columns):
            logger.error(f"Missing required columns: {required_cols - set(df.columns)}")
            raise PatternDetectionError(f"DataFrame must contain columns: {required_cols}")

        detected = []
        for name, func, min_rows in cls._PATTERNS:
            if len(df) < min_rows:
                continue
            try:
                if func(df):
                    detected.append(name)
            except Exception as e:
                logger.warning(f"Pattern '{name}' detection failed: {e}")
        return detected

    # --- Pattern Implementations ---

    @staticmethod
    def is_hammer(df: pd.DataFrame) -> bool:
        row = df.iloc[-1]
        body = abs(row.close - row.open)
        lower_wick = min(row.open, row.close) - row.low
        upper_wick = row.high - max(row.open, row.close)
        return body > 0 and (lower_wick > 2 * body) and (upper_wick < body)

    @staticmethod
    def is_bullish_engulfing(df: pd.DataFrame) -> bool:
        prev, last = df.iloc[-2], df.iloc[-1]
        return (
            prev.close < prev.open and
            last.close > last.open and
            last.open < prev.close and
            last.close > prev.open
        )

    @staticmethod
    def is_morning_star(df: pd.DataFrame) -> bool:
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        return (
            first.close < first.open and
            abs(second.close - second.open) < abs(first.open - first.close) * 0.3 and
            third.close > third.open and
            third.close > (first.open + first.close) / 2
        )

    @staticmethod
    def is_piercing_pattern(df: pd.DataFrame) -> bool:
        prev, last = df.iloc[-2], df.iloc[-1]
        midpoint = (prev.open + prev.close) / 2
        return (
            prev.close < prev.open and
            last.open < prev.close and
            last.close > midpoint
        )

    @staticmethod
    def is_bullish_harami(df: pd.DataFrame) -> bool:
        prev, last = df.iloc[-2], df.iloc[-1]
        return (
            prev.close < prev.open and
            last.open > prev.close and
            last.close < prev.open and
            last.close > last.open
        )

    @staticmethod
    def is_three_white_soldiers(df: pd.DataFrame) -> bool:
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        return (
            first.close > first.open and
            second.close > second.open and
            third.close > third.open and
            second.open > first.open and
            third.open > second.open and
            second.close > first.close and
            third.close > second.close
        )

    @staticmethod
    def is_inverted_hammer(df: pd.DataFrame) -> bool:
        row = df.iloc[-1]
        body = abs(row.close - row.open)
        upper_wick = row.high - max(row.open, row.close)
        lower_wick = min(row.open, row.close) - row.low
        return body > 0 and (upper_wick > 2 * body) and (lower_wick < body)

    @staticmethod
    def is_doji(df: pd.DataFrame) -> bool:
        row = df.iloc[-1]
        return abs(row.open - row.close) <= (row.high - row.low) * 0.1

    @staticmethod
    def is_morning_doji_star(df: pd.DataFrame) -> bool:
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        second_open_close_diff = abs(second.open - second.close)
        return (
            first.close < first.open and
            second_open_close_diff <= (second.high - second.low) * 0.1 and
            third.close > third.open and
            third.close > (first.open + first.close) / 2
        )

    @staticmethod
    def is_bullish_abandoned_baby(df: pd.DataFrame) -> bool:
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        return (
            first.close < first.open and
            abs(second.open - second.close) < (second.high - second.low) * 0.1 and
            second.low > first.high and
            third.open > second.high and third.close > third.open
        )

    @staticmethod
    def is_bullish_belt_hold(df: pd.DataFrame) -> bool:
        row = df.iloc[-1]
        return (
            row.close > row.open and
            row.open == row.low and
            (row.close - row.open) > (row.high - row.close) * 0.5
        )

    @staticmethod
    def is_three_inside_up(df: pd.DataFrame) -> bool:
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        return (
            first.close < first.open and
            second.open > first.close and
            second.close < first.open and
            second.close > second.open and
            third.close > second.close
        )

    @staticmethod
    def is_rising_window(df: pd.DataFrame) -> bool:
        prev, last = df.iloc[-2], df.iloc[-1]
        return prev.high < last.low

# --- Register patterns with required minimum rows ---
CandlestickPatterns.register_pattern("Hammer", CandlestickPatterns.is_hammer, min_rows=1)
CandlestickPatterns.register_pattern("Bullish Engulfing", CandlestickPatterns.is_bullish_engulfing, min_rows=2)
CandlestickPatterns.register_pattern("Morning Star", CandlestickPatterns.is_morning_star, min_rows=3)
CandlestickPatterns.register_pattern("Piercing Pattern", CandlestickPatterns.is_piercing_pattern, min_rows=2)
CandlestickPatterns.register_pattern("Bullish Harami", CandlestickPatterns.is_bullish_harami, min_rows=2)
CandlestickPatterns.register_pattern("Three White Soldiers", CandlestickPatterns.is_three_white_soldiers, min_rows=3)
CandlestickPatterns.register_pattern("Inverted Hammer", CandlestickPatterns.is_inverted_hammer, min_rows=1)
CandlestickPatterns.register_pattern("Doji", CandlestickPatterns.is_doji, min_rows=1)
CandlestickPatterns.register_pattern("Morning Doji Star", CandlestickPatterns.is_morning_doji_star, min_rows=3)
CandlestickPatterns.register_pattern("Bullish Abandoned Baby", CandlestickPatterns.is_bullish_abandoned_baby, min_rows=3)
CandlestickPatterns.register_pattern("Bullish Belt Hold", CandlestickPatterns.is_bullish_belt_hold, min_rows=1)
CandlestickPatterns.register_pattern("Three Inside Up", CandlestickPatterns.is_three_inside_up, min_rows=3)
CandlestickPatterns.register_pattern("Rising Window", CandlestickPatterns.is_rising_window, min_rows=2)

# --- End of patterns.py ---