"""
patterns.py

Expanded candlestick pattern detection module.
Includes basic, intermediate, and newly added advanced bullish patterns.
Provides pattern names when detected for better logging and monitoring.
"""

import pandas as pd

class CandlestickPatterns:
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> list:
        """Return a list of bullish patterns detected with their detection logic."""
        pattern_checks = [
            ("Hammer", CandlestickPatterns.is_hammer),
            ("Bullish Engulfing", CandlestickPatterns.is_bullish_engulfing),
            ("Morning Star", CandlestickPatterns.is_morning_star),
            ("Piercing Pattern", CandlestickPatterns.is_piercing_pattern),
            ("Bullish Harami", CandlestickPatterns.is_bullish_harami),
            ("Three White Soldiers", CandlestickPatterns.is_three_white_soldiers),
            ("Inverted Hammer", CandlestickPatterns.is_inverted_hammer),
            ("Doji", CandlestickPatterns.is_doji),
            ("Morning Doji Star", CandlestickPatterns.is_morning_doji_star),
            ("Bullish Abandoned Baby", CandlestickPatterns.is_bullish_abandoned_baby),
            ("Bullish Belt Hold", CandlestickPatterns.is_bullish_belt_hold),
            ("Three Inside Up", CandlestickPatterns.is_three_inside_up),
            ("Rising Window", CandlestickPatterns.is_rising_window)
        ]

        detected = []
        for name, func in pattern_checks:
            try:
                if func(df):
                    detected.append(name)
            except Exception:
                continue

        return detected

    @classmethod
    def get_pattern_names(cls):
        """Return the list of pattern names."""
        pattern_checks = [
            ("Hammer", cls.is_hammer),
            ("Bullish Engulfing", cls.is_bullish_engulfing),
            ("Morning Star", cls.is_morning_star),
            ("Piercing Pattern", cls.is_piercing_pattern),
            ("Bullish Harami", cls.is_bullish_harami),
            ("Three White Soldiers", cls.is_three_white_soldiers),
            ("Inverted Hammer", cls.is_inverted_hammer),
            ("Doji", cls.is_doji),
            ("Morning Doji Star", cls.is_morning_doji_star),
            ("Bullish Abandoned Baby", cls.is_bullish_abandoned_baby),
            ("Bullish Belt Hold", cls.is_bullish_belt_hold),
            ("Three Inside Up", cls.is_three_inside_up),
            ("Rising Window", cls.is_rising_window)
        ]
        return [name for name, _ in pattern_checks]

    @staticmethod
    def is_hammer(df: pd.DataFrame) -> bool:
        o, h, l, c = df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        body = abs(c - o)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        return (lower_wick > 2 * body) and (upper_wick < body)

    @staticmethod
    def is_bullish_engulfing(df: pd.DataFrame) -> bool:
        prev, last = df.iloc[-2], df.iloc[-1]
        return (prev.close < prev.open) and (last.close > last.open) and (last.open < prev.close) and (last.close > prev.open)

    @staticmethod
    def is_morning_star(df: pd.DataFrame) -> bool:
        if len(df) < 3:
            return False
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        return (
            first.close < first.open and
            abs(second.close - second.open) < (first.open - first.close) * 0.3 and
            third.close > third.open and
            third.close > (first.open + first.close) / 2
        )

    @staticmethod
    def is_piercing_pattern(df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        prev, last = df.iloc[-2], df.iloc[-1]
        midpoint = (prev.open + prev.close) / 2
        return (prev.close < prev.open and last.open < prev.close and last.close > midpoint)

    @staticmethod
    def is_bullish_harami(df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        prev, last = df.iloc[-2], df.iloc[-1]
        return (prev.close < prev.open and last.open > prev.close and last.close < prev.open and last.close > last.open)

    @staticmethod
    def is_three_white_soldiers(df: pd.DataFrame) -> bool:
        if len(df) < 3:
            return False
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
        o, h, l, c = df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        return (upper_wick > 2 * body) and (lower_wick < body)

    @staticmethod
    def is_doji(df: pd.DataFrame) -> bool:
        o, c = df['open'].iloc[-1], df['close'].iloc[-1]
        return abs(o - c) <= (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.1

    @staticmethod
    def is_morning_doji_star(df: pd.DataFrame) -> bool:
        if len(df) < 3:
            return False
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
        if len(df) < 3:
            return False
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        return (
            first.close < first.open and
            abs(second.open - second.close) < (second.high - second.low) * 0.1 and
            second.low > first.high and
            third.open > second.high and third.close > third.open
        )

    @staticmethod
    def is_bullish_belt_hold(df: pd.DataFrame) -> bool:
        o, h, l, c = df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        return (c > o) and (o == l) and ((c - o) > (h - c) * 0.5)

    @staticmethod
    def is_three_inside_up(df: pd.DataFrame) -> bool:
        if len(df) < 3:
            return False
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        return (
            first.close < first.open and
            second.open > first.close and second.close < first.open and second.close > second.open and
            third.close > second.close
        )

    @staticmethod
    def is_rising_window(df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        prev, last = df.iloc[-2], df.iloc[-1]
        return prev.high < last.low