"""
patterns.py

Production-grade candlestick pattern detection module.
- Modular, extensible, and robust.
- Includes input validation, logging, and error handling.
- Designed for integration with ML and rule-based pipelines.
"""

from utils.logger import setup_logger
from typing import List, Callable, Tuple
import pandas as pd

logger = setup_logger(__name__)

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
        """
        The Hammer is a bullish reversal pattern that forms during a downtrend.
        
        Visual: A candle with a small body at the top and a long lower shadow (at least 
        twice the length of the body) with little to no upper shadow.
        
        Psychology: After a downtrend, bears push prices lower, but bulls step in and 
        drive prices back up to close near the opening price, signaling that bulls 
        are regaining control.
        
        Significance: Often signals a potential bottom or support level has been reached, 
        suggesting a reversal from bearish to bullish sentiment.
        """
        row = df.iloc[-1]
        body = abs(row.close - row.open)
        lower_wick = min(row.open, row.close) - row.low
        upper_wick = row.high - max(row.open, row.close)
        return body > 0 and (lower_wick > 2 * body) and (upper_wick < body)

    @staticmethod
    def is_bullish_engulfing(df: pd.DataFrame) -> bool:
        """
        The Bullish Engulfing pattern is a two-candle reversal pattern that occurs during a downtrend.
        
        Visual: A small bearish (red/black) candle followed by a larger bullish (green/white) 
        candle that completely engulfs the body of the previous candle.
        
        Psychology: After a downtrend, bears lose momentum and bulls take control, 
        overpowering the previous bearish sentiment. The larger bullish candle shows 
        strong buying pressure.
        
        Significance: Signals a potential reversal from a downtrend to an uptrend, 
        especially when appearing at support levels or after extended downward movements.
        """
        prev, last = df.iloc[-2], df.iloc[-1]
        return (
            prev.close < prev.open and
            last.close > last.open and
            last.open < prev.close and
            last.close > prev.open
        )

    @staticmethod
    def is_morning_star(df: pd.DataFrame) -> bool:
        """
        The Morning Star is a three-candle bullish reversal pattern that signifies a potential bottom.
        
        Visual: A long bearish candle, followed by a small-bodied candle (star) that gaps 
        down, followed by a bullish candle that closes well into the first candle's body.
        
        Psychology: After sellers push prices down (first candle), uncertainty enters 
        the market (second candle/star), followed by strong buying pressure (third candle) 
        confirming a shift in sentiment.
        
        Significance: Considered a strong reversal signal, especially when the third 
        candle retraces deeply into the first candle, showing rejection of lower prices.
        """
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        return (
            first.close < first.open and
            abs(second.close - second.open) < abs(first.open - first.close) * 0.3 and
            third.close > third.open and
            third.close > (first.open + first.close) / 2
        )

    @staticmethod
    def is_piercing_pattern(df: pd.DataFrame) -> bool:
        """
        The Piercing Pattern is a two-candle bullish reversal pattern found in downtrends.
        
        Visual: A bearish candle followed by a bullish candle that opens below the 
        previous candle's low but closes above the midpoint of the previous candle.
        
        Psychology: After a downtrend, bears appear to maintain control with a bearish 
        candle. The next day opens even lower (bearish sentiment), but bulls take control 
        and push prices up significantly, showing a shift in momentum.
        
        Significance: Signals potential reversal of a downtrend, particularly when it 
        appears after a prolonged decline or at support levels.
        """
        prev, last = df.iloc[-2], df.iloc[-1]
        midpoint = (prev.open + prev.close) / 2
        return (
            prev.close < prev.open and
            last.open < prev.close and
            last.close > midpoint
        )

    @staticmethod
    def is_bullish_harami(df: pd.DataFrame) -> bool:
        """
        The Bullish Harami is a two-candle reversal pattern that appears during downtrends.
        
        Visual: A large bearish candle followed by a smaller bullish candle that is 
        completely contained within the body of the previous candle (like a pregnant woman, 
        which is what "harami" means in Japanese).
        
        Psychology: After strong selling pressure (first candle), a small bullish candle 
        indicates indecision and potential exhaustion of the downtrend. The contrast in 
        candle size suggests a weakening of the bearish momentum.
        
        Significance: Indicates potential reversal of the prevailing downtrend, especially 
        when confirmed by increased volume on the second day or subsequent bullish price action.
        """
        prev, last = df.iloc[-2], df.iloc[-1]
        return (
            prev.close < prev.open and
            last.open > prev.close and
            last.close < prev.open and
            last.close > last.open
        )

    @staticmethod
    def is_three_white_soldiers(df: pd.DataFrame) -> bool:
        """
        Three White Soldiers is a bullish reversal pattern consisting of three consecutive bullish candles.
        
        Visual: Three consecutive bullish candles, each opening within the body of the 
        previous candle and closing higher than the previous candle's high.
        
        Psychology: Demonstrates persistent buying pressure over three periods, with 
        bulls gaining confidence and momentum with each successive candle. Bears are 
        continuously overpowered by bulls.
        
        Significance: A strong bullish reversal signal that shows increasing conviction 
        among buyers. Most reliable when appearing after a downtrend or at support levels, 
        suggesting a potential trend change.
        """
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
        """
        The Inverted Hammer is a bullish reversal candlestick pattern that typically appears at the bottom of downtrends.
        
        Visual: A candle with a small body at the bottom and a long upper shadow (at least 
        twice the length of the body) with little to no lower shadow, resembling an upside-down hammer.
        
        Psychology: After a downtrend, bulls attempt to drive prices higher (long upper shadow), 
        but face selling pressure that pushes prices back down. However, the close near the 
        open suggests bears couldn't maintain complete control, hinting at weakening downward momentum.
        
        Significance: While not as strong as a regular hammer, it suggests potential buying interest 
        emerging and often precedes a trend reversal, especially when confirmed by a bullish candle 
        the following period.
        """
        row = df.iloc[-1]
        body = abs(row.close - row.open)
        upper_wick = row.high - max(row.open, row.close)
        lower_wick = min(row.open, row.close) - row.low
        return body > 0 and (upper_wick > 2 * body) and (lower_wick < body)

    @staticmethod
    def is_doji(df: pd.DataFrame) -> bool:
        """
        A Doji is a candlestick pattern where the opening and closing prices are virtually equal.
        
        Visual: A candlestick with a very small body (or no body) and upper and/or lower shadows 
        of varying length, resembling a cross or plus sign.
        
        Psychology: Represents market indecision where neither bulls nor bears gain control. 
        The session opens and closes at nearly the same level despite price fluctuations during the period.
        
        Significance: By itself, a Doji indicates equilibrium or indecision, but when appearing 
        after a strong trend or at support/resistance levels, it can signal potential reversal. 
        The longer the shadows, the more volatile the period was, showing greater uncertainty.
        """
        row = df.iloc[-1]
        return abs(row.open - row.close) <= (row.high - row.low) * 0.1

    @staticmethod
    def is_morning_doji_star(df: pd.DataFrame) -> bool:
        """
        The Morning Doji Star is a three-candle bullish reversal pattern similar to the Morning Star, 
        but with a Doji as the middle candle.
        
        Visual: A long bearish candle, followed by a Doji that gaps down, followed by a bullish 
        candle that closes well into the first candle's body.
        
        Psychology: After strong selling (first candle), the market enters indecision (Doji), 
        followed by strong buying pressure (third candle). The Doji represents the point where 
        selling pressure diminishes and buying pressure begins to build.
        
        Significance: Considered even more reliable than the regular Morning Star due to the 
        clear indecision signaled by the Doji. Strongly indicates a potential bottom and trend reversal.
        """
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
        """
        The Bullish Abandoned Baby is a rare but powerful three-candle reversal pattern.
        
        Visual: A bearish candle, followed by a Doji that gaps down (with no overlap in 
        price range with either the first or third candle), followed by a bullish candle 
        that gaps up from the Doji.
        
        Psychology: After a downtrend (first candle), a Doji forms with gaps on both sides, 
        representing complete indecision isolated from the previous trend. The subsequent 
        bullish candle (third) confirms a reversal in sentiment as buyers take control.
        
        Significance: One of the most reliable reversal patterns due to the complete isolation 
        of the middle Doji (the "abandoned baby"). Signals a strong shift from bearish to bullish momentum.
        """
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        return (
            first.close < first.open and
            abs(second.open - second.close) < (second.high - second.low) * 0.1 and
            second.low > first.high and
            third.open > second.high and third.close > third.open
        )

    @staticmethod
    def is_bullish_belt_hold(df: pd.DataFrame) -> bool:
        """
        The Bullish Belt Hold is a single-candle reversal pattern that appears in downtrends.
        
        Visual: A bullish candle that opens at or near the low of the day (little to no 
        lower shadow) and closes significantly higher, creating a strong bullish body.
        
        Psychology: After a downtrend, the day opens at a low point (showing bears still 
        in control), but bulls immediately take over and drive prices higher throughout the 
        session, closing near the high and showing strong buying pressure.
        
        Significance: Suggests a potential reversal of the downtrend, particularly when the 
        candle is long and appears after an extended decline. The lack of a lower shadow 
        indicates immediate rejection of lower prices.
        """
        row = df.iloc[-1]
        return (
            row.close > row.open and
            row.open == row.low and
            (row.close - row.open) > (row.high - row.close) * 0.5
        )

    @staticmethod
    def is_three_inside_up(df: pd.DataFrame) -> bool:
        """
        The Three Inside Up is a three-candle bullish reversal pattern that begins with a Bullish Harami.
        
        Visual: A large bearish candle, followed by a smaller bullish candle contained within 
        the first (a Bullish Harami), followed by a third bullish candle that closes above 
        the high of the second candle.
        
        Psychology: After a downtrend (first candle), the smaller bullish candle indicates 
        diminishing bearish momentum. The third candle confirms the reversal as bulls gain 
        control and push prices higher, breaking the previous pattern.
        
        Significance: More reliable than a simple Bullish Harami because the third candle 
        provides confirmation of the reversal. Signals a shift from a downtrend to a potential uptrend.
        """
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
        """
        The Rising Window (also called a Bullish Gap) is a two-candle continuation pattern in an uptrend.
        
        Visual: Two consecutive candles where the low of the second candle is higher than 
        the high of the first candle, creating a price gap or "window" between them.
        
        Psychology: During an uptrend, strong buying pressure causes the second day to open 
        above the previous day's high, indicating enthusiasm among buyers and unwillingness 
        to enter at lower prices.
        
        Significance: Confirms the strength of an existing uptrend and suggests continuation 
        of the bullish momentum. The gap often acts as a support level in future price movements.
        """
        prev, last = df.iloc[-2], df.iloc[-1]
        return prev.high < last.low
    
    @staticmethod
    def is_bearish_engulfing(df: pd.DataFrame) -> bool:
        """
        The Bearish Engulfing is a two-candle bearish reversal pattern seen during uptrends.

        Visual: A small bullish candle followed by a larger bearish candle that completely 
        engulfs the previous candle's body.

        Psychology: Signals that bears have taken control after a temporary bullish push. 
        The strength of the bearish move shows overwhelming selling pressure.

        Significance: Suggests a potential top or resistance level and may signal the beginning 
        of a downtrend.
        """
        prev, last = df.iloc[-2], df.iloc[-1]
        return (
            prev.close > prev.open and
            last.close < last.open and
            last.open > prev.close and
            last.close < prev.open
        )

    @staticmethod
    def is_evening_star(df: pd.DataFrame) -> bool:
        """
        The Evening Star is a three-candle bearish reversal pattern appearing at market tops.

        Visual: A large bullish candle, followed by a small-bodied candle (gap up), and a 
        large bearish candle that closes deep into the first candle's body.

        Psychology: Shows transition from bullish strength to indecision and then strong 
        bearish follow-through.

        Significance: Strong signal of trend reversal from bullish to bearish, especially 
        when supported by volume.
        """
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        return (
            first.close > first.open and
            abs(second.close - second.open) < abs(first.close - first.open) * 0.3 and
            third.close < third.open and
            third.close < (first.open + first.close) / 2
        )

    @staticmethod
    def is_three_black_crows(df: pd.DataFrame) -> bool:
        """
        The Three Black Crows is a strong bearish reversal pattern consisting of three 
        consecutive bearish candles.

        Visual: Each candle opens within the body of the previous and closes lower, 
        showing consistent selling.

        Psychology: Indicates sustained bearish pressure and confidence in the downtrend.

        Significance: Strong bearish signal when occurring after an uptrend or at resistance.
        """
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        return (
            first.close < first.open and
            second.close < second.open and
            third.close < third.open and
            second.open < first.open and
            third.open < second.open and
            second.close < first.close and
            third.close < second.close
        )

    @staticmethod
    def is_bearish_harami(df: pd.DataFrame) -> bool:
        """
        The Bearish Harami is a two-candle reversal pattern seen during uptrends.

        Visual: A large bullish candle followed by a smaller bearish candle that is 
        completely contained within the previous candle's body.

        Psychology: Suggests bullish momentum is stalling and bears may be gaining 
        control.

        Significance: Indicates possible reversal to a downtrend, especially with confirmation.
        """
        prev, last = df.iloc[-2], df.iloc[-1]
        return (
            prev.close > prev.open and
            last.open < prev.close and
            last.close > prev.open and
            last.close < last.open
        )

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
CandlestickPatterns.register_pattern("Bearish Engulfing", CandlestickPatterns.is_bearish_engulfing, min_rows=2)
CandlestickPatterns.register_pattern("Evening Star", CandlestickPatterns.is_evening_star, min_rows=3)
CandlestickPatterns.register_pattern("Three Black Crows", CandlestickPatterns.is_three_black_crows, min_rows=3)
CandlestickPatterns.register_pattern("Bearish Harami", CandlestickPatterns.is_bearish_harami, min_rows=2)
# --- End of patterns.py ---