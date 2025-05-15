import pandas as pd
import numpy as np
from utils.logger import setup_logger
from utils.technicals.indicators import (
    add_rsi, add_macd, add_bollinger_bands, add_atr, add_sma, add_ema,
    IndicatorError, validate_dataframe
)

logger = setup_logger(__name__)

class TechnicalAnalysis:
    """
    Provides technical indicator calculations and composite signal evaluation for financial time series.
    All indicator methods return pandas Series for easy integration.
    """

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.data = data.copy()

    def sma(self, period=20, close_col='close'):
        """Simple Moving Average."""
        try:
            validate_dataframe(self.data, [close_col])
            return self.data[close_col].rolling(window=period, min_periods=1).mean()
        except Exception as e:
            logger.error(f"SMA calculation failed: {e}")
            raise IndicatorError(f"Failed to calculate SMA: {e}") from e

    def ema(self, period=20, close_col='close'):
        """Exponential Moving Average."""
        try:
            validate_dataframe(self.data, [close_col])
            return self.data[close_col].ewm(span=period, adjust=False).mean()
        except Exception as e:
            logger.error(f"EMA calculation failed: {e}")
            raise IndicatorError(f"Failed to calculate EMA: {e}") from e

    def rsi(self, period=14, close_col='close'):
        """Relative Strength Index."""
        try:
            df = add_rsi(self.data, length=period, close_col=close_col)
            return df['rsi']
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            raise

    def macd(self, fast_period=12, slow_period=26, signal_period=9, close_col='close'):
        """Moving Average Convergence Divergence."""
        try:
            df = add_macd(self.data, fast=fast_period, slow=slow_period, signal=signal_period, close_col=close_col)
            return df['macd'], df['macd_signal']
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            raise

    def bollinger_bands(self, period=20, std_dev=2, close_col='close'):
        """Bollinger Bands."""
        try:
            df = add_bollinger_bands(self.data, length=period, std=std_dev, close_col=close_col)
            return df['bb_upper'], df['bb_lower']
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            raise

    def atr(self, period=14):
        """Average True Range."""
        try:
            df = add_atr(self.data, length=period)
            return df['atr']
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            raise

    def evaluate(
        self,
        market_data=None,
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        bb_period=20,
        bb_std=2
    ):
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
            rsi = self.rsi(period=rsi_period)
            macd, macd_sig = self.macd(
                fast_period=macd_fast,
                slow_period=macd_slow,
                signal_period=macd_signal
            )
            bb_upper, bb_lower = self.bollinger_bands(
                period=bb_period,
                std_dev=bb_std
            )
            close = df['close']

            # RSI Score
            rsi_score = ((rsi.iloc[-1] - 50) / 50) if not pd.isna(rsi.iloc[-1]) else 0

            # MACD Score (normalized and clamped)
            macd_diff = macd.iloc[-1] - macd_sig.iloc[-1]
            macd_range = max(abs(macd.max()), abs(macd.min()), 1e-6)
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

    def calculate_atr(self):
        """Wrapper for ATR calculation."""
        try:
            atr_series = self.atr(period=3)
            if atr_series is None or pd.isna(atr_series.iloc[-1]):
                return None
            return float(atr_series.iloc[-1])
        except Exception:
            return None

    def calculate_price_target_fib(
        self,
        lookback: int = 20,
        extension: float = 0.618
    ) -> float:
        """
        Fibonacci‐extension price target:
          - Finds the highest high and lowest low over the past `lookback` bars.
          - Projects the target = swing_high + ( (high–low) * extension ).
        """
        if len(self.data) < lookback:
            return float(self.data['close'].iloc[-1])

        recent     = self.data.iloc[-lookback:]
        swing_high = recent['high'].max()
        swing_low  = recent['low'].min()
        diff       = swing_high - swing_low
        return float(swing_high + diff * extension)

    def calculate_price_target(self):
        # now just use our Fibonacci‐extension routine
        return self.calculate_price_target_fib(lookback=30, extension=0.618)

    @staticmethod
    def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all standard technical indicators and composite signal to the DataFrame.
        """
        try:
            df = add_rsi(df)
            df = add_macd(df)
            df = add_bollinger_bands(df)
            df = add_atr(df)
            df = add_sma(df)
            df = add_ema(df)
            df = df.copy()
            # Optionally add composite signal and price targets
            from utils.technicals.indicators import add_composite_signal, calculate_price_target
            df = add_composite_signal(df)
            df = calculate_price_target(df)
            return df
        except Exception as e:
            logger.error(f"Failed to enrich DataFrame with technical indicators: {e}")
            return df
