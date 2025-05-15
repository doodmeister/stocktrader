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

    def evaluate(self, market_data=None):
        """
        Evaluate market data using technical indicators and return a composite signal.
        Returns a float in [-1, 1] (bearish to bullish).
        """
        df = market_data if market_data is not None else self.data
        if df is None or df.empty or not all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            return None

        # Pad if needed
        if len(df) < 10:
            last_price = df['close'].iloc[-1]
            padding_length = 10 - len(df)
            padding = pd.DataFrame({
                'close': [last_price] * padding_length,
                'high': [last_price] * padding_length,
                'low': [last_price] * padding_length,
                'volume': [df['volume'].iloc[-1]] * padding_length
            }, index=range(padding_length))
            df = pd.concat([padding, df]).reset_index(drop=True)

        # RSI
        try:
            rsi_value = self.rsi(period=3).iloc[-1]
        except Exception:
            rsi_value = 50

        if pd.isna(rsi_value):
            rsi_value = 50

        # MACD
        try:
            macd_line, signal_line = self.macd(3, 6, 3)
            macd_value = macd_line.iloc[-1] - signal_line.iloc[-1]
        except Exception:
            macd_value = 0

        if pd.isna(macd_value):
            macd_value = 0

        # Bollinger Bands
        try:
            upper_band, lower_band = self.bollinger_bands(3)
            current_price = df['close'].iloc[-1]
            band_range = upper_band.iloc[-1] - lower_band.iloc[-1]
            if band_range == 0 or pd.isna(band_range):
                bb_position = 0.5
            else:
                bb_position = (current_price - lower_band.iloc[-1]) / band_range
                bb_position = max(0, min(1, bb_position))
        except Exception:
            bb_position = 0.5

        # Normalize signals
        rsi_signal = (rsi_value - 50) / 50
        macd_signal = np.tanh(macd_value)
        bb_signal = 2 * (bb_position - 0.5)

        # Combine signals
        combined_signal = (rsi_signal + macd_signal + bb_signal) / 3
        return float(max(min(combined_signal, 1), -1))

    def calculate_atr(self):
        """Wrapper for ATR calculation."""
        try:
            atr_series = self.atr(period=3)
            if atr_series is None or pd.isna(atr_series.iloc[-1]):
                return None
            return float(atr_series.iloc[-1])
        except Exception:
            return None

    def calculate_price_target(self):
        """Calculate price target using multiple technical indicators."""
        try:
            if self.data is None or len(self.data) < 2:
                return None
            current_price = float(self.data['close'].iloc[-1])
            price_changes = self.data['close'].diff()
            trend_strength = price_changes.mean()
            trend_positive = trend_strength > 0
            price_std = self.data['close'].std()
            is_sideways = price_std < (current_price * 0.005)
            if is_sideways:
                return current_price
            momentum = price_changes.iloc[-1] if not pd.isna(price_changes.iloc[-1]) else 0
            strong_momentum = abs(momentum) > abs(trend_strength)
            atr = self.calculate_atr()
            if atr is None or atr == 0:
                atr = current_price * 0.02
            upper_band, lower_band = self.bollinger_bands(period=3)
            if trend_positive:
                last_prices = self.data['close'].iloc[-3:]
                if len(last_prices) >= 3 and all(last_prices.diff().dropna() > 0):
                    avg_increase = last_prices.diff().mean()
                    trend_target = current_price + (5 * avg_increase)
                else:
                    trend_target = current_price * 1.15
                volatility_target = current_price + (8 * atr)
                band_target = float(upper_band.iloc[-1]) + (5 * atr)
                target = max(trend_target, volatility_target, band_target)
                if strong_momentum:
                    target += (2 * atr)
                min_target = current_price * 1.20
                target = max(target, min_target)
            else:
                target = min(current_price - (2 * atr), float(lower_band.iloc[-1]))
            return float(target)
        except Exception as e:
            logger.error(f"Error calculating price target: {e}")
            return float(self.data['close'].iloc[-1]) * 1.20

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
