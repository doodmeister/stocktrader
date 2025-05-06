import pandas as pd
import numpy as np

class TechnicalAnalysis:
    """
    Provides technical indicator calculations and composite signal evaluation for financial time series.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    def sma(self, period=20):
        """Simple Moving Average."""
        return self.data['close'].rolling(window=period).mean()

    def ema(self, period=20):
        """Exponential Moving Average."""
        return self.data['close'].ewm(span=period, adjust=False).mean()

    def rsi(self, period=14):
        """Relative Strength Index."""
        delta = self.data['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean().replace(0, np.finfo(float).eps)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi.iloc[:period] = np.nan
        return rsi

    def macd(self, fast_period=12, slow_period=26, signal_period=9):
        """Moving Average Convergence Divergence."""
        fast_ema = self.data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = self.data['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line, signal_line

    def bollinger_bands(self, period=20, std_dev=2):
        """Bollinger Bands."""
        sma = self.data['close'].rolling(window=period).mean()
        std = self.data['close'].rolling(window=period).std().replace(0, self.data['close'].mean() * 0.0001)
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, lower_band

    def atr(self, period=14):
        """Average True Range."""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close'].shift()
        tr = pd.concat([
            high - low,
            abs(high - close),
            abs(low - close)
        ], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

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
        rsi_value = self.rsi(period=3).iloc[-1]
        if pd.isna(rsi_value):
            rsi_value = 50

        # MACD
        macd_line, signal_line = self.macd(3, 6, 3)
        macd_value = macd_line.iloc[-1] - signal_line.iloc[-1]
        if pd.isna(macd_value):
            macd_value = 0

        # Bollinger Bands
        upper_band, lower_band = self.bollinger_bands(3)
        current_price = df['close'].iloc[-1]
        band_range = upper_band.iloc[-1] - lower_band.iloc[-1]
        if band_range == 0 or pd.isna(band_range):
            bb_position = 0.5
        else:
            bb_position = (current_price - lower_band.iloc[-1]) / band_range
            bb_position = max(0, min(1, bb_position))

        # Normalize signals
        rsi_signal = (rsi_value - 50) / 50
        macd_signal = np.tanh(macd_value)
        bb_signal = 2 * (bb_position - 0.5)

        # Combine signals
        combined_signal = (rsi_signal + macd_signal + bb_signal) / 3
        return float(max(min(combined_signal, 1), -1))

    def calculate_atr(self, symbol=None):
        """Wrapper for ATR calculation."""
        atr_series = self.atr(period=3)
        if atr_series is None or pd.isna(atr_series.iloc[-1]):
            return None
        return float(atr_series.iloc[-1])

    def calculate_price_target(self, symbol=None):
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
            atr = self.calculate_atr(symbol)
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
            print(f"Error calculating price target: {e}")
            return current_price * 1.20

    def main():
        # Example usage of the TechnicalAnalysis class
        sample_data = pd.DataFrame({
            'close': [100, 102, 104, 103, 105],
            'high': [101, 103, 105, 104, 106],
            'low': [99, 101, 103, 102, 104],
            'volume': [1000, 1100, 1200, 1150, 1300]
        })
        ta = TechnicalAnalysis(sample_data)
        print("RSI:", ta.rsi())
        print("MACD:", ta.macd())
        print("Bollinger Bands:", ta.bollinger_bands())
        print("ATR:", ta.atr())
        print("Composite Signal:", ta.evaluate())
    
    if __name__ == "__main__":
        main()
