import pytest
import pandas as pd
import numpy as np
from utils.etrade_candlestick_bot import CandlestickPatterns
from utils.backtester import Backtest

# --- CandlestickPatterns Tests ---

def test_is_hammer_true():
    s = pd.Series({'open': 10.0, 'high': 10.3, 'low': 8.5, 'close': 10.2})
    assert CandlestickPatterns.is_hammer(s)

def test_is_hammer_false():
    s = pd.Series({'open': 10.0, 'high': 12.0, 'low': 9.5, 'close': 11.0})
    assert not CandlestickPatterns.is_hammer(s)


def test_bullish_engulfing_true():
    prev = pd.Series({'open': 12.0, 'high': 12.5, 'low': 11.5, 'close': 10.0})  # bearish
    curr = pd.Series({'open': 9.0, 'high': 13.0, 'low': 8.5, 'close': 13.5})     # bullish engulfing
    assert CandlestickPatterns.is_bullish_engulfing(prev, curr)

def test_bullish_engulfing_false():
    prev = pd.Series({'open': 10.0, 'high': 11.0, 'low': 9.5, 'close': 11.5})
    curr = pd.Series({'open': 11.6, 'high': 12.0, 'low': 11.0, 'close': 11.4})
    assert not CandlestickPatterns.is_bullish_engulfing(prev, curr)


def test_bearish_engulfing_true():
    prev = pd.Series({'open': 10.0, 'high': 11.0, 'low': 9.5, 'close': 12.0})  # bullish
    curr = pd.Series({'open': 13.0, 'high': 13.5, 'low': 8.5, 'close': 9.0})   # bearish engulfing
    assert CandlestickPatterns.is_bearish_engulfing(prev, curr)


def test_bearish_engulfing_false():
    prev = pd.Series({'open': 10.0, 'high': 11.0, 'low': 9.5, 'close': 11.0})
    curr = pd.Series({'open': 11.1, 'high': 11.5, 'low': 11.0, 'close': 11.2})
    assert not CandlestickPatterns.is_bearish_engulfing(prev, curr)


def test_doji_true():
    s = pd.Series({'open': 10.0, 'high': 11.0, 'low': 9.0, 'close': 10.05})
    assert CandlestickPatterns.is_doji(s, threshold=0.1)


def test_doji_false():
    s = pd.Series({'open': 10.0, 'high': 11.0, 'low': 9.0, 'close': 10.3})
    assert not CandlestickPatterns.is_doji(s, threshold=0.1)

# --- Backtester Tests ---

def test_backtester_no_trades():
    dates = pd.date_range('2025-01-01', periods=3, freq='D')
    data = pd.DataFrame(
        index=dates,
        data={
            'open': [100, 100, 100],
            'high': [100, 100, 100],
            'low': [100, 100, 100],
            'close': [100, 100, 100],
            'volume': [1, 1, 1]
        }
    )
    bt = Backtest({'SYM': data}, initial_capital=1000.0)
    # strategy that never trades
    bt.simulate(lambda df: 0)
    metrics = bt.compute_metrics()
    assert metrics['total_return'] == 0.0
    assert metrics['max_drawdown'] == 0.0

# --- ML Pipeline Tests Placeholder ---
# You can add tests for prepare_dataset() and train_and_evaluate()

if __name__ == '__main__':
    pytest.main()
