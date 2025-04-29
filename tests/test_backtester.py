import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from backtester import Backtest, BacktestConfig, Trade, BacktestResults

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    data = {
        'AAPL': pd.DataFrame({
            'open': np.random.uniform(150, 160, len(dates)),
            'high': np.random.uniform(155, 165, len(dates)),
            'low': np.random.uniform(145, 155, len(dates)),
            'close': np.random.uniform(150, 160, len(dates)),
            'volume': np.random.uniform(1000000, 2000000, len(dates))
        }, index=dates)
    }
    return data

@pytest.fixture
def backtest_config():
    return BacktestConfig(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        risk_free_rate=0.02,
        position_size_limit=0.2
    )

def test_backtest_initialization(backtest_config):
    bt = Backtest(backtest_config)
    assert bt.capital == backtest_config.initial_capital
    assert len(bt.trades) == 0
    assert bt.positions == {}

def test_data_validation(sample_data, backtest_config):
    bt = Backtest(backtest_config)
    
    # Test valid data
    bt.validate_data(sample_data)
    
    # Test invalid data
    invalid_data = {'AAPL': pd.DataFrame({'close': [100, 101]})}
    with pytest.raises(ValueError):
        bt.validate_data(invalid_data)

def test_simple_strategy(sample_data, backtest_config):
    bt = Backtest(backtest_config)
    
    def simple_strategy(df):
        return 1 if len(df) > 0 else 0
    
    results = bt.simulate(sample_data, simple_strategy)
    assert isinstance(results, BacktestResults)
    assert results.num_trades > 0

def test_trade_validation():
    valid_trade = Trade(
        date=datetime.now(),
        symbol='AAPL',
        side='BUY',
        price=150.0,
        quantity=1
    )
    assert valid_trade

    with pytest.raises(ValueError):
        Trade(
            date=datetime.now(),
            symbol='AAPL',
            side='INVALID',
            price=150.0,
            quantity=1
        )

def test_export_results(sample_data, backtest_config, tmp_path):
    bt = Backtest(backtest_config)
    
    def simple_strategy(df):
        return 1 if len(df) > 0 else 0
    
    bt.simulate(sample_data, simple_strategy)
    bt.export_results(tmp_path)
    
    assert (tmp_path / 'equity_curve.csv').exists()
    assert (tmp_path / 'trades.csv').exists()