"""
Unit tests for the neural network backtesting module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock

from pages.nn_backtest import (
    BacktestParams,
    generate_sma_crossover_signals,
    generate_pattern_nn_signals,
    generate_classic_ml_signals,
    run_strategy_backtest
)

class TestBacktestParams:
    """Test cases for BacktestParams validation model."""
    
    def test_valid_params(self):
        """Test valid parameter creation."""
        params = BacktestParams(
            symbol="AAPL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            strategy="SMA Crossover",
            initial_capital=100000,
            risk_per_trade=2.0
        )
        assert params.symbol == "AAPL"
        assert params.strategy == "SMA Crossover"
    
    def test_invalid_symbol(self):
        """Test invalid symbol validation."""
        with pytest.raises(ValueError):
            BacktestParams(
                symbol="",
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                strategy="SMA Crossover",
                initial_capital=100000,
                risk_per_trade=2.0
            )
    
    def test_invalid_date_range(self):
        """Test invalid date range validation."""
        with pytest.raises(ValueError):
            BacktestParams(
                symbol="AAPL",
                start_date=date(2023, 12, 31),
                end_date=date(2023, 1, 1),
                strategy="SMA Crossover",
                initial_capital=100000,
                risk_per_trade=2.0
            )
    
    def test_invalid_capital(self):
        """Test invalid capital validation."""
        with pytest.raises(ValueError):
            BacktestParams(
                symbol="AAPL",
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                strategy="SMA Crossover",
                initial_capital=500,  # Below minimum
                risk_per_trade=2.0
            )

class TestSignalGeneration:
    """Test cases for signal generation functions."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        data = {
            'open': 100 + np.random.randn(len(dates)) * 2,
            'high': 102 + np.random.randn(len(dates)) * 2,
            'low': 98 + np.random.randn(len(dates)) * 2,
            'close': 100 + np.random.randn(len(dates)) * 2,
            'volume': 1000000 + np.random.randint(-100000, 100000, len(dates))
        }
        
        df = pd.DataFrame(data, index=dates)
        # Ensure high >= max(open, close) and low <= min(open, close)
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        return df
    
    def test_sma_crossover_signals(self, sample_ohlcv_data):
        """Test SMA crossover signal generation."""
        signals = generate_sma_crossover_signals(sample_ohlcv_data, fast_period=5, slow_period=10)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlcv_data)
        assert signals.dtype == int
        assert all(signal in [-1, 0, 1] for signal in signals)
    
    def test_sma_crossover_empty_data(self):
        """Test SMA crossover with empty data."""
        empty_df = pd.DataFrame()
        signals = generate_sma_crossover_signals(empty_df)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == 0 or all(signals == 0)
    
    @patch('pages.nn_backtest.load_pattern_nn_cached')
    def test_pattern_nn_signals(self, mock_load_model, sample_ohlcv_data):
        """Test PatternNN signal generation."""
        # Mock model
        mock_model = Mock()
        mock_model.return_value = Mock()
        mock_model.return_value.__getitem__ = Mock(return_value=Mock())
        mock_load_model.return_value = mock_model
        
        # Mock torch operations
        with patch('torch.tensor') as mock_tensor, \
             patch('torch.argmax') as mock_argmax:
            
            mock_tensor.return_value = Mock()
            mock_argmax.return_value = Mock()
            mock_argmax.return_value.item.return_value = 1
            
            signals = generate_pattern_nn_signals(sample_ohlcv_data, "fake_model.pth")
            
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(sample_ohlcv_data)
    
    @patch('pages.nn_backtest.load_classic_model_cached')
    def test_classic_ml_signals(self, mock_load_model, sample_ohlcv_data):
        """Test classic ML signal generation."""
        # Mock model with feature_names_in_
        mock_model = Mock()
        mock_model.feature_names_in_ = ['close', 'volume']
        mock_model.predict.return_value = np.array([1, 0, -1] * (len(sample_ohlcv_data) // 3 + 1))[:len(sample_ohlcv_data)]
        mock_load_model.return_value = mock_model
        
        signals = generate_classic_ml_signals(sample_ohlcv_data, "fake_model.joblib")
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlcv_data)
        assert all(signal in [-1, 0, 1] for signal in signals)

class TestBacktestExecution:
    """Test cases for backtest execution."""
    
    @pytest.fixture
    def sample_params(self):
        """Create sample backtest parameters."""
        return BacktestParams(
            symbol="AAPL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            strategy="SMA Crossover",
            initial_capital=100000,
            risk_per_trade=2.0
        )
    
    @patch('pages.nn_backtest.load_ohlcv')
    @patch('pages.nn_backtest.Backtest')
    def test_successful_backtest(self, mock_backtest_class, mock_load_ohlcv, sample_params):
        """Test successful backtest execution."""
        # Mock OHLCV data
        mock_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })
        mock_load_ohlcv.return_value = mock_df
        
        # Mock backtest instance
        mock_backtest = Mock()
        mock_backtest.simulate.return_value = Mock()
        mock_backtest.simulate.return_value.dict.return_value = {
            'total_return': 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.02,
            'num_trades': 5
        }
        mock_backtest.trades = []
        mock_backtest.equity_curve = pd.DataFrame({
            'equity': [100000, 102000, 105000]
        })
        mock_backtest_class.return_value = mock_backtest
        
        results = run_strategy_backtest(sample_params)
        
        assert results is not None
        assert 'metrics' in results
        assert 'equity_curve' in results
        assert 'trade_log' in results
        assert results['strategy'] == 'SMA Crossover'
    
    @patch('pages.nn_backtest.load_ohlcv')
    def test_backtest_no_data(self, mock_load_ohlcv, sample_params):
        """Test backtest with no data."""
        mock_load_ohlcv.return_value = pd.DataFrame()
        
        results = run_strategy_backtest(sample_params)
        
        assert results is None

# Integration test (requires actual models and data)
@pytest.mark.integration
class TestIntegration:
    """Integration tests that require actual models and data."""
    
    def test_full_backtest_workflow(self):
        """Test complete backtest workflow with real data."""
        # This test would require actual model files and data
        # Skip if running in CI/CD environment
        pytest.skip("Integration test requires actual models and data")

if __name__ == "__main__":
    pytest.main([__file__])