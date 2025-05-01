"""
Offline backtesting framework for candlestick trading strategies.

This module provides a robust framework for backtesting trading strategies
against historical market data, with comprehensive performance metrics and
risk analysis capabilities.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single trade execution."""
    date: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: int
    fees: float = 0.0

    def __post_init__(self):
        if self.side not in ['BUY', 'SELL']:
            raise ValueError("Trade side must be either 'BUY' or 'SELL'")
        if self.price <= 0:
            raise ValueError("Trade price must be positive")
        if self.quantity <= 0:
            raise ValueError("Trade quantity must be positive")

class BacktestConfig(BaseModel):
    """Configuration parameters for backtest execution."""
    initial_capital: float = Field(gt=0)
    commission_rate: float = Field(ge=0, le=0.01, default=0.001)
    slippage_rate: float = Field(ge=0, le=0.01, default=0.0005)
    risk_free_rate: float = Field(ge=0, le=0.1, default=0.02)
    position_size_limit: float = Field(ge=0, le=1.0, default=0.2)

    class Config:
        arbitrary_types_allowed = True

class BacktestResults(BaseModel):
    """Container for backtest performance metrics."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float

class Backtest:
    """Backtesting engine for trading strategies."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.equity_curve: pd.Series = pd.Series()
        self.trades: List[Trade] = []
        self.positions: Dict[str, int] = {}
        self.capital: float = self.config.initial_capital

    def validate_data(self, data: Dict[str, pd.DataFrame]) -> None:
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        for symbol, df in data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"DataFrame for {symbol} must have DatetimeIndex")
            if not required_columns.issubset(df.columns):
                raise ValueError(f"DataFrame for {symbol} missing required columns")
            if df.isnull().any().any():
                logger.warning(f"Found NaN values in data for {symbol}")

    def simulate(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_fn: Callable[[pd.DataFrame], float]
    ) -> BacktestResults:
        try:
            self.validate_data(data)
            self.reset()
            self.positions = {sym: 0 for sym in data}

            dates = pd.date_range(
                start=min(df.index.min() for df in data.values()),
                end=max(df.index.max() for df in data.values()),
                freq='D'
            )

            equity_history = []

            for date in dates:
                daily_value = self.capital
                for symbol, df in data.items():
                    if date in df.index:
                        try:
                            signal = strategy_fn(df.loc[:date])
                            self._process_signal(date, symbol, signal, df.loc[date])
                            current_price = df.loc[date]['close']
                            daily_value += self.positions[symbol] * current_price
                        except Exception as e:
                            logger.error(f"Error processing {symbol} on {date}: {e}")
                            continue
                equity_history.append({'date': date, 'equity': daily_value})

            self.equity_curve = pd.DataFrame(equity_history).set_index('date')['equity']
            return self._compute_metrics()

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise

    def _process_signal(
        self,
        date: datetime,
        symbol: str,
        signal: float,
        market_data: pd.Series
    ) -> None:
        price = market_data['close']
        available_capital = self.capital * self.config.position_size_limit

        if signal > 0 and self.capital >= price:
            max_shares = int(available_capital / price)
            shares = min(max_shares, 1)
            cost = shares * price * (1 + self.config.commission_rate + self.config.slippage_rate)
            if cost <= self.capital:
                self.positions[symbol] += shares
                self.capital -= cost
                self.trades.append(Trade(date, symbol, 'BUY', price, shares))
        elif signal < 0 and self.positions[symbol] > 0:
            shares = self.positions[symbol]
            proceeds = shares * price * (1 - self.config.commission_rate - self.config.slippage_rate)
            self.positions[symbol] = 0
            self.capital += proceeds
            self.trades.append(Trade(date, symbol, 'SELL', price, shares))

    def _compute_metrics(self) -> BacktestResults:
        returns = self.equity_curve.pct_change().dropna()
        if len(returns) == 0:
            raise ValueError("No trades executed during backtest period")

        profitable_trades = sum(
            1 for t in self.trades if t.side == 'SELL' and 
            any(bt.side == 'BUY' and bt.symbol == t.symbol and bt.price < t.price for bt in self.trades)
        )
        total_trades = len([t for t in self.trades if t.side == 'SELL'])

        return BacktestResults(
            total_return=(self.equity_curve.iloc[-1] / self.config.initial_capital) - 1,
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            max_drawdown=self._calculate_max_drawdown(),
            win_rate=profitable_trades / total_trades if total_trades > 0 else 0,
            profit_factor=self._calculate_profit_factor(),
            num_trades=total_trades,
            avg_trade_return=np.mean(returns) if len(returns) > 0 else 0
        )

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        excess_returns = returns - self.config.risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def _calculate_max_drawdown(self) -> float:
        peak = self.equity_curve.expanding(min_periods=1).max()
        drawdown = (self.equity_curve - peak) / peak
        return drawdown.min()

    def _calculate_profit_factor(self) -> float:
        returns = pd.Series([t.price for t in self.trades if t.side == 'SELL'])
        costs = pd.Series([t.price for t in self.trades if t.side == 'BUY'])
        if len(returns) == 0 or len(costs) == 0:
            return 0.0
        profits = (returns - costs).clip(lower=0).sum()
        losses = abs((returns - costs).clip(upper=0).sum())
        return profits / losses if losses != 0 else float('inf')

    def export_results(self, output_dir: Path) -> None:
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.equity_curve.to_csv(output_dir / 'equity_curve.csv')
            trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
            trades_df.to_csv(output_dir / 'trades.csv', index=False)
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            raise

# Strategy examples and data loaders

def load_ohlcv(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq='B')
    np.random.seed(hash(symbol) % 2**32)
    price = np.cumsum(np.random.randn(len(dates))) + 100
    df = pd.DataFrame({
        'open': price + np.random.uniform(-1, 1, len(dates)),
        'high': price + np.random.uniform(0, 2, len(dates)),
        'low': price - np.random.uniform(0, 2, len(dates)),
        'close': price + np.random.uniform(-1, 1, len(dates)),
        'volume': np.random.randint(100000, 200000, len(dates))
    }, index=dates)
    df.index.name = 'date'
    return df


def moving_average_crossover_strategy(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> float:
    if len(df) < slow:
        return 0
    fast_ma = df['close'].rolling(window=fast).mean()
    slow_ma = df['close'].rolling(window=slow).mean()
    if fast_ma.iloc[-2] < slow_ma.iloc[-2] and fast_ma.iloc[-1] > slow_ma.iloc[-1]:
        return 1
    elif fast_ma.iloc[-2] > slow_ma.iloc[-2] and fast_ma.iloc[-1] < slow_ma.iloc[-1]:
        return -1
    return 0


def rsi_strategy(df: pd.DataFrame, period: int = 14, overbought: float = 70, oversold: float = 30) -> float:
    if len(df) < period + 1:
        return 0
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    if rsi.iloc[-1] < oversold:
        return 1
    elif rsi.iloc[-1] > overbought:
        return -1
    return 0


def macd_strategy(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
    if len(df) < slow + signal:
        return 0
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    if macd.iloc[-2] < macd_signal.iloc[-2] and macd.iloc[-1] > macd_signal.iloc[-1]:
        return 1
    elif macd.iloc[-2] > macd_signal.iloc[-2] and macd.iloc[-1] < macd_signal.iloc[-1]:
        return -1
    return 0


def run_backtest(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    strategy: str,
    initial_capital: float,
    risk_per_trade: float
) -> Dict[str, Any]:
    df = load_ohlcv(symbol, start_date, end_date)
    data = {symbol: df}
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.0005,
        risk_free_rate=0.02,
        position_size_limit=risk_per_trade / 100.0
    )
    bt = Backtest(config)
    if strategy == "Moving Average Crossover":
        strategy_fn = moving_average_crossover_strategy
    elif strategy == "RSI Strategy":
        strategy_fn = rsi_strategy
    elif strategy == "MACD Strategy":
        strategy_fn = macd_strategy
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    results = bt.simulate(data, strategy_fn)
    equity_curve = bt.equity_curve.reset_index().rename(columns={'index': 'date'})
    trades = pd.DataFrame([t.__dict__ for t in bt.trades])
    metrics = results.dict()
    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trade_log": trades
    }

# Convenience alias for Streamlit pages

def run_backtest_wrapper(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    strategy: str,
    initial_capital: float,
    risk_per_trade: float
) -> Dict[str, Any]:
    """
    Wrapper that matches Streamlit page import.
    """
    return run_backtest(symbol, start_date, end_date, strategy, initial_capital, risk_per_trade)
