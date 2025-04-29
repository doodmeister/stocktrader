# backtester.py
"""
Offline backtesting framework for candlestick strategies.

Classes:
    Backtest:
        - load_history(symbol, start_date, end_date)
        - simulate(strategy_fn, initial_capital)
        - compute_metrics()
        - export_results(filepath)

Functions:
    sharpe_ratio(returns, risk_free_rate=0.0)
    max_drawdown(equity_curve)
"""
import pandas as pd
import numpy as np
from typing import Callable, Dict

class Backtest:
    def __init__(self, data: Dict[str, pd.DataFrame], initial_capital: float = 100000.0):
        """
        data: mapping symbol -> historical OHLCV DataFrame with DatetimeIndex
        initial_capital: starting cash balance
        """
        self.data = data
        self.initial_capital = initial_capital
        self.equity_curve = pd.DataFrame()
        self.trades = []  # list of executed trades

    def simulate(self, strategy_fn: Callable[[pd.DataFrame], pd.Series]):
        """
        Run backtest using a strategy function.
        strategy_fn should accept a DataFrame and return a Series of signals:
            1 for BUY, -1 for SELL, 0 for HOLD.
        """
        # Initialize
        capital = self.initial_capital
        positions = {sym: 0 for sym in self.data}
        equity = []

        # Align dates across symbols
        dates = pd.date_range(
            start=min(df.index.min() for df in self.data.values()),
            end=max(df.index.max() for df in self.data.values()),
            freq='D'
        )

        for date in dates:
            daily_value = capital
            # generate signals per symbol
            for sym, df in self.data.items():
                if date in df.index:
                    signal = strategy_fn(df.loc[:date])
                    price = df.loc[date]['close']
                    if signal == 1 and capital >= price:
                        # buy one share
                        positions[sym] += 1
                        capital -= price
                        self.trades.append((date, sym, 'BUY', price))
                    elif signal == -1 and positions[sym] > 0:
                        # sell one share
                        positions[sym] -= 1
                        capital += price
                        self.trades.append((date, sym, 'SELL', price))
                    # update daily_value
                    daily_value += positions[sym] * price
            equity.append({'date': date, 'equity': daily_value})

        self.equity_curve = pd.DataFrame(equity).set_index('date')

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute performance metrics: Sharpe ratio, max drawdown, total return.
        """
        returns = self.equity_curve['equity'].pct_change().dropna()
        total_return = (self.equity_curve['equity'].iloc[-1] / self.initial_capital) - 1
        sharpe = sharpe_ratio(returns)
        mdd = max_drawdown(self.equity_curve['equity'])
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': mdd
        }

    def export_results(self, filepath: str):
        """
        Export equity curve and trades to CSV files.
        """
        self.equity_curve.to_csv(f"{filepath}_equity.csv")
        trades_df = pd.DataFrame(self.trades, columns=['date','symbol','side','price'])
        trades_df.to_csv(f"{filepath}_trades.csv", index=False)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate annualized Sharpe ratio.
    """
    excess = returns - risk_free_rate/252
    return np.sqrt(252) * excess.mean() / excess.std()


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate the maximum drawdown.
    """
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max
    return drawdown.min()
