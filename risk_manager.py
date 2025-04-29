# risk_manager.py
"""
Risk management module for position sizing, stop-loss, and take-profit calculations.

Functions:
    - position_size(account_value, risk_pct, entry_price, stop_loss_price) -> int
    - calculate_atr_stop_loss(df, period=14, multiplier=1.5) -> float
    - place_stop_order(symbol, stop_price, quantity)
    - place_take_profit_order(symbol, profit_price, quantity)
"""
import pandas as pd
from typing import Union

def position_size(
    account_value: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float
) -> int:
    """
    Calculate the position size (number of shares) based on fixed fractional risk.

    account_value: total account capital
    risk_pct: fraction of account to risk per trade (e.g., 0.01 for 1%)
    entry_price: price at which position is opened
    stop_loss_price: price at which to exit for max loss

    Returns: number of shares (integer)
    """
    risk_amount = account_value * risk_pct
    risk_per_share = abs(entry_price - stop_loss_price)
    size = risk_amount / risk_per_share if risk_per_share > 0 else 0
    return int(size)


def calculate_atr_stop_loss(
    df: pd.DataFrame,
    period: int = 14,
    multiplier: float = 1.5
) -> Union[float, None]:
    """
    Calculate stop-loss price based on ATR.
    df: DataFrame with columns ['high','low','close'] indexed by date
    period: ATR lookback
    multiplier: ATR multiplier for stop-loss distance

    Returns: stop_loss_price for the last bar or None if insufficient data
    """
    if len(df) < period + 1:
        return None
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    stop_price = df['close'].iloc[-1] - multiplier * atr
    return stop_price


def place_stop_order(symbol: str, stop_price: float, quantity: int):
    """
    Placeholder to integrate with broker API for stop-loss order.
    """
    # TODO: implement using ETradeClient or other broker client
    raise NotImplementedError("Broker stop order integration needed")


def place_take_profit_order(symbol: str, profit_price: float, quantity: int):
    """
    Placeholder to integrate with broker API for take-profit order.
    """
    # TODO: implement using ETradeClient or other broker client
    raise NotImplementedError("Broker take-profit order integration needed")
