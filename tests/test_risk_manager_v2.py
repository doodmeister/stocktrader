import pytest
import pandas as pd
from core.risk_manager_v2 import RiskManager, RiskParameters, InvalidRiskConfig

@pytest.fixture
def ohlc_df():
    # Minimal OHLC data for ATR calculation
    data = {
        'high': [110, 112, 115, 117, 120],
        'low': [100, 105, 108, 110, 115],
        'close': [105, 110, 112, 115, 118]
    }
    return pd.DataFrame(data)

def test_position_size_with_stop_loss(ohlc_df):
    params = RiskParameters(
        account_value=10000,
        risk_pct=0.01,
        entry_price=100,
        stop_loss_price=95,
        trade_side='long'
    )
    rm = RiskManager(max_position_pct=0.2)
    pos = rm.calculate_position_size(params)
    assert pos.shares > 0
    assert pos.max_loss <= params.account_value * params.risk_pct

def test_position_size_with_atr(ohlc_df):
    params = RiskParameters(
        account_value=10000,
        risk_pct=0.01,
        entry_price=120,
        trade_side='short'
    )
    rm = RiskManager(max_position_pct=0.2)
    pos = rm.calculate_position_size(params, ohlc_df=ohlc_df)
    assert pos.shares > 0

def test_invalid_stop_loss():
    with pytest.raises(ValueError):
        RiskParameters(
            account_value=10000,
            risk_pct=0.01,
            entry_price=100,
            stop_loss_price=105,  # Invalid for long
            trade_side='long'
        )

def test_validate_order():
    rm = RiskManager(max_position_pct=0.2)
    assert rm.validate_order("AAPL", 10, "BUY", 10000, 100)
    assert not rm.validate_order("AAPL", 0, "BUY", 10000, 100)
    assert not rm.validate_order("AAPL", 10, "BUY", 10000, 0)
    assert not rm.validate_order("AAPL", 10, "HOLD", 10000, 100)
    assert not rm.validate_order("", 10, "BUY", 10000, 100)