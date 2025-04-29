"""Unit tests for risk management functionality"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from risk_manager import RiskManager, RiskParameters

@pytest.fixture
def risk_manager():
    return RiskManager(max_position_pct=0.25)

@pytest.fixture
def sample_ohlc_data():
    dates = pd.date_range(start='2025-01-01', periods=20)
    data = {
        'open': np.random.uniform(100, 110, 20),
        'high': np.random.uniform(105, 115, 20),
        'low': np.random.uniform(95, 105, 20),
        'close': np.random.uniform(100, 110, 20)
    }
    return pd.DataFrame(data, index=dates)

def test_position_size_calculation(risk_manager):
    params = RiskParameters(
        account_value=100000,
        risk_pct=0.01,
        entry_price=100.0,
        stop_loss_price=98.0
    )
    
    position = risk_manager.calculate_position_size(params)
    
    assert position.shares > 0
    assert position.risk_amount == 1000.0  # 1% of 100k
    assert position.max_loss <= 1000.0
    
def test_position_size_limits(risk_manager):
    params = RiskParameters(
        account_value=100000,
        risk_pct=0.05,  # 5% risk
        entry_price=100.0,
        stop_loss_price=95.0
    )
    
    position = risk_manager.calculate_position_size(params)
    
    # Should be limited by max_position_pct (25%)
    assert position.shares <= (100000 * 0.25) / 100.0

def test_atr_stop_loss(risk_manager, sample_ohlc_data):
    params = RiskParameters(
        account_value=100000,
        risk_pct=0.01,
        entry_price=100.0,
        atr_period=14,
        atr_multiplier=1.5
    )
    
    stop_price = risk_manager.calculate_atr_stop_loss(sample_ohlc_data, params)
    
    assert stop_price is not None
    assert stop_price < sample_ohlc_data['close'].iloc[-1]

def test_invalid_risk_parameters():
    with pytest.raises(ValueError):
        RiskParameters(
            account_value=-100000,
            risk_pct=0.15,  # > 10% max
            entry_price=100.0
        )

def test_order_validation(risk_manager):
    validation = risk_manager.validate_order(
        symbol="AAPL",
        price=150.0,
        quantity=100
    )
    assert validation["valid"] is True
    
    validation = risk_manager.validate_order(
        symbol="",
        price=-1,
        quantity=0
    )
    assert validation["valid"] is False
    assert len(validation["messages"]) == 3