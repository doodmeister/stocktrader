"""
Risk management module for E*Trade trading bot position sizing and order management.

Handles position sizing calculations, stop-loss/take-profit computations, and order placement
with proper risk controls and input validation.

Classes:
    RiskManager: Encapsulates risk management functionality with configurable parameters
"""
import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Union, Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)

class RiskParameters(BaseModel):
    """Validation model for risk management parameters"""
    account_value: float = Field(..., gt=0)
    risk_pct: float = Field(..., gt=0, le=0.1)  # Max 10% risk per trade
    entry_price: float = Field(..., gt=0)
    stop_loss_price: Optional[float] = Field(None)
    atr_period: int = Field(default=14, ge=5, le=50)
    atr_multiplier: float = Field(default=1.5, gt=0, le=5.0)
    
    @validator('stop_loss_price')
    def validate_stop_price(cls, v, values):
        if v and v >= values['entry_price']:
            raise ValueError("Stop loss must be below entry for long positions")
        return v

@dataclass
class PositionSize:
    """Position sizing calculation result"""
    shares: int
    risk_amount: float
    risk_per_share: float
    max_loss: float

class RiskManager:
    """Risk management system for position sizing and order placement"""
    
    def __init__(self, max_position_pct: float = 0.25):
        """
        Initialize risk manager with position limits.
        
        Args:
            max_position_pct: Maximum single position size as % of account
        """
        self.max_position_pct = max_position_pct
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate risk manager configuration"""
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("max_position_pct must be between 0 and 1")

    def calculate_position_size(self, params: RiskParameters) -> PositionSize:
        """
        Calculate position size based on account risk parameters.
        
        Args:
            params: Validated risk parameters
            
        Returns:
            PositionSize object with shares and risk metrics
            
        Raises:
            ValueError: If parameters are invalid
        """
        try:
            # Calculate initial position size
            risk_amount = Decimal(str(params.account_value)) * Decimal(str(params.risk_pct))
            risk_per_share = abs(Decimal(str(params.entry_price)) - 
                               Decimal(str(params.stop_loss_price or 0)))

            if risk_per_share <= 0:
                raise ValueError("Invalid risk per share")

            # Calculate shares with decimal precision
            shares = int((risk_amount / risk_per_share).quantize(Decimal('1'), 
                                                               rounding=ROUND_DOWN))

            # Apply position size limits
            max_shares = int((Decimal(str(params.account_value)) * 
                            Decimal(str(self.max_position_pct)) / 
                            Decimal(str(params.entry_price))).quantize(Decimal('1'),
                                                                     rounding=ROUND_DOWN))
            shares = min(shares, max_shares)

            return PositionSize(
                shares=shares,
                risk_amount=float(risk_amount),
                risk_per_share=float(risk_per_share),
                max_loss=float(risk_amount)
            )

        except (ValueError, ArithmeticError) as e:
            logger.error(f"Position size calculation failed: {str(e)}")
            raise ValueError(f"Position sizing failed: {str(e)}")

    def calculate_atr_stop_loss(
        self,
        df: pd.DataFrame,
        params: RiskParameters
    ) -> Optional[float]:
        """
        Calculate ATR-based stop loss price.
        
        Args:
            df: OHLC price data
            params: Risk parameters with ATR settings
            
        Returns:
            Stop loss price or None if calculation fails
        """
        try:
            if len(df) < params.atr_period + 1:
                logger.warning(f"Insufficient data for ATR ({len(df)} bars)")
                return None

            # Validate DataFrame columns
            required_cols = {'high', 'low', 'close'}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"DataFrame missing required columns: {required_cols}")

            # Calculate True Range and ATR
            tr = pd.DataFrame({
                'hl': df['high'] - df['low'],
                'hc': abs(df['high'] - df['close'].shift()),
                'lc': abs(df['low'] - df['close'].shift())
            }).max(axis=1)

            atr = tr.rolling(window=params.atr_period).mean().iloc[-1]
            
            if pd.isna(atr):
                logger.warning("ATR calculation resulted in NaN")
                return None

            stop_price = df['close'].iloc[-1] - (params.atr_multiplier * atr)
            
            return round(float(stop_price), 2)

        except Exception as e:
            logger.error(f"ATR stop loss calculation failed: {str(e)}")
            return None

    def validate_order(self, symbol: str, price: float, quantity: int) -> Dict:
        """
        Validate order parameters before submission.
        
        Args:
            symbol: Trading symbol
            price: Order price
            quantity: Order quantity
            
        Returns:
            Dict with validation status and messages
        """
        validation = {"valid": True, "messages": []}
        
        if not symbol or not isinstance(symbol, str):
            validation["valid"] = False
            validation["messages"].append("Invalid symbol")
            
        if not price or price <= 0:
            validation["valid"] = False
            validation["messages"].append("Invalid price")
            
        if not quantity or quantity <= 0:
            validation["valid"] = False
            validation["messages"].append("Invalid quantity")
            
        return validation
