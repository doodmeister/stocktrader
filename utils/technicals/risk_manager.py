from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field, field_validator  # For Pydantic v2
import pandas as pd
import math
from utils.logger import setup_logger

# Configure logger
logger = setup_logger(__name__)

class RiskParameters(BaseModel):
    """
    Validation model for risk management parameters.
    """
    account_value: float = Field(..., gt=0)
    risk_pct: float = Field(..., gt=0, le=0.1)  # Max 10% risk per trade
    entry_price: float = Field(..., gt=0)
    stop_loss_price: Optional[float] = Field(None, gt=0)
    atr_period: int = Field(default=14, ge=5, le=50)
    atr_multiplier: float = Field(default=1.5, gt=0, le=5.0)

    @field_validator('stop_loss_price')
    def validate_stop_price(cls, v, info):
        if v is not None:
            entry_price = info.data.get('entry_price')
            if entry_price is None:
                raise ValueError("entry_price is required to validate stop_loss_price")
            
            if v >= entry_price:
                raise ValueError("stop_loss_price must be below entry_price for longs")
        return v

@dataclass
class PositionSize:
    """Position sizing calculation result."""
    shares: int
    risk_amount: float
    risk_per_share: float
    max_loss: float

class RiskManager:
    """
    Risk management system for position sizing and order validation.
    """
    def __init__(self, max_position_pct: float = 0.25):
        if not (0 < max_position_pct <= 1):
            raise ValueError("max_position_pct must be between 0 and 1")
        self.max_position_pct = max_position_pct
        logger.info(f"RiskManager initialized: max_position_pct={self.max_position_pct}")

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int) -> float:
        """
        Calculate Average True Range (ATR) from OHLC DataFrame.
        """
        df = df.copy()
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = (df['high'] - df['close'].shift(1)).abs()
        df['L-PC'] = (df['low'] - df['close'].shift(1)).abs()
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        atr = df['TR'].rolling(window=period, min_periods=1).mean().iloc[-1]
        logger.debug(f"Calculated ATR={atr} over period={period}")
        return float(atr)

    def get_stop_loss_price(self, entry_price: float, atr: float, multiplier: Optional[float] = None) -> float:
        """
        Compute stop loss price based on entry, ATR, and multiplier.
        """
        m = multiplier if multiplier is not None else 1.0
        stop_price = entry_price - atr * m
        logger.debug(f"Computed stop_loss_price={stop_price} (entry={entry_price}, atr={atr}, mult={m})")
        return round(stop_price, 2)

    def calculate_position_size(self, params: RiskParameters) -> PositionSize:
        """
        Calculate optimal position size based on risk parameters and manager config.
        """
        # Determine stop loss
        if params.stop_loss_price is not None:
            stop_price = params.stop_loss_price
        else:
            # Placeholder: user must supply ATR DataFrame externally if needed
            raise ValueError("stop_loss_price required when ATR DataFrame not provided")

        risk_per_share = params.entry_price - stop_price
        if risk_per_share <= 0:
            raise ValueError("Risk per share must be positive")

        risk_amount = params.account_value * params.risk_pct
        raw_shares = risk_amount / risk_per_share

        # Cap by max_position_pct
        max_shares = math.floor((params.account_value * self.max_position_pct) / params.entry_price)
        shares = min(math.floor(raw_shares), max_shares)
        max_loss = shares * risk_per_share

        logger.info(
            f"Position sizing => shares={shares}, risk_amount={risk_amount:.2f},"
            f" risk_per_share={risk_per_share:.2f}, max_loss={max_loss:.2f}"
        )
        return PositionSize(
            shares=shares,
            risk_amount=round(risk_amount, 2),
            risk_per_share=round(risk_per_share, 2),
            max_loss=round(max_loss, 2)
        )

    def validate_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        account_value: float,
        entry_price: float
    ) -> bool:
        """
        Validate order parameters before execution.
        """
        if quantity <= 0:
            logger.warning(f"Invalid order: quantity must be > 0 (got {quantity})")
            return False

        position_value = quantity * entry_price
        if position_value > account_value * self.max_position_pct:
            logger.warning(
                f"Invalid order: position_value={position_value:.2f} exceeds max "
                f"{self.max_position_pct * 100:.0f}% of account"
            )
            return False

        if side.upper() not in ("BUY", "SELL"):
            logger.warning(f"Invalid order: side must be BUY or SELL (got {side})")
            return False

        logger.debug(f"Order validated for {symbol}: quantity={quantity}, side={side}")
        return True