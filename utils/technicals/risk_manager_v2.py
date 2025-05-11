from dataclasses import dataclass
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ValidationError
import pandas as pd
import math
from utils.logger import setup_logger

# Configure logger
logger = setup_logger(__name__)

class InvalidRiskConfig(Exception):
    """Custom exception for invalid risk configuration."""
    pass

class RiskParameters(BaseModel):
    account_value: float = Field(..., gt=0, description="Total account value, must be positive.")
    risk_pct: float = Field(..., gt=0, le=0.1, description="Risk per trade as a fraction (max 10%).")
    entry_price: float = Field(..., gt=0, description="Trade entry price, must be positive.")
    stop_loss_price: Optional[float] = Field(None, gt=0, description="Stop loss price, must be positive if set.")
    atr_period: int = Field(default=14, ge=5, le=50, description="ATR period for volatility-based stops.")
    atr_multiplier: float = Field(default=1.5, gt=0, le=5.0, description="ATR multiplier for stop calculation.")
    trade_side: Literal['long', 'short'] = Field(default='long', description="Trade direction: 'long' or 'short'.")

    @field_validator('stop_loss_price')
    def validate_stop_price(cls, v, info):
        if v is not None:
            entry_price = info.data.get('entry_price')
            trade_side = info.data.get('trade_side', 'long').lower()
            if entry_price is None:
                raise ValueError("entry_price is required to validate stop_loss_price")
            if trade_side == 'long' and v >= entry_price:
                raise ValueError("stop_loss_price must be below entry_price for longs")
            elif trade_side == 'short' and v <= entry_price:
                raise ValueError("stop_loss_price must be above entry_price for shorts")
        return v

@dataclass(frozen=True)
class PositionSize:
    shares: int
    risk_amount: float
    risk_per_share: float
    max_loss: float

class RiskManager:
    def __init__(self, max_position_pct: float = 0.25):
        if not (0 < max_position_pct <= 1):
            logger.error("max_position_pct must be between 0 and 1")
            raise ValueError("max_position_pct must be between 0 and 1")
        self.max_position_pct = max_position_pct
        logger.info(f"RiskManager initialized: max_position_pct={self.max_position_pct}")

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int) -> float:
        required_cols = {'high', 'low', 'close'}
        if not required_cols.issubset(df.columns):
            logger.error(f"DataFrame missing required columns: {required_cols - set(df.columns)}")
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        if len(df) < period:
            logger.warning("DataFrame length less than ATR period; ATR may be inaccurate.")
        df = df.copy()
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = (df['high'] - df['close'].shift(1)).abs()
        df['L-PC'] = (df['low'] - df['close'].shift(1)).abs()
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        atr = df['TR'].rolling(window=period, min_periods=1).mean().iloc[-1]
        logger.debug(f"Calculated ATR: {atr:.4f} (period={period})")
        return float(atr)

    def derive_stop_loss(self, params: RiskParameters, df: pd.DataFrame) -> float:
        atr = self.calculate_atr(df, params.atr_period)
        if params.trade_side == 'long':
            stop = params.entry_price - atr * params.atr_multiplier
        elif params.trade_side == 'short':
            stop = params.entry_price + atr * params.atr_multiplier
        else:
            logger.error(f"Invalid trade_side: {params.trade_side}")
            raise InvalidRiskConfig(f"Invalid trade_side: {params.trade_side}")
        stop = round(stop, 2)
        logger.info(f"Derived stop loss: {stop} (ATR={atr:.4f}, multiplier={params.atr_multiplier})")
        return stop

    def calculate_position_size(
        self, params: RiskParameters, ohlc_df: Optional[pd.DataFrame] = None
    ) -> PositionSize:
        try:
            if params.stop_loss_price is not None:
                stop_price = params.stop_loss_price
            elif ohlc_df is not None:
                stop_price = self.derive_stop_loss(params, ohlc_df)
            else:
                logger.error("Must provide either stop_loss_price or OHLC DataFrame for ATR-based stop.")
                raise InvalidRiskConfig("Must provide either stop_loss_price or OHLC DataFrame for ATR-based stop.")

            if params.trade_side == 'long':
                risk_per_share = params.entry_price - stop_price
            elif params.trade_side == 'short':
                risk_per_share = stop_price - params.entry_price
            else:
                logger.error(f"Unsupported trade side: {params.trade_side}")
                raise InvalidRiskConfig(f"Unsupported trade side: {params.trade_side}")

            if risk_per_share <= 0:
                logger.error("Risk per share must be positive")
                raise InvalidRiskConfig("Risk per share must be positive")

            risk_amount = params.account_value * params.risk_pct
            raw_shares = risk_amount / risk_per_share
            max_shares = math.floor((params.account_value * self.max_position_pct) / params.entry_price)
            shares = min(math.floor(raw_shares), max_shares)
            max_loss = shares * risk_per_share

            logger.info(
                f"Position sizing => shares={shares}, risk_amount={risk_amount:.2f}, "
                f"risk_per_share={risk_per_share:.2f}, max_loss={max_loss:.2f}"
            )
            return PositionSize(
                shares=shares,
                risk_amount=round(risk_amount, 2),
                risk_per_share=round(risk_per_share, 2),
                max_loss=round(max_loss, 2)
            )
        except Exception as e:
            logger.exception("Error calculating position size")
            raise

    def validate_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        account_value: float,
        entry_price: float
    ) -> bool:
        if not symbol or not isinstance(symbol, str):
            logger.warning("Invalid order: symbol must be a non-empty string")
            return False
        if quantity <= 0:
            logger.warning(f"Invalid order: quantity must be > 0 (got {quantity})")
            return False
        if entry_price <= 0 or account_value <= 0:
            logger.warning("Invalid order: entry_price and account_value must be > 0")
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
