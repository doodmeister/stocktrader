"""
Enhanced Risk Management System for E*Trade Trading Bot

This module provides comprehensive risk management capabilities including:
- Position sizing based on risk percentage and ATR
- Stop-loss calculation using technical indicators
- Portfolio exposure limits and validation
- Trade validation with comprehensive checks

Key Features:
- Pydantic-based parameter validation
- ATR-based stop loss calculation
- Position size optimization
- Comprehensive logging and error handling
- Thread-safe operations
- Environment-based configuration
- Enhanced type safety with Literal, TypeGuard, and NewType
- Structured position data types
"""

import os
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Union, TypeGuard, NewType, TypedDict, List
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field, field_validator, ConfigDict
import pandas as pd
import numpy as np
import math
import threading
from contextlib import contextmanager
from datetime import datetime
from dotenv import load_dotenv
from utils.logger import setup_logger

# Load environment variables
load_dotenv()

# Configure logger
logger = setup_logger(__name__)

# Enhanced Type Definitions
TradeSide = Literal['long', 'short']
OrderSide = Literal['BUY', 'SELL']
ATRMethod = Literal['sma', 'ema']
StopLossMethod = Literal['atr', 'percentage', 'support_resistance']
RiskLevel = Literal['LOW', 'MEDIUM', 'HIGH']
PositionStatus = Literal['OPEN', 'CLOSED', 'PARTIAL']

# NewType for better type safety
AccountValue = NewType('AccountValue', float)
RiskPercentage = NewType('RiskPercentage', float)
Price = NewType('Price', float)
Shares = NewType('Shares', int)
DollarAmount = NewType('DollarAmount', float)

# Structured Position Data Types
class PositionDict(TypedDict):
    """TypedDict for position data in dictionaries."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    value: float
    unrealized_pnl: float
    side: str
    entry_date: str

class MarketDataDict(TypedDict, total=False):
    """TypedDict for market data with optional fields."""
    volume: float
    spread: float
    bid: float
    ask: float
    last_price: float
    avg_volume: Optional[float]
    volatility: Optional[float]

@dataclass(frozen=True)
class Position:
    """
    Immutable position data structure with comprehensive details.
    
    This replaces Dict[str, float] usage and provides type safety.
    """
    symbol: str
    quantity: Shares
    entry_price: Price
    current_price: Price
    side: TradeSide
    entry_date: datetime
    stop_loss_price: Optional[Price] = None
    take_profit_price: Optional[Price] = None
    position_id: Optional[str] = None
    status: PositionStatus = 'OPEN'
    
    def __post_init__(self):
        """Validate position data."""
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if self.current_price <= 0:
            raise ValueError("Current price must be positive")
    
    @property
    def value(self) -> DollarAmount:
        """Calculate current position value."""
        return DollarAmount(self.quantity * self.current_price)
    
    @property
    def cost_basis(self) -> DollarAmount:
        """Calculate original cost basis."""
        return DollarAmount(self.quantity * self.entry_price)
    
    @property
    def unrealized_pnl(self) -> DollarAmount:
        """Calculate unrealized profit/loss."""
        if self.side == 'long':
            return DollarAmount((self.current_price - self.entry_price) * self.quantity)
        else:  # short position
            return DollarAmount((self.entry_price - self.current_price) * self.quantity)
    
    @property
    def unrealized_pnl_percent(self) -> RiskPercentage:
        """Calculate unrealized P&L as percentage."""
        return RiskPercentage(self.unrealized_pnl / self.cost_basis)
    
    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable."""
        return self.unrealized_pnl > 0
    
    @property
    def risk_amount(self) -> DollarAmount:
        """Calculate current risk amount if stop loss is set."""
        if self.stop_loss_price is None:
            return DollarAmount(0.0)
        
        if self.side == 'long':
            risk_per_share = self.current_price - self.stop_loss_price
        else:
            risk_per_share = self.stop_loss_price - self.current_price
            
        return DollarAmount(max(0, risk_per_share * self.quantity))
    
    def to_dict(self) -> PositionDict:
        """Convert to dictionary format for backward compatibility."""
        return PositionDict(
            symbol=self.symbol,
            quantity=self.quantity,
            entry_price=float(self.entry_price),
            current_price=float(self.current_price),
            value=float(self.value),
            unrealized_pnl=float(self.unrealized_pnl),
            side=self.side,
            entry_date=self.entry_date.isoformat()
        )
    
    @classmethod
    def from_dict(cls, data: PositionDict) -> 'Position':
        """Create Position from dictionary data."""
        return cls(
            symbol=data['symbol'],
            quantity=Shares(data['quantity']),
            entry_price=Price(data['entry_price']),
            current_price=Price(data['current_price']),
            side=data['side'],  # type: ignore
            entry_date=datetime.fromisoformat(data['entry_date'])
        )

@dataclass(frozen=True)
class Portfolio:
    """
    Immutable portfolio data structure containing multiple positions.
    """
    positions: List[Position]
    account_value: AccountValue
    cash_balance: DollarAmount
    daily_pnl: DollarAmount = DollarAmount(0.0)
    
    @property
    def total_position_value(self) -> DollarAmount:
        """Calculate total value of all positions."""
        return DollarAmount(sum(pos.value for pos in self.positions))
    
    @property
    def total_unrealized_pnl(self) -> DollarAmount:
        """Calculate total unrealized P&L."""
        return DollarAmount(sum(pos.unrealized_pnl for pos in self.positions))
    
    @property
    def total_cost_basis(self) -> DollarAmount:
        """Calculate total cost basis of all positions."""
        return DollarAmount(sum(pos.cost_basis for pos in self.positions))
    
    @property
    def cash_percentage(self) -> RiskPercentage:
        """Calculate cash as percentage of account value."""
        return RiskPercentage(self.cash_balance / self.account_value)
    
    @property
    def position_count(self) -> int:
        """Get number of open positions."""
        return len([pos for pos in self.positions if pos.status == 'OPEN'])
    
    @property
    def largest_position_pct(self) -> RiskPercentage:
        """Get largest position as percentage of account value."""
        if not self.positions:
            return RiskPercentage(0.0)
        max_value = max(pos.value for pos in self.positions)
        return RiskPercentage(max_value / self.account_value)
    
    @property
    def symbols(self) -> List[str]:
        """Get list of symbols in portfolio."""
        return [pos.symbol for pos in self.positions]
    
    @property
    def long_positions(self) -> List[Position]:
        """Get all long positions."""
        return [pos for pos in self.positions if pos.side == 'long']
    
    @property
    def short_positions(self) -> List[Position]:
        """Get all short positions."""
        return [pos for pos in self.positions if pos.side == 'short']
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        for pos in self.positions:
            if pos.symbol.upper() == symbol.upper():
                return pos
        return None
    
    def get_position_value_pct(self, symbol: str) -> RiskPercentage:
        """Get position value as percentage of account value."""
        position = self.get_position(symbol)
        if position is None:
            return RiskPercentage(0.0)
        return RiskPercentage(position.value / self.account_value)
    
    def to_legacy_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to legacy dictionary format for backward compatibility."""
        return {
            pos.symbol: {
                'quantity': float(pos.quantity),
                'entry_price': float(pos.entry_price),
                'current_price': float(pos.current_price),
                'value': float(pos.value),
                'unrealized_pnl': float(pos.unrealized_pnl)
            }
            for pos in self.positions
        }

class RiskManagerError(Exception):
    """Base exception for risk management errors."""
    pass

class InvalidRiskConfig(RiskManagerError):
    """Exception for invalid risk configuration."""
    pass

class InsufficientDataError(RiskManagerError):
    """Exception for insufficient market data."""
    pass

class PositionSizeError(RiskManagerError):
    """Exception for position sizing errors."""
    pass

def is_valid_trade_side(side: str) -> TypeGuard[TradeSide]:
    """Type guard to check if a string is a valid trade side."""
    return side.lower() in ('long', 'short')

def is_valid_order_side(side: str) -> TypeGuard[OrderSide]:
    """Type guard to check if a string is a valid order side."""
    return side.upper() in ('BUY', 'SELL')

def normalize_trade_side(side: str) -> TradeSide:
    """Normalize and validate trade side input."""
    normalized = side.lower().strip()
    if not is_valid_trade_side(normalized):
        raise ValueError(f"Invalid trade side: {side}. Must be 'long' or 'short'")
    return normalized  # type: ignore

def normalize_order_side(side: str) -> OrderSide:
    """Normalize and validate order side input."""
    normalized = side.upper().strip()
    if not is_valid_order_side(normalized):
        raise ValueError(f"Invalid order side: {side}. Must be 'BUY' or 'SELL'")
    return normalized  # type: ignore

def validate_price(price: float, min_price: float = 0.01) -> Price:
    """Validate and return a Price type."""
    if price < min_price:
        raise ValueError(f"Price ${price:.4f} is below minimum ${min_price}")
    return Price(price)

def validate_account_value(value: float, min_value: float = 1000.0) -> AccountValue:
    """Validate and return an AccountValue type."""
    if value < min_value:
        raise ValueError(f"Account value ${value:,.2f} is below minimum ${min_value:,.2f}")
    return AccountValue(value)

def validate_risk_percentage(pct: float, max_pct: float = 0.1) -> RiskPercentage:
    """Validate and return a RiskPercentage type."""
    if not (0 < pct <= max_pct):
        raise ValueError(f"Risk percentage {pct*100:.1f}% must be between 0% and {max_pct*100:.0f}%")
    return RiskPercentage(pct)

class RiskConfigManager:
    """Centralized configuration manager for risk parameters."""
    
    @staticmethod
    def get_max_positions() -> int:
        """Get maximum positions from environment."""
        return int(os.getenv('MAX_POSITIONS', 5))
    
    @staticmethod
    def get_max_loss_percent() -> RiskPercentage:
        """Get maximum loss percentage from environment."""
        return RiskPercentage(float(os.getenv('MAX_LOSS_PERCENT', 0.02)))
    
    @staticmethod
    def get_profit_target_percent() -> RiskPercentage:
        """Get profit target percentage from environment."""
        return RiskPercentage(float(os.getenv('PROFIT_TARGET_PERCENT', 0.03)))
    
    @staticmethod
    def get_max_daily_loss() -> RiskPercentage:
        """Get maximum daily loss from environment."""
        return RiskPercentage(float(os.getenv('MAX_DAILY_LOSS', 0.05)))
    
    @staticmethod
    def get_max_order_value() -> DollarAmount:
        """Get maximum order value from environment."""
        return DollarAmount(float(os.getenv('MAX_ORDER_VALUE', 10000.0)))
    
    @staticmethod
    def get_max_position_pct() -> RiskPercentage:
        """Get maximum position percentage from environment."""
        return RiskPercentage(float(os.getenv('MAX_POSITION_PCT', 0.25)))
    
    @staticmethod
    def get_max_correlation_exposure() -> RiskPercentage:
        """Get maximum correlation exposure."""
        return RiskPercentage(float(os.getenv('MAX_CORRELATION_EXPOSURE', 0.50)))
    
    @staticmethod
    def get_min_liquidity_threshold() -> DollarAmount:
        """Get minimum liquidity threshold."""
        return DollarAmount(float(os.getenv('MIN_LIQUIDITY_THRESHOLD', 1000000.0)))
    
    @staticmethod
    def get_default_atr_period() -> int:
        """Get default ATR period."""
        return int(os.getenv('DEFAULT_ATR_PERIOD', 14))
    
    @staticmethod
    def get_default_atr_multiplier() -> float:
        """Get default ATR multiplier."""
        return float(os.getenv('DEFAULT_ATR_MULTIPLIER', 1.5))

class RiskParameters(BaseModel):
    """
    Validated risk parameters for position sizing and stop-loss calculation.
    
    All monetary values should be in the same currency as the trading account.
    Risk percentage is expressed as a decimal (0.02 = 2%).
    """
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=True,
        str_strip_whitespace=True,
        extra='forbid'
    )
    
    account_value: AccountValue = Field(
        ..., 
        gt=0, 
        description="Total account value in base currency"
    )
    risk_pct: RiskPercentage = Field(
        ..., 
        gt=0, 
        le=0.1, 
        description="Risk per trade as decimal (max 10%)"
    )
    entry_price: Price = Field(
        ..., 
        gt=0, 
        description="Planned entry price for the trade"
    )
    stop_loss_price: Optional[Price] = Field(
        None, 
        gt=0, 
        description="Manual stop loss price (optional)"
    )
    atr_period: int = Field(
        default_factory=RiskConfigManager.get_default_atr_period, 
        ge=5, 
        le=50, 
        description="ATR lookback period for volatility calculation"
    )
    atr_multiplier: float = Field(
        default_factory=RiskConfigManager.get_default_atr_multiplier, 
        gt=0, 
        le=5.0, 
        description="ATR multiplier for stop distance"
    )
    trade_side: TradeSide = Field(
        default='long', 
        description="Trade direction"
    )
    max_position_value: Optional[DollarAmount] = Field(
        None, 
        gt=0, 
        description="Maximum position value override"
    )

    @field_validator('account_value', mode='before')
    def validate_account_value_field(cls, v: float) -> AccountValue:
        """Validate account value using type-safe validator."""
        return validate_account_value(v)

    @field_validator('risk_pct', mode='before')
    def validate_risk_percentage_field(cls, v: float) -> RiskPercentage:
        """Validate risk percentage using type-safe validator."""
        max_loss_percent = RiskConfigManager.get_max_loss_percent()
        validated_pct = validate_risk_percentage(v, max_pct=0.1)
        
        if v > max_loss_percent:
            logger.warning(
                f"Risk percentage ({v*100:.1f}%) exceeds configured maximum "
                f"({max_loss_percent*100:.1f}%)"
            )
        if v > 0.05:  # 5% warning threshold
            logger.warning(f"High risk percentage detected: {v*100:.1f}%")
        
        return validated_pct

    @field_validator('entry_price', mode='before')
    def validate_entry_price_field(cls, v: float) -> Price:
        """Validate entry price using type-safe validator."""
        price = validate_price(v)
        if v > 50000:  # Maximum $50,000 per share (sanity check)
            logger.warning(f"High entry price detected: ${v:,.2f}")
        return price

    @field_validator('stop_loss_price', mode='before')
    def validate_stop_price(cls, v: Optional[float], info) -> Optional[Price]:
        """Validate stop loss price is logically correct for trade direction."""
        if v is None:
            return v
            
        # Validate as Price type first
        stop_price = validate_price(v)
        
        # Access other field values from info.data
        entry_price = info.data.get('entry_price') if info.data else None
        trade_side = info.data.get('trade_side', 'long') if info.data else 'long'
        
        if entry_price is None:
            raise ValueError("entry_price is required when stop_loss_price is provided")
        
        # Normalize trade side
        if isinstance(trade_side, str):
            trade_side = normalize_trade_side(trade_side)
            
        if trade_side == 'long' and stop_price >= entry_price:
            raise ValueError(
                f"Stop loss price (${stop_price}) must be below entry price (${entry_price}) for long trades"
            )
        elif trade_side == 'short' and stop_price <= entry_price:
            raise ValueError(
                f"Stop loss price (${stop_price}) must be above entry price (${entry_price}) for short trades"
            )
            
        return stop_price

    @field_validator('atr_period')
    def validate_atr_period(cls, v: int) -> int:
        """Validate ATR period is reasonable."""
        if v < 5:
            raise ValueError("ATR period must be at least 5 for statistical significance")
        if v > 50:
            raise ValueError("ATR period should not exceed 50 for responsiveness")
        return v

    @field_validator('atr_multiplier')
    def validate_atr_multiplier(cls, v: float) -> float:
        """Validate ATR multiplier is reasonable."""
        if v < 0.5:
            raise ValueError("ATR multiplier should be at least 0.5 to allow for volatility")
        if v > 5.0:
            raise ValueError("ATR multiplier should not exceed 5.0 to prevent excessive stops")
        return v

    @field_validator('trade_side', mode='before')
    def validate_trade_side_field(cls, v: str) -> TradeSide:
        """Normalize and validate trade direction using type guard."""
        return normalize_trade_side(v)

    @field_validator('max_position_value', mode='before')
    def validate_max_position_value(cls, v: Optional[float], info) -> Optional[DollarAmount]:
        """Validate maximum position value override."""
        if v is None:
            return v
        
        if v < 100:
            raise ValueError("Max position value should be at least $100")
        if v > 1_000_000:
            logger.warning(f"Large max position value: ${v:,.2f}")
            
        max_pos_value = DollarAmount(v)
        
        # Check against account value if available
        account_value = info.data.get('account_value') if info.data else None
        if account_value and max_pos_value > account_value:
            raise ValueError(
                f"Max position value ${max_pos_value:,.2f} cannot exceed account value ${account_value:,.2f}"
            )
            
        return max_pos_value

@dataclass(frozen=True)
class PositionSize:
    """
    Immutable position sizing result with comprehensive metrics.
    """
    shares: Shares
    risk_amount: DollarAmount
    risk_per_share: Price
    max_loss: DollarAmount
    position_value: DollarAmount
    stop_loss_price: Price
    risk_reward_ratio: Optional[float] = None
    meets_daily_limit: bool = True
    meets_order_limit: bool = True
    
    def __post_init__(self):
        """Validate position size metrics with type safety."""
        if self.shares < 0:
            raise ValueError("Shares must be non-negative")
        if self.risk_amount < 0:
            raise ValueError("Risk amount must be non-negative")
        if self.position_value < 0:
            raise ValueError("Position value must be non-negative")
        if self.risk_per_share < 0:
            raise ValueError("Risk per share must be non-negative")

    @property
    def is_viable(self) -> bool:
        """Check if position is viable for trading."""
        return (
            self.shares > 0 and 
            self.meets_daily_limit and 
            self.meets_order_limit and
            self.risk_amount > 0
        )

    def get_position_percentage(self, account_value: AccountValue) -> RiskPercentage:
        """Calculate position as percentage of account value."""
        return RiskPercentage((self.position_value / account_value))

class RiskManager:
    """
    Thread-safe risk management system with enhanced type safety.
    """
    
    def __init__(
        self, 
        max_position_pct: Optional[RiskPercentage] = None,
        max_correlation_exposure: Optional[RiskPercentage] = None,
        min_liquidity_threshold: Optional[DollarAmount] = None,
        load_from_env: bool = True
    ):
        """Initialize risk manager with portfolio constraints."""
        if load_from_env:
            self.max_position_pct = max_position_pct or RiskConfigManager.get_max_position_pct()
            self.max_correlation_exposure = max_correlation_exposure or RiskConfigManager.get_max_correlation_exposure()
            self.min_liquidity_threshold = min_liquidity_threshold or RiskConfigManager.get_min_liquidity_threshold()
            self.max_positions = RiskConfigManager.get_max_positions()
            self.max_daily_loss = RiskConfigManager.get_max_daily_loss()
            self.max_order_value = RiskConfigManager.get_max_order_value()
        else:
            self.max_position_pct = max_position_pct or RiskPercentage(0.25)
            self.max_correlation_exposure = max_correlation_exposure or RiskPercentage(0.50)
            self.min_liquidity_threshold = min_liquidity_threshold or DollarAmount(1000000.0)
            self.max_positions = 5
            self.max_daily_loss = RiskPercentage(0.05)
            self.max_order_value = DollarAmount(10000.0)
            
        # Validation
        self._validate_configuration()
        self._lock = threading.RLock()
        
        logger.info(
            f"RiskManager initialized from {'environment' if load_from_env else 'parameters'}: "
            f"max_position_pct={self.max_position_pct}, "
            f"max_correlation_exposure={self.max_correlation_exposure}, "
            f"min_liquidity_threshold=${self.min_liquidity_threshold:,.0f}, "
            f"max_positions={self.max_positions}, "
            f"max_daily_loss={self.max_daily_loss*100:.1f}%, "
            f"max_order_value=${self.max_order_value:,.0f}"
        )

    def _validate_configuration(self) -> None:
        """Validate risk manager configuration with type safety."""
        if not (0 < self.max_position_pct <= 1):
            raise ValueError(f"max_position_pct must be between 0 and 1, got {self.max_position_pct}")
        if not (0 < self.max_correlation_exposure <= 1):
            raise ValueError(f"max_correlation_exposure must be between 0 and 1, got {self.max_correlation_exposure}")
        if self.min_liquidity_threshold < 0:
            raise ValueError(f"min_liquidity_threshold must be non-negative, got {self.min_liquidity_threshold}")
        if self.max_positions <= 0:
            raise ValueError(f"max_positions must be positive, got {self.max_positions}")
        if not (0 < self.max_daily_loss <= 1):
            raise ValueError(f"max_daily_loss must be between 0 and 1, got {self.max_daily_loss}")
        if self.max_order_value <= 0:
            raise ValueError(f"max_order_value must be positive, got {self.max_order_value}")

    @staticmethod
    def calculate_atr(
        df: pd.DataFrame, 
        period: int, 
        method: ATRMethod = 'sma'
    ) -> float:
        """Calculate Average True Range with type-safe method parameter."""
        required_cols = {'high', 'low', 'close'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
            
        if len(df) < 2:
            raise InsufficientDataError(
                f"Need at least 2 rows for ATR calculation, got {len(df)}"
            )
            
        if len(df) < period:
            logger.warning(
                f"DataFrame length ({len(df)}) less than ATR period ({period}). "
                "ATR calculation may be less reliable."
            )
            
        try:
            df_calc = df.copy()
            
            # Handle potential NaN values
            df_calc = df_calc.dropna(subset=['high', 'low', 'close'])
            if len(df_calc) < 2:
                raise InsufficientDataError("Insufficient non-NaN data for ATR calculation")
            
            # Calculate True Range components
            df_calc['h_l'] = df_calc['high'] - df_calc['low']
            df_calc['h_pc'] = np.abs(df_calc['high'] - df_calc['close'].shift(1))
            df_calc['l_pc'] = np.abs(df_calc['low'] - df_calc['close'].shift(1))
            
            # True Range is the maximum of the three components
            df_calc['true_range'] = df_calc[['h_l', 'h_pc', 'l_pc']].max(axis=1)
            
            # Calculate ATR using specified method
            if method == 'ema':
                atr_series = df_calc['true_range'].ewm(span=period, adjust=False).mean()
            else:  # method == 'sma'
                atr_series = df_calc['true_range'].rolling(
                    window=period, 
                    min_periods=min(period, len(df_calc))
                ).mean()
            
            atr = atr_series.iloc[-1]
            
            if pd.isna(atr) or atr <= 0:
                raise InsufficientDataError("ATR calculation resulted in invalid value")
                
            logger.debug(
                f"Calculated ATR: {atr:.4f} (period={period}, method={method}, "
                f"data_points={len(df_calc)})"
            )
            return float(atr)
            
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            raise InsufficientDataError(f"ATR calculation failed: {e}") from e

    def derive_stop_loss(
        self, 
        params: RiskParameters, 
        df: pd.DataFrame,
        method: StopLossMethod = 'atr'
    ) -> Price:
        """Calculate stop loss price using specified method with type safety."""
        try:
            if method == 'atr':
                atr = self.calculate_atr(df, params.atr_period)
                
                if params.trade_side == 'long':
                    stop_loss = params.entry_price - (atr * params.atr_multiplier)
                elif params.trade_side == 'short':
                    stop_loss = params.entry_price + (atr * params.atr_multiplier)
                else:
                    raise InvalidRiskConfig(f"Invalid trade_side: {params.trade_side}")
                    
            elif method == 'percentage':
                stop_pct = self.max_daily_loss / 2  # Conservative: half of daily limit per trade
                if params.trade_side == 'long':
                    stop_loss = params.entry_price * (1 - stop_pct)
                else:
                    stop_loss = params.entry_price * (1 + stop_pct)
                    
            else:
                raise InvalidRiskConfig(f"Unsupported stop loss method: {method}")
            
            # Validate and return as Price type
            validated_stop = validate_price(stop_loss)
            
            # Additional validation
            max_stop_distance = params.entry_price * 0.15  # 15% max stop distance
            actual_distance = abs(validated_stop - params.entry_price)
            
            if actual_distance > max_stop_distance:
                logger.warning(
                    f"Stop loss distance ({actual_distance:.2f}) exceeds 15% of entry price. "
                    f"Consider adjusting parameters."
                )
            
            logger.info(
                f"Derived stop loss: ${validated_stop} (method={method}, "
                f"distance=${actual_distance:.2f}, {actual_distance/params.entry_price*100:.1f}%)"
            )
            return validated_stop
            
        except Exception as e:
            logger.error(f"Stop loss calculation failed: {e}")
            raise InvalidRiskConfig(f"Stop loss calculation failed: {e}") from e

    @contextmanager
    def _thread_safe_calculation(self):
        """Context manager for thread-safe calculations."""
        with self._lock:
            try:
                yield
            except Exception as e:
                logger.error(f"Thread-safe calculation failed: {e}")
                raise

    def calculate_position_size(
        self, 
        params: RiskParameters, 
        ohlc_df: Optional[pd.DataFrame] = None,
        target_price: Optional[Price] = None,
        current_daily_loss: DollarAmount = DollarAmount(0.0)
    ) -> PositionSize:
        """Calculate optimal position size with type-safe parameters."""
        with self._thread_safe_calculation():
            try:
                # Determine stop loss price
                if params.stop_loss_price is not None:
                    stop_price = params.stop_loss_price
                    logger.debug(f"Using manual stop loss: ${stop_price}")
                elif ohlc_df is not None:
                    stop_price = self.derive_stop_loss(params, ohlc_df)
                    logger.debug(f"Calculated ATR-based stop loss: ${stop_price}")
                else:
                    raise InvalidRiskConfig(
                        "Must provide either stop_loss_price or OHLC DataFrame for stop calculation"
                    )

                # Calculate risk per share
                if params.trade_side == 'long':
                    risk_per_share = params.entry_price - stop_price
                elif params.trade_side == 'short':
                    risk_per_share = stop_price - params.entry_price
                else:
                    raise InvalidRiskConfig(f"Invalid trade_side: {params.trade_side}")

                if risk_per_share <= 0:
                    raise PositionSizeError(
                        f"Risk per share must be positive. Entry: ${params.entry_price}, "
                        f"Stop: ${stop_price}, Side: {params.trade_side}"
                    )

                # Calculate position size based on risk
                risk_amount = DollarAmount(params.account_value * params.risk_pct)
                raw_shares = risk_amount / risk_per_share
                
                # Apply position size constraints
                max_position_value = params.max_position_value or DollarAmount(
                    params.account_value * self.max_position_pct
                )
                
                # Consider max_order_value from configuration
                max_position_value = DollarAmount(min(max_position_value, self.max_order_value))
                
                max_shares_by_value = int(max_position_value / params.entry_price)
                
                # Final share count (always round down for safety)
                shares = Shares(min(int(raw_shares), max_shares_by_value))
                
                # Check daily loss limits
                actual_risk = DollarAmount(shares * risk_per_share)
                meets_daily_limit = (current_daily_loss + actual_risk) <= DollarAmount(params.account_value * self.max_daily_loss)
                
                # Check order value limits
                position_value = DollarAmount(shares * params.entry_price)
                meets_order_limit = position_value <= self.max_order_value
                
                # Adjust shares if limits exceeded
                if not meets_daily_limit:
                    remaining_daily_risk = DollarAmount((params.account_value * self.max_daily_loss) - current_daily_loss)
                    shares = Shares(min(shares, int(max(0, remaining_daily_risk) / risk_per_share)))
                    logger.warning(f"Position size reduced to respect daily loss limit")
                
                if not meets_order_limit:
                    shares = Shares(min(shares, int(self.max_order_value / params.entry_price)))
                    logger.warning(f"Position size reduced to respect order value limit")
                
                # Ensure non-negative shares
                shares = Shares(max(0, shares))
                
                # Recalculate final metrics
                actual_risk = DollarAmount(shares * risk_per_share)
                position_value = DollarAmount(shares * params.entry_price)
                
                # Calculate risk/reward ratio if target provided
                risk_reward_ratio = None
                if target_price is not None and shares > 0:
                    if params.trade_side == 'long':
                        reward_per_share = target_price - params.entry_price
                    else:
                        reward_per_share = params.entry_price - target_price
                        
                    if reward_per_share > 0:
                        risk_reward_ratio = reward_per_share / risk_per_share

                result = PositionSize(
                    shares=shares,
                    risk_amount=DollarAmount(round(actual_risk, 2)),
                    risk_per_share=Price(round(risk_per_share, 4)),
                    max_loss=DollarAmount(round(actual_risk, 2)),
                    position_value=DollarAmount(round(position_value, 2)),
                    stop_loss_price=Price(round(stop_price, 2)),
                    risk_reward_ratio=round(risk_reward_ratio, 2) if risk_reward_ratio else None,
                    meets_daily_limit=(current_daily_loss + actual_risk) <= DollarAmount(params.account_value * self.max_daily_loss),
                    meets_order_limit=position_value <= self.max_order_value
                )

                if shares <= 0:
                    logger.warning(
                        f"Calculated position size is 0 shares. Risk too small or limits exceeded. "
                        f"Raw shares: {raw_shares:.2f}, Max by value: {max_shares_by_value}, "
                        f"Daily limit check: {result.meets_daily_limit}, Order limit check: {result.meets_order_limit}"
                    )
                else:
                    logger.info(
                        f"Position sizing complete: {shares} shares, "
                        f"${actual_risk:.2f} risk, ${position_value:.2f} value, "
                        f"RR: {risk_reward_ratio:.2f if risk_reward_ratio else 'N/A'}, "
                        f"Daily limit OK: {result.meets_daily_limit}, "
                        f"Order limit OK: {result.meets_order_limit}"
                    )
                
                return result
                
            except (InvalidRiskConfig, PositionSizeError):
                raise
            except Exception as e:
                logger.exception("Unexpected error in position size calculation")
                raise PositionSizeError(f"Position size calculation failed: {e}") from e

    def validate_order(
        self,
        symbol: str,
        quantity: Shares,
        side: str,
        entry_price: Price,
        account_value: AccountValue,
        current_positions: Optional[Union[Portfolio, List[Position], Dict[str, DollarAmount]]] = None,
        market_data: Optional[MarketDataDict] = None,
        current_daily_loss: DollarAmount = DollarAmount(0.0)
    ) -> Dict[str, Any]:
        """Comprehensive order validation with structured position data."""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'risk_metrics': {}
        }
        
        try:
            # Basic parameter validation
            if not symbol or not isinstance(symbol, str):
                result['errors'].append("Symbol must be a non-empty string")
                
            # Sanitize symbol (remove special characters)
            clean_symbol = ''.join(c for c in symbol.upper() if c.isalnum())
            if clean_symbol != symbol.upper():
                result['warnings'].append(f"Symbol sanitized from '{symbol}' to '{clean_symbol}'")
                
            if quantity <= 0:
                result['errors'].append(f"Quantity must be positive (got {quantity})")
                
            if entry_price <= 0:
                result['errors'].append(f"Entry price must be positive (got ${entry_price})")
                
            if account_value <= 0:
                result['errors'].append(f"Account value must be positive (got ${account_value:,.2f})")
                
            # Validate and normalize order side
            try:
                normalized_side = normalize_order_side(side)
            except ValueError:
                result['errors'].append(f"Side must be 'BUY' or 'SELL' (got '{side}')")
                normalized_side = 'BUY'  # Default for further calculations
            
            # Position size validation
            position_value = DollarAmount(quantity * entry_price)
            max_position_value = DollarAmount(account_value * self.max_position_pct)
            
            result['risk_metrics']['position_value'] = float(position_value)
            result['risk_metrics']['max_allowed'] = float(max_position_value)
            result['risk_metrics']['position_pct'] = (position_value / account_value) * 100
            result['risk_metrics']['max_order_value'] = float(self.max_order_value)
            
            if position_value > max_position_value:
                result['errors'].append(
                    f"Position value ${position_value:,.2f} exceeds maximum "
                    f"${max_position_value:,.2f} ({self.max_position_pct*100:.0f}% of account)"
                )
            
            # Check configured order value limit
            if position_value > self.max_order_value:
                result['errors'].append(
                    f"Position value ${position_value:,.2f} exceeds configured maximum "
                    f"order value ${self.max_order_value:,.2f}"
                )
            
            # Position count validation - handle different input types
            current_position_count = 0
            total_exposure = position_value
            
            if isinstance(current_positions, Portfolio):
                current_position_count = current_positions.position_count
                total_exposure = DollarAmount(current_positions.total_position_value + position_value)
            elif isinstance(current_positions, list):
                current_position_count = len([pos for pos in current_positions if pos.status == 'OPEN'])
                total_exposure = DollarAmount(sum(pos.value for pos in current_positions) + position_value)
            elif isinstance(current_positions, dict):
                current_position_count = len(current_positions)
                total_exposure = DollarAmount(sum(current_positions.values()) + position_value)
            
            if current_position_count >= self.max_positions:
                result['errors'].append(
                    f"Maximum positions ({self.max_positions}) already reached. "
                    f"Current positions: {current_position_count}"
                )
            
            # Daily loss limit validation
            max_daily_loss_amount = DollarAmount(account_value * self.max_daily_loss)
            result['risk_metrics']['current_daily_loss'] = float(current_daily_loss)
            result['risk_metrics']['max_daily_loss'] = float(max_daily_loss_amount)
            result['risk_metrics']['daily_loss_pct'] = (current_daily_loss / account_value) * 100
            
            if current_daily_loss >= max_daily_loss_amount:
                result['errors'].append(
                    f"Daily loss limit (${max_daily_loss_amount:,.2f}) already reached. "
                    f"Current loss: ${current_daily_loss:,.2f}"
                )
            
            # Concentration risk check
            concentration_pct = (total_exposure / account_value) * 100
            result['risk_metrics']['total_exposure_pct'] = concentration_pct
            
            if concentration_pct > self.max_correlation_exposure * 100:
                result['warnings'].append(
                    f"Total exposure {concentration_pct:.1f}% exceeds recommended "
                    f"{self.max_correlation_exposure*100:.0f}%"
                )
            
            # Market data validation
            if market_data:
                daily_volume = market_data.get('volume', 0)
                if daily_volume < self.min_liquidity_threshold:
                    result['warnings'].append(
                        f"Low liquidity: ${daily_volume:,.0f} daily volume "
                        f"(minimum ${self.min_liquidity_threshold:,.0f})"
                    )
                
                bid_ask_spread = market_data.get('spread', 0)
                if bid_ask_spread > entry_price * 0.01:  # 1% spread threshold
                    result['warnings'].append(
                        f"Wide bid-ask spread: ${bid_ask_spread:.2f} "
                        f"({bid_ask_spread/entry_price*100:.2f}%)"
                    )
                
                # Additional market data checks
                if 'volatility' in market_data and market_data['volatility'] and market_data['volatility'] > 0.3:
                    result['warnings'].append(f"High volatility detected: {market_data['volatility']*100:.1f}%")
            
            # Set final validation status
            result['valid'] = len(result['errors']) == 0
            
            if result['valid']:
                logger.debug(f"Order validated: {clean_symbol} {quantity} shares {normalized_side}")
            else:
                logger.warning(f"Order validation failed: {'; '.join(result['errors'])}")
                
            return result
            
        except Exception as e:
            logger.exception("Error during order validation")
            return {
                'valid': False,
                'errors': [f"Validation error: {e}"],
                'warnings': [],
                'risk_metrics': {}
            }
    
    def get_portfolio_risk_summary(
        self, 
        portfolio: Union[Portfolio, Dict[str, Dict[str, float]]],
        account_value: Optional[AccountValue] = None,
        current_daily_loss: DollarAmount = DollarAmount(0.0)
    ) -> Dict[str, Any]:
        """Generate comprehensive portfolio risk summary with structured data."""
        try:
            # Handle different input types
            if isinstance(portfolio, Portfolio):
                positions = portfolio.positions
                account_val = account_value or portfolio.account_value
                total_position_value = portfolio.total_position_value
                cash_pct = portfolio.cash_percentage * 100
                num_positions = portfolio.position_count
                max_position_pct = portfolio.largest_position_pct * 100
                position_pcts = {
                    pos.symbol: (pos.value / account_val * 100)
                    for pos in positions
                }
            else:
                # Legacy dictionary format
                if account_value is None:
                    raise ValueError("account_value is required when using dictionary format")
                account_val = account_value
                total_position_value = DollarAmount(sum(pos['value'] for pos in portfolio.values()))
                cash_pct = max(0, (account_val - total_position_value) / account_val * 100)
                position_pcts = {
                    symbol: (pos['value'] / account_val * 100)
                    for symbol, pos in portfolio.items()
                }
                max_position_pct = max(position_pcts.values()) if position_pcts else 0
                num_positions = len(portfolio)
            
            # Daily loss metrics
            daily_loss_pct = (current_daily_loss / account_val) * 100
            daily_loss_limit_pct = self.max_daily_loss * 100
            
            # Risk scoring (0-100, lower is better)
            risk_score = 0
            if max_position_pct > self.max_position_pct * 100:
                risk_score += 30
            if num_positions < 5:  # Diversification penalty
                risk_score += 20
            if cash_pct < 10:  # Low cash penalty
                risk_score += 15
            if num_positions >= self.max_positions:  # Position limit penalty
                risk_score += 25
            if daily_loss_pct > daily_loss_limit_pct * 0.75:  # Near daily limit
                risk_score += 20
            
            # Determine risk level
            risk_level: RiskLevel = 'HIGH' if risk_score > 60 else 'MEDIUM' if risk_score > 30 else 'LOW'
                
            return {
                'total_positions': num_positions,
                'max_positions': self.max_positions,
                'total_exposure_pct': (total_position_value / account_val) * 100,
                'cash_pct': cash_pct,
                'max_position_pct': max_position_pct,
                'avg_position_pct': sum(position_pcts.values()) / len(position_pcts) if position_pcts else 0,
                'position_breakdown': position_pcts,
                'daily_loss_amount': float(current_daily_loss),
                'daily_loss_pct': daily_loss_pct,
                'daily_loss_limit_pct': daily_loss_limit_pct,
                'daily_loss_remaining': float(max(0, DollarAmount(account_val * self.max_daily_loss) - current_daily_loss)),
                'risk_score': min(risk_score, 100),
                'risk_level': risk_level,
                'recommendations': self._generate_risk_recommendations(
                    max_position_pct, num_positions, cash_pct, daily_loss_pct
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio risk summary: {e}")
            return {'error': str(e)}

    def _generate_risk_recommendations(
        self, 
        max_position_pct: float, 
        num_positions: int, 
        cash_pct: float,
        daily_loss_pct: float
    ) -> list:
        """Generate actionable risk management recommendations."""
        recommendations = []
        
        if max_position_pct > self.max_position_pct * 100:
            recommendations.append(
                f"Reduce largest position size (currently {max_position_pct:.1f}%, "
                f"target <{self.max_position_pct*100:.0f}%)"
            )
            
        if num_positions < 5:
            recommendations.append(
                f"Increase diversification (currently {num_positions} positions, target 5-10)"
            )
            
        if num_positions >= self.max_positions:
            recommendations.append(
                f"At maximum position limit ({self.max_positions}). "
                "Consider closing positions before opening new ones."
            )
            
        if cash_pct < 10:
            recommendations.append(
                f"Maintain higher cash reserves (currently {cash_pct:.1f}%, target >10%)"
            )
            
        if daily_loss_pct > self.max_daily_loss * 100 * 0.75:
            recommendations.append(
                f"Approaching daily loss limit ({daily_loss_pct:.1f}% of "
                f"{self.max_daily_loss*100:.1f}% limit). Consider reducing position sizes."
            )
            
        if not recommendations:
            recommendations.append("Portfolio risk profile is within acceptable parameters")
            
        return recommendations

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get current configuration summary for debugging/monitoring."""
        return {
            'max_position_pct': float(self.max_position_pct),
            'max_correlation_exposure': float(self.max_correlation_exposure),
            'min_liquidity_threshold': float(self.min_liquidity_threshold),
            'max_positions': self.max_positions,
            'max_daily_loss': float(self.max_daily_loss),
            'max_order_value': float(self.max_order_value),
            'default_atr_period': RiskConfigManager.get_default_atr_period(),
            'default_atr_multiplier': RiskConfigManager.get_default_atr_multiplier(),
        }
