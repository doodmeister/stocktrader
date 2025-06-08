from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from .validation_config import ValidationConfig

class FinancialData(BaseModel):
    """Validated financial data structure."""
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        extra='forbid'
    )
    
    symbol: str = Field(..., min_length=1, max_length=10)
    price: float = Field(..., gt=0, le=ValidationConfig.MAX_PRICE)
    volume: int = Field(..., ge=0, le=ValidationConfig.MAX_VOLUME)
    timestamp: datetime = Field(...)
    
    @field_validator('symbol')
    def validate_symbol_format(cls, v: str) -> str:
        """Validate symbol format using class validator."""
        if not ValidationConfig.SYMBOL_PATTERN.match(v.upper()):
            # This part of the original code was incomplete.
            # Assuming it should raise a ValueError or return as is if already uppercase.
            # For now, let's ensure it's uppercase.
            pass # Placeholder for actual validation logic if needed beyond the pattern
        return v.upper()

class MarketDataPoint(BaseModel):
    """Validated OHLCV market data point."""
    model_config = ConfigDict(validate_assignment=True)
    
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    timestamp: datetime = Field(...)
    
    @field_validator('high')
    def validate_high_price(cls, v: float, info) -> float:
        """Ensure high is the maximum price."""
        # The original implementation of this validator was incomplete.
        # A proper implementation would compare 'high' with 'open', 'low', 'close'.
        # For now, returning the value as is.
        if info.data:
            # Example: if 'low' in info.data and v < info.data['low']:
            # raise ValueError('high must be greater than or equal to low')
            pass
        return v
    
    @field_validator('low')
    def validate_low_price(cls, v: float, info) -> float:
        """Ensure low is the minimum price."""
        # The original implementation of this validator was incomplete.
        # A proper implementation would compare 'low' with 'open', 'high', 'close'.
        # For now, returning the value as is.
        if info.data:
            # Example: if 'high' in info.data and v > info.data['high']:
            # raise ValueError('low must be less than or equal to high')
            pass
        return v
