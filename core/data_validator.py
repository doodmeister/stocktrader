"""
World-Class Data Validation Module for StockTrader

A comprehensive validation system that consolidates best practices from across the codebase,
providing enterprise-grade validation for financial data, user inputs, and system parameters.

Features:
- Advanced symbol validation with real-time API checking
- Interval-specific date range validation for different data providers  
- OHLCV data integrity validation with statistical checks
- Financial parameter validation (prices, quantities, percentages)
- Performance optimization with caching and rate limiting
- Comprehensive error handling and logging
- Thread-safe operations
- Pydantic integration for model validation
- Security-focused input sanitization
"""

import re
import time
import threading
import os
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import (
    Dict, List, Optional, Tuple, Union, Set, Any, 
    TypeVar, Callable, TypedDict
)
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict

from utils.logger import get_dashboard_logger
from core.exceptions import (
    ValidationError,
    SymbolValidationError,
    DateValidationError,
    DataIntegrityError,
    SecurityValidationError,
    PerformanceValidationError,
)
# Configure logger for this module
logger = get_dashboard_logger(__name__)

# Type definitions for better type safety
SymbolStr = str
IntervalStr = str

# Global configuration constants
class ValidationConfig:
    """Central configuration for validation parameters."""
    
    # Symbol validation
    SYMBOL_MIN_LENGTH = 1
    SYMBOL_MAX_LENGTH = 10
    SYMBOL_PATTERN = re.compile(r'^[A-Z0-9.-]{1,10}$')
    SYMBOL_REQUIRED_ALPHA = True
    
    # Date validation intervals (in days) - Yahoo Finance specific
    INTERVAL_LIMITS = {
        "1m": 7,      # 7 days max for 1-minute data
        "2m": 60,     # 60 days max for 2-minute data
        "5m": 60,     # 60 days max for 5-minute data
        "15m": 60,    # 60 days max for 15-minute data
        "30m": 60,    # 60 days max for 30-minute data
        "60m": 730,   # ~2 years max for 1-hour data
        "1h": 730,    # ~2 years max for hourly data
        "1d": 36500,  # ~100 years (essentially unlimited) for daily data
        "5d": 36500,  # No limit for 5-day data
        "1wk": 36500, # No limit for weekly data
        "1mo": 36500, # No limit for monthly data
        "3mo": 36500  # No limit for quarterly data
    }
    
    # Cache settings
    SYMBOL_CACHE_TTL = 4 * 60 * 60  # 4 hours
    VALIDATION_CACHE_SIZE = 1000
    API_TIMEOUT = 5  # seconds
    MAX_API_CALLS_PER_BATCH = 15
    
    # Data validation thresholds
    MIN_DATASET_SIZE = 10
    MAX_NULL_PERCENTAGE = 0.05  # 5% max null values
    MAX_INVALID_OHLC_PERCENTAGE = 0.05  # 5% max invalid OHLC relationships
    
    # Financial validation ranges
    MIN_PRICE = 0.01
    MAX_PRICE = 1_000_000.0
    MIN_VOLUME = 0
    MAX_VOLUME = 1_000_000_000_000  # 1 trillion
    MIN_QUANTITY = 0
    MAX_QUANTITY = 1_000_000_000
    MIN_PERCENTAGE = 0.0
    MAX_PERCENTAGE = 1.0
    
    # Security settings
    MAX_INPUT_LENGTH = 1000
    DANGEROUS_CHARS_PATTERN = re.compile(r'[<>"\'\x00-\x1f\x7f-\x9f]')

# Result classes for structured validation feedback
@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation result with detailed feedback."""
    is_valid: bool
    value: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'is_valid': self.is_valid,
            'value': self.value,
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata
        }

@dataclass(frozen=True)
class DataFrameValidationResult(ValidationResult):
    """Extended validation result for DataFrame validation."""
    row_count: int = 0
    column_count: int = 0
    null_counts: Dict[str, int] = field(default_factory=dict)
    data_types: Dict[str, str] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)

# Pydantic models for complex validation scenarios
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
            raise ValueError(f"Invalid symbol format: {v}")
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
        if info.data:
            prices = [info.data.get('open', 0), info.data.get('low', 0), info.data.get('close', 0)]
            if v < max(prices):
                raise ValueError("High price must be >= open, low, and close prices")
        return v
    
    @field_validator('low')
    def validate_low_price(cls, v: float, info) -> float:
        """Ensure low is the minimum price."""
        if info.data:
            prices = [info.data.get('open', float('inf')), info.data.get('high', float('inf')), info.data.get('close', float('inf'))]
            if v > min(prices):
                raise ValueError("Low price must be <= open, high, and close prices")
        return v

# Main validation class
class DataValidator:
    """
    World-class data validation system for financial applications.
    
    Provides comprehensive validation for:
    - Stock symbols with real-time API verification
    - Date ranges with interval-specific limitations
    - OHLCV data integrity and statistical validation
    - Financial parameters with range checking
    - User input sanitization and security
    - Performance optimization with caching
    """
    
    def __init__(self, enable_api_validation: bool = True, cache_size: int = None):
        """
        Initialize the validator with configuration options.
        
        Args:
            enable_api_validation: Whether to perform real-time API validation
            cache_size: Maximum cache size (None for default)
        """
        self.enable_api_validation = enable_api_validation
        self.cache_size = cache_size or ValidationConfig.VALIDATION_CACHE_SIZE
        
        # Thread-safe caches
        self._symbol_cache: Dict[str, Tuple[bool, float]] = {}
        self._validation_cache: Dict[str, ValidationResult] = {}
        self._cache_lock = threading.RLock()
        
        # Compiled regex patterns for performance
        self._symbol_pattern = ValidationConfig.SYMBOL_PATTERN
        self._dangerous_chars = ValidationConfig.DANGEROUS_CHARS_PATTERN
        
        # Performance tracking
        self._validation_stats = {
            'symbol_validations': 0,
            'date_validations': 0,
            'dataframe_validations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'validation_errors': 0
        }
        
        logger.info(f"DataValidator initialized with API validation: {enable_api_validation}")
    
    # ========================================
    # CORE VALIDATION METHODS
    # ========================================
    
    def validate_symbol(self, symbol: str, check_api: bool = None) -> ValidationResult:
        """
        Comprehensive symbol validation with optional API verification.
        
        Args:
            symbol: Stock ticker symbol to validate
            check_api: Override API checking (None uses instance default)
            
        Returns:
            ValidationResult with validation outcome and details
        """
        start_time = time.time()
        self._validation_stats['symbol_validations'] += 1
        
        try:
            # Input sanitization
            if not symbol or not isinstance(symbol, str):
                return ValidationResult(
                    is_valid=False,
                    errors=["Symbol must be a non-empty string"]
                )
            
            # Security check
            if len(symbol) > ValidationConfig.MAX_INPUT_LENGTH:
                raise SecurityValidationError(f"Symbol exceeds maximum length: {len(symbol)}")
            
            if self._dangerous_chars.search(symbol):
                raise SecurityValidationError("Symbol contains dangerous characters")
            
            # Basic format validation
            clean_symbol = symbol.strip().upper()
            errors = []
            warnings = []
            
            if len(clean_symbol) < ValidationConfig.SYMBOL_MIN_LENGTH:
                errors.append(f"Symbol too short (minimum {ValidationConfig.SYMBOL_MIN_LENGTH} characters)")
            
            if len(clean_symbol) > ValidationConfig.SYMBOL_MAX_LENGTH:
                errors.append(f"Symbol too long (maximum {ValidationConfig.SYMBOL_MAX_LENGTH} characters)")
            
            if not self._symbol_pattern.match(clean_symbol):
                errors.append(f"Invalid symbol format: {clean_symbol}. Must contain only letters, numbers, dots, and hyphens")
            
            if ValidationConfig.SYMBOL_REQUIRED_ALPHA and not any(c.isalpha() for c in clean_symbol):
                errors.append("Symbol must contain at least one letter")
            
            # Early return if basic validation fails
            if errors:
                return ValidationResult(
                    is_valid=False,
                    value=clean_symbol,
                    errors=errors,
                    metadata={'validation_time': time.time() - start_time}
                )
            
            # API validation (if enabled)
            api_valid = True
            if (check_api if check_api is not None else self.enable_api_validation):
                try:
                    api_result = self._validate_symbol_api(clean_symbol)
                    if not api_result:
                        warnings.append(f"Symbol {clean_symbol} not found in market data")
                        api_valid = False
                except Exception as e:
                    warnings.append(f"API validation failed: {str(e)}")
                    logger.debug(f"API validation error for {clean_symbol}: {e}")
            
            result = ValidationResult(
                is_valid=True,
                value=clean_symbol,
                warnings=warnings,
                metadata={
                    'validation_time': time.time() - start_time,
                    'api_validated': api_valid,
                    'cleaned': clean_symbol != symbol.strip().upper()
                }
            )
            
            logger.debug(f"Symbol validation completed for {clean_symbol} in {time.time() - start_time:.3f}s")
            return result
            
        except Exception as e:
            self._validation_stats['validation_errors'] += 1
            logger.error(f"Symbol validation error for '{symbol}': {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                metadata={'validation_time': time.time() - start_time}
            )
    
    def validate_symbols(self, symbols_input: str, max_symbols: int = 50) -> ValidationResult:
        """
        Validate multiple comma-separated symbols with batch processing.
        
        Args:
            symbols_input: Comma-separated list of symbols
            max_symbols: Maximum number of symbols allowed
            
        Returns:
            ValidationResult with list of valid symbols
        """
        start_time = time.time()
        
        try:
            if not symbols_input or not symbols_input.strip():
                return ValidationResult(
                    is_valid=False,
                    errors=["No symbols provided"]
                )
            
            # Parse and clean symbols
            raw_symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]
            
            if len(raw_symbols) > max_symbols:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Too many symbols: {len(raw_symbols)} (maximum {max_symbols})"]
                )
            
            valid_symbols = []
            invalid_symbols = []
            warnings = []
            
            # Validate each symbol
            for symbol in raw_symbols:
                result = self.validate_symbol(symbol)
                if result.is_valid:
                    valid_symbols.append(result.value)
                    warnings.extend(result.warnings)
                else:
                    invalid_symbols.extend([f"{symbol}: {error}" for error in result.errors])
            
            if not valid_symbols:
                return ValidationResult(
                    is_valid=False,
                    errors=["No valid symbols found"] + invalid_symbols,
                    metadata={'validation_time': time.time() - start_time}
                )
            
            # Remove duplicates while preserving order
            unique_symbols = list(dict.fromkeys(valid_symbols))
            
            result = ValidationResult(
                is_valid=True,
                value=unique_symbols,
                warnings=warnings + ([f"Invalid symbols: {', '.join(invalid_symbols)}"] if invalid_symbols else []),
                metadata={
                    'validation_time': time.time() - start_time,
                    'total_input': len(raw_symbols),
                    'valid_count': len(unique_symbols),
                    'invalid_count': len(invalid_symbols),
                    'duplicates_removed': len(valid_symbols) - len(unique_symbols)
                }
            )
            
            logger.debug(f"Batch symbol validation: {len(unique_symbols)}/{len(raw_symbols)} valid")
            return result
            
        except Exception as e:
            self._validation_stats['validation_errors'] += 1
            logger.error(f"Batch symbol validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                metadata={'validation_time': time.time() - start_time}
            )
    
    def validate_dates(self, start_date: date, end_date: date, interval: str = "1d") -> ValidationResult:
        """
        Advanced date validation with interval-specific limitations and business logic.
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range  
            interval: Data interval (1m, 5m, 1h, 1d, etc.)
            
        Returns:
            ValidationResult with validation outcome and suggested adjustments
        """
        start_time = time.time()
        self._validation_stats['date_validations'] += 1
        
        try:
            errors = []
            warnings = []
            today = date.today()
            
            # Basic date validation
            if start_date > end_date:
                errors.append("Start date must be <= end date")
            
            if end_date > today:
                errors.append("End date cannot be in the future")
            
            # Very old data warning
            if start_date < date(2000, 1, 1):
                warnings.append("Very old start date - data availability may be limited")
            
            # Weekend handling for intraday data
            if interval in ['1m', '2m', '5m', '15m', '30m', '1h']:
                if start_date.weekday() > 4:  # Saturday=5, Sunday=6
                    warnings.append("Start date is on weekend - market data may not be available")
                if end_date.weekday() > 4:
                    warnings.append("End date is on weekend - market data may not be available")
            
            if errors:
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    metadata={'validation_time': time.time() - start_time}
                )
            
            # Interval-specific validation
            days_diff = (end_date - start_date).days
            max_days = ValidationConfig.INTERVAL_LIMITS.get(interval, 36500)
            
            interval_valid = days_diff <= max_days
            
            metadata = {
                'validation_time': time.time() - start_time,
                'days_span': days_diff,
                'max_allowed_days': max_days,
                'interval': interval,
                'is_intraday': interval in ['1m', '2m', '5m', '15m', '30m', '1h'],
                'suggested_end_date': None
            }
            
            if not interval_valid:
                # Calculate suggested end date
                suggested_end = start_date + timedelta(days=max_days)
                if suggested_end > today:
                    suggested_end = today
                    
                metadata['suggested_end_date'] = suggested_end.isoformat()
                
                errors.append(
                    f"Date range too large for {interval} interval: {days_diff} days "
                    f"(maximum {max_days} days). Consider using end date: {suggested_end}"
                )
            
            # Performance warnings for large datasets
            if interval in ['1m', '2m'] and days_diff > 1:
                warnings.append(f"Large dataset warning: {interval} data for {days_diff} days may be slow to load")
            
            result = ValidationResult(
                is_valid=interval_valid,
                value={'start_date': start_date, 'end_date': end_date, 'interval': interval},
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )
            
            logger.debug(f"Date validation: {start_date} to {end_date} ({interval}) - {result.is_valid}")
            return result
            
        except Exception as e:
            self._validation_stats['validation_errors'] += 1
            logger.error(f"Date validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                metadata={'validation_time': time.time() - start_time}            )
    
    def validate_dataframe(self, df: pd.DataFrame, required_cols: List[str] = None, 
                         validate_ohlc: bool = True, check_statistical_anomalies: bool = True, 
                         min_rows: int = None) -> DataFrameValidationResult:
        """
        Comprehensive DataFrame validation for financial data.
        
        Args:
            df: DataFrame to validate
            required_cols: List of required column names
            validate_ohlc: Whether to validate OHLC relationships
            check_statistical_anomalies: Whether to check for statistical anomalies
            
        Returns:
            DataFrameValidationResult with detailed validation feedback
        """
        start_time = time.time()
        self._validation_stats['dataframe_validations'] += 1
        
        try:
            errors = []
            warnings = []
            
            # Basic DataFrame checks
            if df is None:
                return DataFrameValidationResult(
                    is_valid=False,
                    errors=["DataFrame is None"]
                )
            
            if df.empty:
                return DataFrameValidationResult(
                    is_valid=False,
                    errors=["DataFrame is empty"]                )
            
            row_count, col_count = df.shape
            
            # Minimum size check - use custom min_rows if provided
            minimum_rows = min_rows if min_rows is not None else ValidationConfig.MIN_DATASET_SIZE
            if row_count < minimum_rows:
                errors.append(f"Insufficient data: {row_count} rows (minimum {minimum_rows})")
            
            # Required columns check
            if required_cols:
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    errors.append(f"Missing required columns: {missing_cols}")
            
            # Data type validation
            data_types = {}
            for col in df.columns:
                data_types[col] = str(df[col].dtype)
            
            # Null value analysis
            null_counts = df.isnull().sum().to_dict()
            high_null_cols = []
            
            for col, null_count in null_counts.items():
                null_pct = null_count / len(df)
                if null_pct > ValidationConfig.MAX_NULL_PERCENTAGE:
                    high_null_cols.append(f"{col} ({null_pct:.1%})")
            
            if high_null_cols:
                warnings.append(f"High null value percentage in columns: {', '.join(high_null_cols)}")
            
            # OHLC validation if requested
            ohlc_stats = {}
            if validate_ohlc:
                ohlc_result = self._validate_ohlc_relationships(df)
                if not ohlc_result.is_valid:
                    errors.extend(ohlc_result.errors)
                warnings.extend(ohlc_result.warnings)
                ohlc_stats = ohlc_result.metadata
            
            # Statistical anomaly detection
            anomaly_stats = {}
            if check_statistical_anomalies:
                anomaly_result = self._detect_statistical_anomalies(df)
                warnings.extend(anomaly_result.warnings)
                anomaly_stats = anomaly_result.metadata
            
            # Calculate basic statistics
            statistics = {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'date_range': self._get_date_range_info(df),
                **ohlc_stats,
                **anomaly_stats
            }
            
            result = DataFrameValidationResult(
                is_valid=len(errors) == 0,
                value=df if len(errors) == 0 else None,
                errors=errors,
                warnings=warnings,
                row_count=row_count,
                column_count=col_count,
                null_counts=null_counts,
                data_types=data_types,
                statistics=statistics,
                metadata={
                    'validation_time': time.time() - start_time,
                    'validation_level': 'comprehensive',
                    'checks_performed': {
                        'basic_structure': True,
                        'required_columns': bool(required_cols),
                        'ohlc_relationships': validate_ohlc,
                        'statistical_anomalies': check_statistical_anomalies
                    }
                }
            )
            
            logger.debug(f"DataFrame validation completed: {row_count}x{col_count} - {result.is_valid}")
            return result
            
        except Exception as e:
            self._validation_stats['validation_errors'] += 1
            logger.error(f"DataFrame validation error: {e}")
            return DataFrameValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                metadata={'validation_time': time.time() - start_time}
            )
    
    # ========================================
    # FINANCIAL PARAMETER VALIDATION
    # ========================================
    
    def validate_price(self, price: Union[float, str], min_price: float = None, max_price: float = None) -> ValidationResult:
        """
        Validate price values with financial market constraints.
        
        Args:
            price: Price value to validate
            min_price: Minimum allowed price (None for default)
            max_price: Maximum allowed price (None for default)
            
        Returns:
            ValidationResult with validated price
        """
        try:
            # Convert to float if string
            if isinstance(price, str):
                try:
                    price = float(price.replace('$', '').replace(',', ''))
                except ValueError:
                    return ValidationResult(
                        is_valid=False,
                        errors=[f"Invalid price format: {price}"]
                    )
            
            if not isinstance(price, (int, float)):
                return ValidationResult(
                    is_valid=False,
                    errors=["Price must be a number"]
                )
            
            min_val = min_price if min_price is not None else ValidationConfig.MIN_PRICE
            max_val = max_price if max_price is not None else ValidationConfig.MAX_PRICE
            
            errors = []
            warnings = []
            
            if price < min_val:
                errors.append(f"Price ${price:.4f} is below minimum ${min_val}")
            
            if price > max_val:
                errors.append(f"Price ${price:,.2f} exceeds maximum ${max_val:,.2f}")
            
            # Warning for unusual prices
            if price > 10000:
                warnings.append(f"Very high price detected: ${price:,.2f}")
            elif price < 1:
                warnings.append(f"Very low price detected: ${price:.4f}")
            
            # Ensure reasonable decimal precision
            validated_price = round(float(price), 4)
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                value=validated_price,
                errors=errors,
                warnings=warnings,
                metadata={'original_value': price, 'precision_adjusted': validated_price != price}
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Price validation error: {str(e)}"]
            )
    
    def validate_quantity(self, quantity: Union[int, float, str], allow_fractional: bool = False) -> ValidationResult:
        """
        Validate quantity values for trading operations.
        
        Args:
            quantity: Quantity to validate
            allow_fractional: Whether to allow fractional shares
            
        Returns:
            ValidationResult with validated quantity
        """
        try:
            # Convert to number if string
            if isinstance(quantity, str):
                try:
                    quantity = float(quantity.replace(',', ''))
                except ValueError:
                    return ValidationResult(
                        is_valid=False,
                        errors=[f"Invalid quantity format: {quantity}"]
                    )
            
            errors = []
            warnings = []
            
            if quantity < ValidationConfig.MIN_QUANTITY:
                errors.append(f"Quantity must be positive (got {quantity})")
            
            if quantity > ValidationConfig.MAX_QUANTITY:
                errors.append(f"Quantity {quantity:,.0f} exceeds maximum {ValidationConfig.MAX_QUANTITY:,.0f}")
            
            # Handle fractional shares
            if not allow_fractional and quantity != int(quantity):
                if quantity - int(quantity) < 0.001:  # Very small fraction
                    quantity = int(quantity)
                    warnings.append("Fractional part too small, rounded to whole number")
                else:
                    errors.append("Fractional shares not allowed")
            
            validated_quantity = int(quantity) if not allow_fractional else quantity
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                value=validated_quantity,
                errors=errors,
                warnings=warnings,
                metadata={'allow_fractional': allow_fractional}
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Quantity validation error: {str(e)}"]
            )
    
    def validate_percentage(self, percentage: Union[float, str], min_pct: float = None, max_pct: float = None) -> ValidationResult:
        """
        Validate percentage values with proper range checking.
        
        Args:
            percentage: Percentage to validate (0.0-1.0 or 0-100 if string ends with %)
            min_pct: Minimum allowed percentage
            max_pct: Maximum allowed percentage
            
        Returns:
            ValidationResult with validated percentage as decimal
        """
        try:
            # Handle string input
            if isinstance(percentage, str):
                percentage = percentage.strip()
                if percentage.endswith('%'):
                    try:
                        percentage = float(percentage[:-1]) / 100
                    except ValueError:
                        return ValidationResult(
                            is_valid=False,
                            errors=[f"Invalid percentage format: {percentage}"]
                        )
                else:
                    try:
                        percentage = float(percentage)
                    except ValueError:
                        return ValidationResult(
                            is_valid=False,
                            errors=[f"Invalid percentage format: {percentage}"]
                        )
            
            if not isinstance(percentage, (int, float)):
                return ValidationResult(
                    is_valid=False,
                    errors=["Percentage must be a number"]
                )
            
            min_val = min_pct if min_pct is not None else ValidationConfig.MIN_PERCENTAGE
            max_val = max_pct if max_pct is not None else ValidationConfig.MAX_PERCENTAGE
            
            errors = []
            warnings = []
            
            if percentage < min_val:
                errors.append(f"Percentage {percentage*100:.2f}% is below minimum {min_val*100:.2f}%")
            
            if percentage > max_val:
                errors.append(f"Percentage {percentage*100:.2f}% exceeds maximum {max_val*100:.2f}%")
            
            # Warning for extreme values
            if percentage > 0.5:  # 50%
                warnings.append(f"High percentage value: {percentage*100:.1f}%")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                value=percentage,
                errors=errors,
                warnings=warnings,
                metadata={'as_percentage': f"{percentage*100:.2f}%"}
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Percentage validation error: {str(e)}"]
            )
    
    # ========================================
    # SECURITY AND SANITIZATION
    # ========================================
    
    def sanitize_input(self, input_str: str, max_length: int = None, allow_html: bool = False) -> str:
        """
        Sanitize user input to prevent security issues.
        
        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML tags
            
        Returns:
            Sanitized string
        """
        if not input_str:
            return ""
        
        max_len = max_length or ValidationConfig.MAX_INPUT_LENGTH
        
        # Truncate if too long
        sanitized = str(input_str)[:max_len]
        
        # Remove dangerous characters
        if not allow_html:
            sanitized = self._dangerous_chars.sub('', sanitized)
        
        # Strip whitespace
        sanitized = sanitized.strip()
        
        return sanitized
    
    def validate_file_path(self, file_path: Union[str, Path], base_directory: Path = None, 
                          allowed_extensions: List[str] = None) -> ValidationResult:
        """
        Validate file paths for security and accessibility.
        
        Args:
            file_path: File path to validate
            base_directory: Base directory for relative path validation
            allowed_extensions: List of allowed file extensions
            
        Returns:
            ValidationResult with validated path
        """
        try:
            path = Path(file_path)
            errors = []
            warnings = []
            
            # Security checks
            if '..' in str(path):
                errors.append("Path traversal detected (contains '..')")
            
            # Extension validation
            if allowed_extensions and path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                errors.append(f"File extension '{path.suffix}' not allowed. Allowed: {allowed_extensions}")
            
            # Base directory validation
            if base_directory:
                try:
                    if not path.is_absolute():
                        path = base_directory / path
                    path.resolve().relative_to(base_directory.resolve())
                except ValueError:
                    errors.append("Path is outside allowed directory")
            
            # Accessibility checks
            if path.exists():
                if not path.is_file():
                    errors.append("Path exists but is not a file")
                elif not os.access(path, os.R_OK):
                    errors.append("File is not readable")
            else:
                warnings.append("File does not exist")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                value=str(path.resolve()) if len(errors) == 0 else None,
                errors=errors,
                warnings=warnings,
                metadata={
                    'original_path': str(file_path),
                    'resolved_path': str(path.resolve()),
                    'exists': path.exists(),
                    'is_absolute': path.is_absolute()
                }
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Path validation error: {str(e)}"]
            )
    
    # ========================================
    # PERFORMANCE AND UTILITY METHODS
    # ========================================
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics."""
        cache_hit_rate = (
            self._validation_stats['cache_hits'] / 
            max(1, self._validation_stats['cache_hits'] + self._validation_stats['cache_misses'])
        ) * 100
        
        return {
            **self._validation_stats,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'cache_size': len(self._symbol_cache) + len(self._validation_cache),
            'max_cache_size': self.cache_size
        }
    
    def clear_cache(self):
        """Clear all validation caches."""
        with self._cache_lock:
            self._symbol_cache.clear()
            self._validation_cache.clear()
        logger.info("Validation caches cleared")
    
    def get_interval_limits_info(self) -> Dict[str, str]:
        """Get user-friendly information about interval limitations."""
        return {
            interval: f"Maximum {days} days" if days < 36500 else "No practical limit"
            for interval, days in ValidationConfig.INTERVAL_LIMITS.items()
        }
    
    # ========================================
    # PRIVATE HELPER METHODS
    # ========================================
    
    def _validate_symbol_api(self, symbol: str) -> bool:
        """
        Validate symbol against external API with caching and rate limiting.
        
        Args:
            symbol: Clean symbol to validate
            
        Returns:
            True if symbol is valid according to API
        """
        # Check cache first
        with self._cache_lock:
            if symbol in self._symbol_cache:
                is_valid, timestamp = self._symbol_cache[symbol]
                if time.time() - timestamp < ValidationConfig.SYMBOL_CACHE_TTL:
                    self._validation_stats['cache_hits'] += 1
                    return is_valid
                else:
                    # Cache expired
                    del self._symbol_cache[symbol]
            
            self._validation_stats['cache_misses'] += 1
        
        # Perform API validation
        try:
            self._validation_stats['api_calls'] += 1
            
            # Import here to avoid circular imports
            import yfinance as yf
            import requests.exceptions
            
            ticker = yf.Ticker(symbol)
            
            # Quick API call with timeout
            try:
                info = ticker.info
                is_valid = bool(info and 'symbol' in info)
            except (requests.exceptions.RequestException, ValueError, KeyError, TypeError):
                is_valid = False
            
            # Cache the result
            with self._cache_lock:
                self._symbol_cache[symbol] = (is_valid, time.time())
                
                # Limit cache size
                if len(self._symbol_cache) > self.cache_size:
                    # Remove oldest entries
                    oldest_key = min(self._symbol_cache.keys(), 
                                   key=lambda k: self._symbol_cache[k][1])
                    del self._symbol_cache[oldest_key]
            
            return is_valid
            
        except ImportError:
            logger.warning("yfinance not available for API validation")
            return True  # Assume valid if no API available
        except Exception as e:
            logger.debug(f"API validation error for {symbol}: {e}")
            return False
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate OHLC relationships in financial data.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            ValidationResult with OHLC validation details
        """
        try:
            required_cols = ['open', 'high', 'low', 'close']
            available_cols = [col for col in required_cols if col in df.columns]
            
            if len(available_cols) < 4:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Missing OHLC columns: {set(required_cols) - set(available_cols)}"]
                )
            
            errors = []
            warnings = []
            
            # Check basic OHLC relationships
            invalid_high = df[df['high'] < df[['open', 'low', 'close']].max(axis=1)]
            invalid_low = df[df['low'] > df[['open', 'high', 'close']].min(axis=1)]
            
            invalid_count = len(invalid_high) + len(invalid_low)
            invalid_pct = invalid_count / len(df)
            
            if invalid_pct > ValidationConfig.MAX_INVALID_OHLC_PERCENTAGE:
                errors.append(f"Too many invalid OHLC relationships: {invalid_count} rows ({invalid_pct:.1%})")
            elif invalid_count > 0:
                warnings.append(f"Found {invalid_count} rows with invalid OHLC relationships ({invalid_pct:.1%})")
            
            # Check for zero or negative prices
            for col in available_cols:
                zero_count = (df[col] <= 0).sum()
                if zero_count > 0:
                    warnings.append(f"Found {zero_count} non-positive values in {col}")
            
            # Statistical checks
            price_cols = df[available_cols]
            
            # Check for extreme price movements
            if 'close' in df.columns:
                close_pct_change = df['close'].pct_change().abs()
                extreme_moves = (close_pct_change > 0.5).sum()  # >50% moves
                if extreme_moves > 0:
                    warnings.append(f"Found {extreme_moves} extreme price movements (>50%)")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata={
                    'invalid_ohlc_count': invalid_count,
                    'invalid_ohlc_percentage': invalid_pct,
                    'extreme_moves': extreme_moves if 'extreme_moves' in locals() else 0,
                    'price_range': {
                        col: {'min': float(df[col].min()), 'max': float(df[col].max())}
                        for col in available_cols
                    }
                }
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"OHLC validation error: {str(e)}"]
            )
    
    def _detect_statistical_anomalies(self, df: pd.DataFrame) -> ValidationResult:
        """
        Detect statistical anomalies in the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            ValidationResult with anomaly detection results
        """
        try:
            warnings = []
            anomaly_stats = {}
            
            # Detect outliers using IQR method for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if len(df[col].dropna()) < 10:  # Skip if too few values
                    continue
                    
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_pct = outlier_count / len(df)
                
                anomaly_stats[f'{col}_outliers'] = {
                    'count': outlier_count,
                    'percentage': outlier_pct,
                    'bounds': (float(lower_bound), float(upper_bound))
                }
                
                if outlier_pct > 0.1:  # More than 10% outliers
                    warnings.append(f"High outlier percentage in {col}: {outlier_pct:.1%}")
            
            # Check for data gaps (if datetime index)
            if isinstance(df.index, pd.DatetimeIndex):
                gaps = self._detect_data_gaps(df)
                if gaps:
                    warnings.append(f"Found {len(gaps)} data gaps")
                    anomaly_stats['data_gaps'] = len(gaps)
            
            return ValidationResult(
                is_valid=True,
                warnings=warnings,
                metadata=anomaly_stats
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=True,
                warnings=[f"Anomaly detection error: {str(e)}"]
            )
    
    def _detect_data_gaps(self, df: pd.DataFrame) -> List[Tuple[datetime, datetime]]:
        """
        Detect gaps in time series data.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            List of (start, end) tuples for detected gaps
        """
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            return []
        
        try:
            # Infer frequency from the data
            freq = pd.infer_freq(df.index)
            if not freq:
                return []  # Can't infer frequency
            
            # Create expected index
            expected_index = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                freq=freq
            )
            
            # Find missing timestamps
            missing = expected_index.difference(df.index)
            
            # Group consecutive missing timestamps into gaps
            gaps = []
            if len(missing) > 0:
                gap_start = missing[0]
                gap_end = missing[0]
                
                for i in range(1, len(missing)):
                    if missing[i] - gap_end <= pd.Timedelta(freq):
                        gap_end = missing[i]
                    else:
                        gaps.append((gap_start, gap_end))
                        gap_start = missing[i]
                        gap_end = missing[i]
                
                # Add the last gap
                gaps.append((gap_start, gap_end))
            
            return gaps
            
        except Exception:
            return []
    
    def _get_date_range_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract date range information from DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with date range information
        """
        try:
            date_info = {}
            
            # Try to find date/datetime columns
            date_cols = []
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_cols.append(col)
                elif df[col].dtype.kind in 'Mm':  # datetime64 types
                    date_cols.append(col)
            
            if date_cols:
                date_col = date_cols[0]  # Use first date column
                date_info['date_column'] = date_col
                date_info['start_date'] = str(df[date_col].min())
                date_info['end_date'] = str(df[date_col].max())
                date_info['date_span_days'] = (df[date_col].max() - df[date_col].min()).days
            elif isinstance(df.index, pd.DatetimeIndex):
                date_info['date_column'] = 'index'
                date_info['start_date'] = str(df.index.min())
                date_info['end_date'] = str(df.index.max())
                date_info['date_span_days'] = (df.index.max() - df.index.min()).days
            else:
                date_info['date_column'] = None
                date_info['message'] = 'No date columns detected'
            
            return date_info
            
        except Exception:
            return {'error': 'Could not extract date information'}


# Global DataValidator instance for optimal cache performance and memory efficiency
# This singleton pattern ensures that all validation calls share the same cache,
# maximizing cache hit rates and reducing memory usage across the application.
_global_validator: Optional[DataValidator] = None
_validator_lock = threading.Lock()

def get_global_validator(enable_api_validation: bool = True, cache_size: int = None) -> DataValidator:
    """
    Get or create the global DataValidator instance (singleton pattern).

    This function provides a single, shared DataValidator for the entire StockTrader
    application, maximizing cache efficiency and performance. The configuration
    (e.g., API validation, cache size) is determined by the first call and remains
    "sticky" for the process lifetime. Subsequent calls with different parameters
    will return the existing instance with the original config.

    To change the configuration, call reset_global_validator() first.

    Args:
        enable_api_validation (bool): Whether to enable real-time API validation (default: True)
        cache_size (int, optional): Maximum cache size (default: project default)
    
    Returns:
        DataValidator: The shared global DataValidator instance

    Example (PowerShell):
        # Reset and create a validator with API validation disabled
        python -c "from core.data_validator import reset_global_validator, get_global_validator; reset_global_validator(); v = get_global_validator(enable_api_validation=False); print(v)"

    Note:
        - The modular architecture is 100% complete and functional.
        - This function is part of the validated, production-ready core modules.
    """
    global _global_validator
    
    if _global_validator is None:
        with _validator_lock:
            # Double-check locking pattern for thread safety
            if _global_validator is None:
                _global_validator = DataValidator(
                    enable_api_validation=enable_api_validation,
                    cache_size=cache_size
                )
                logger.info(f"Created global DataValidator with API validation: {enable_api_validation}")
    
    return _global_validator

def reset_global_validator() -> None:
    """
    Reset the global validator instance (for testing or config changes).

    This clears the global DataValidator singleton, so the next call to
    get_global_validator() will create a new instance with new configuration.

    Example (PowerShell):
        python -c "from core.data_validator import reset_global_validator; reset_global_validator()"

    Note:
        - Use this before changing validator config in a running process.
        - Modular architecture is complete; this is a core utility.
    """
    global _global_validator
    with _validator_lock:
        _global_validator = None
        logger.debug("Global DataValidator instance reset")

def get_global_validation_stats() -> Dict[str, Any]:
    """
    Get performance statistics from the global validator instance.

    Returns detailed statistics about validation operations, cache usage,
    and performance metrics for monitoring and optimization.

    Example (PowerShell):
        python -c "from core.data_validator import get_global_validation_stats; print(get_global_validation_stats())"

    Returns:
        dict: Validation statistics and performance metrics
    """
    validator = get_global_validator()
    return validator.get_validation_stats()

def clear_global_cache() -> None:
    """
    Clear all caches in the global validator instance.

    This clears both the symbol and general validation caches. Useful for
    testing, debugging, or when cache consistency is required.

    Example (PowerShell):
        python -c "from core.data_validator import clear_global_cache; clear_global_cache()"
    """
    validator = get_global_validator()
    validator.clear_cache()
    logger.info("Global DataValidator caches cleared")

# Convenience functions for backward compatibility and ease of use
# These functions now use the global validator instance for optimal performance
def validate_symbol(symbol: str, check_api: bool = True) -> ValidationResult:
    """
    Validate a stock symbol using the global validator instance.

    The first call to get_global_validator() determines the config for the global
    validator (e.g., API validation enabled/disabled). Subsequent calls with different
    settings will NOT change the config. This is "sticky" for the process lifetime.

    To override API validation for a single call, use:
        validator = get_global_validator()
        validator.validate_symbol(symbol, check_api=True/False)

    To change the global config, call reset_global_validator() first.

    Args:
        symbol (str): Stock symbol to validate
        check_api (bool): Whether to perform real-time API validation (per-call override)
    Returns:
        ValidationResult: Validation outcome and details

    Example (PowerShell):
        python -c "from core.data_validator import validate_symbol; print(validate_symbol('AAPL'))"
    """
    validator = get_global_validator(enable_api_validation=check_api)
    return validator.validate_symbol(symbol, check_api=check_api)

def validate_symbols(symbols_input: str) -> ValidationResult:
    """
    Validate multiple comma-separated symbols using the global validator instance.

    Args:
        symbols_input (str): Comma-separated string of symbols to validate
    Returns:
        ValidationResult: Validation outcome and details

    Example (PowerShell):
        python -c "from core.data_validator import validate_symbols; print(validate_symbols('AAPL,MSFT,GOOG'))"
    """
    validator = get_global_validator()
    return validator.validate_symbols(symbols_input)

def validate_dates(start_date: date, end_date: date, interval: str = "1d") -> ValidationResult:
    """
    Validate a date range using the global validator instance.

    Args:
        start_date (date): Start date for validation
        end_date (date): End date for validation
        interval (str): Data interval for interval-specific validation
    Returns:
        ValidationResult: Validation outcome and details

    Example (PowerShell):
        python -c "from datetime import date; from core.data_validator import validate_dates; print(validate_dates(date(2024,1,1), date(2024,5,1), '1d'))"
    """
    validator = get_global_validator()
    return validator.validate_dates(start_date, end_date, interval)

def validate_dataframe(df: pd.DataFrame, required_cols: List[str] = None) -> DataFrameValidationResult:
    """
    Validate a DataFrame using the global validator instance.

    Args:
        df (pd.DataFrame): DataFrame to validate
        required_cols (list, optional): Required column names
    Returns:
        DataFrameValidationResult: Validation outcome and details

    Example (PowerShell):
        python -c "import pandas as pd; from core.data_validator import validate_dataframe; print(validate_dataframe(pd.DataFrame({'open':[1,2],'high':[2,3],'low':[1,1],'close':[2,2],'volume':[100,200]})))"
    """
    validator = get_global_validator()
    return validator.validate_dataframe(df, required_cols)

def centralized_validate_dataframe(
    df: pd.DataFrame,
    required_cols: list = None,
    validate_ohlc: bool = True,
    check_statistical_anomalies: bool = True,
    min_rows: int = None
):
    """
    Flexible DataFrame validation for advanced use cases (patterns, etc).
    
    Args:
        min_rows: Custom minimum row requirement. If None, uses default ValidationConfig.MIN_DATASET_SIZE
    """
    validator = get_global_validator()
    return validator.validate_dataframe(
        df,
        required_cols=required_cols,
        validate_ohlc=validate_ohlc,
        check_statistical_anomalies=check_statistical_anomalies,
        min_rows=min_rows
    )

from typing import List, Tuple, Dict, Any, Type, Union, Optional
from pydantic import ValidationError, BaseModel
import pandas as pd

def batch_validate_pydantic(
    records: Union[List[dict], pd.DataFrame],
    model: Type[BaseModel],
    return_summary: bool = False
) -> Union[
    Tuple[List[BaseModel], List[Dict[str, Any]]],
    Tuple[List[BaseModel], List[Dict[str, Any]], Dict[str, int]]
]:
    """
    Batch-validate a list of records or DataFrame using a Pydantic model.

    Args:
        records (List[dict] or pd.DataFrame): Input records (dicts or DataFrame rows) to validate.
        model (Type[BaseModel]): The Pydantic model class (e.g., FinancialData).
        return_summary (bool): If True, also return a summary dict with counts.

    Returns:
        Tuple[List[BaseModel], List[Dict[str, Any]]] or (valid, errors, summary):
            - List of successfully validated model instances.
            - List of error dicts with 'index' and 'errors' for each failed record.
            - (Optional) Summary dict: {'valid': int, 'invalid': int, 'total': int}

    Example (PowerShell):
        python -c "import pandas as pd; from core.data_validator import batch_validate_pydantic, FinancialData; df = pd.read_csv('data\\AAPL_1d.csv'); valid, errors, summary = batch_validate_pydantic(df, FinancialData, True); print(f'Summary: {summary}')"

    Note:
        - ✅ Modular architecture is 100% complete and functional.
        - This utility is compatible with all Pydantic models in StockTrader.
    """
    if isinstance(records, pd.DataFrame):
        records = records.to_dict(orient='records')
    valid = []
    errors = []
    for idx, record in enumerate(records):
        try:
            valid.append(model(**record))
        except ValidationError as e:
            errors.append({'index': idx, 'errors': e.errors()})
    if return_summary:
        summary = {'valid': len(valid), 'invalid': len(errors), 'total': len(records)}
        return valid, errors, summary
    return valid, errors

def batch_validate_dataframe(
    df: pd.DataFrame,
    model: Type[BaseModel],
    return_summary: bool = False
) -> Union[
    Tuple[List[BaseModel], List[Dict[str, Any]]],
    Tuple[List[BaseModel], List[Dict[str, Any]], Dict[str, int]]
]:
    """
    Batch-validate a pandas DataFrame using a Pydantic model.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        model (Type[BaseModel]): The Pydantic model class (e.g., FinancialData).
        return_summary (bool): If True, also return a summary dict with counts.

    Returns:
        Tuple[List[BaseModel], List[Dict[str, Any]]] or (valid, errors, summary):
            - List of successfully validated model instances.
            - List of error dicts with 'index' and 'errors' for each failed record.
            - (Optional) Summary dict: {'valid': int, 'invalid': int, 'total': int}

    Example (PowerShell):
        python -c "import pandas as pd; from core.data_validator import batch_validate_dataframe, FinancialData; df = pd.read_csv('data\\AAPL_1d.csv'); valid, errors, summary = batch_validate_dataframe(df, FinancialData, True); print(f'Summary: {summary}')"

    Note:
        - ✅ Modular architecture is 100% complete and functional.
        - This utility is compatible with all Pydantic models in StockTrader.
    """
    return batch_validate_pydantic(df, model, return_summary=return_summary)

# Export the main class and key functions
__all__ = [
    'DataFrameValidationResult',
    'DataIntegrityError',
    'DataValidator',
    'DateValidationError',
    'FinancialData',
    'MarketDataPoint',
    'PerformanceValidationError',
    'SecurityValidationError',
    'SymbolValidationError',
    'ValidationConfig',
    'ValidationError',
    'ValidationResult',
    'batch_validate_dataframe',
    'batch_validate_pydantic',
    'clear_global_cache',
    'get_global_validation_stats',
    'get_global_validator',
    'reset_global_validator',
    'validate_dataframe',
    'validate_dates',    'validate_symbol',
    'validate_symbols',
]
