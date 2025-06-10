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

import logging
import re
import time
import os
import threading
from pathlib import Path
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple

import pandas as pd

# Import the moved classes
from core.validation.validation_config import ValidationConfig
from core.validation.validation_models import FinancialData, MarketDataPoint
from core.validation.validation_results import ValidationResult, DataFrameValidationResult
# Import the new helper functions for DataFrame validation
from core.validation.dataframe_validation_logic import perform_dataframe_validation_logic

# Configure logging for the module
logger = logging.getLogger(__name__)

# Custom Exception Classes
class BaseValidationError(Exception):
    """Base class for all validation errors."""
    pass

class SymbolValidationError(BaseValidationError):
    """Error during symbol validation."""
    pass

class DateValidationError(BaseValidationError):
    """Error during date validation."""
    pass

class DataIntegrityError(BaseValidationError):
    """Error related to data integrity."""
    pass

class SecurityValidationError(BaseValidationError):
    """Error related to security validation (e.g., input sanitization)."""
    pass

class PerformanceValidationError(BaseValidationError):
    """Error related to performance validation (e.g., rate limiting)."""
    pass

# ValidationConfig class has been moved to core/validation_config.py
# FinancialData and MarketDataPoint Pydantic models have been moved to core/validation_models.py
# ValidationResult and DataFrameValidationResult classes have been moved to core/validation_results.py

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
    
    def __init__(self, enable_api_validation: bool = True, cache_size: Optional[int] = None): # MODIFIED
        """
        Initialize the validator with configuration options.
        
        Args:
            enable_api_validation: Whether to perform real-time API validation
            cache_size: Maximum cache size (None for default)
        """
        self.enable_api_validation = enable_api_validation
        self.cache_size = cache_size or ValidationConfig.VALIDATION_CACHE_SIZE
        
        # Thread-safe caches
        self._symbol_cache: Dict[str, Tuple[bool, float]] = {} # Assuming this cache stores (is_valid, timestamp)
        self._validation_cache: Dict[str, ValidationResult] = {} # This cache should store ValidationResult objects
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
        
        logger.info(f"DataValidator initialized with API validation: {enable_api_validation}, Cache Size: {self.cache_size}")
    
    # ========================================
    # CORE VALIDATION METHODS
    # ========================================
    
    def validate_symbol(self, symbol: str, check_api: Optional[bool] = None) -> ValidationResult: # MODIFIED
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
        
        original_symbol_for_logging = symbol # For logging original input if it differs
        
        try:
            # Input sanitization
            if not symbol or not isinstance(symbol, str):
                logger.warning(f"Invalid symbol input type or empty: {symbol}")
                return ValidationResult(
                    is_valid=False,
                    errors=["Symbol must be a non-empty string"],
                    details={'original_symbol': symbol} # MODIFIED
                ) 
            
            # Security check
            if len(symbol) > ValidationConfig.MAX_INPUT_LENGTH:
                logger.error(f"Symbol exceeds maximum input length: {len(symbol)}")
                raise SecurityValidationError(f"Symbol exceeds maximum input length: {len(symbol)}")
            
            if self._dangerous_chars.search(symbol):
                logger.error(f"Symbol contains dangerous characters: {symbol}")
                raise SecurityValidationError("Symbol contains dangerous characters")
            
            # Basic format validation
            clean_symbol = symbol.strip().upper()
            errors: List[str] = []
            warnings_log: List[str] = [] # For logging, not part of ValidationResult directly unless they become errors
            
            if len(clean_symbol) < ValidationConfig.SYMBOL_MIN_LENGTH:
                errors.append(f"Symbol too short (minimum {ValidationConfig.SYMBOL_MIN_LENGTH} characters)")
            
            if len(clean_symbol) > ValidationConfig.SYMBOL_MAX_LENGTH:
                errors.append(f"Symbol too long (maximum {ValidationConfig.SYMBOL_MAX_LENGTH} characters)")
            
            if not self._symbol_pattern.match(clean_symbol):
                errors.append(f"Invalid symbol format: {clean_symbol}. Must contain only letters, numbers, dots, and hyphens.")
            
            if ValidationConfig.SYMBOL_REQUIRED_ALPHA and not any(c.isalpha() for c in clean_symbol):
                errors.append("Symbol must contain at least one letter.")
            
            # Early return if basic validation fails
            if errors:
                validation_time = time.time() - start_time
                logger.info(f"Symbol validation failed (basic checks) for '{original_symbol_for_logging}' -> '{clean_symbol}' in {validation_time:.4f}s. Errors: {errors}")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    details={'cleaned_symbol': clean_symbol, 'original_symbol': original_symbol_for_logging} # MODIFIED
                ) 
            
            # API validation (if enabled) - Placeholder for actual API call logic
            api_valid = True # Assume true if not checked or if API check passes
            api_checked = False
            
            # Cache check
            with self._cache_lock:
                cached_result = self._get_from_cache(f"symbol:{clean_symbol}")
            if cached_result:
                self._validation_stats['cache_hits'] += 1
                # MODIFIED: Update details if necessary, validated_data is not a direct attribute
                if cached_result.details:
                    cached_result.details['cleaned_symbol_on_cache_hit'] = clean_symbol 
                else:
                    cached_result.details = {'cleaned_symbol_on_cache_hit': clean_symbol}
                logger.info(f"Symbol validation cache hit for '{clean_symbol}'. Time: {time.time() - start_time:.4f}s")
                return cached_result

            self._validation_stats['cache_misses'] += 1

            should_check_api = check_api if check_api is not None else self.enable_api_validation
            if should_check_api:
                api_checked = True
                self._validation_stats['api_calls'] += 1
                logger.debug(f"Performing API validation for symbol: {clean_symbol}")
                # --- Placeholder for actual API call ---
                # try:
                #     api_response = some_api_library.validate_symbol(clean_symbol)
                #     if not api_response.is_valid:
                #         api_valid = False
                #         errors.append(f"API validation failed: {api_response.message}")
                # except Exception as api_e:
                #     api_valid = False
                #     errors.append(f"API validation error: {str(api_e)}")
                #     logger.warning(f"API call failed for symbol {clean_symbol}: {api_e}")
                # --- End Placeholder ---
                # For now, let's assume API validation passes if attempted
                if not errors: # Only if no other errors yet
                    pass # api_valid remains true

            validation_time = time.time() - start_time
            log_metadata = {
                'validation_time_seconds': round(validation_time, 4),
                'api_validated': api_valid and api_checked, # True only if API was checked and passed
                'api_checked': api_checked,
                'cleaned_symbol': clean_symbol,
                'original_symbol': original_symbol_for_logging,
                'symbol_changed_during_cleaning': clean_symbol != symbol.strip() # More precise cleaning check
            }

            if warnings_log:
                 log_metadata['warnings'] = warnings_log
            
            if errors: # If API validation (or other checks) added errors
                logger.info(f"Symbol validation failed for '{original_symbol_for_logging}' -> '{clean_symbol}'. Metadata: {log_metadata}. Errors: {errors}")
                result = ValidationResult(is_valid=False, errors=errors, details={**log_metadata, 'final_status': 'failed'}) # MODIFIED
            else:
                logger.info(f"Symbol validation successful for '{original_symbol_for_logging}' -> '{clean_symbol}'. Metadata: {log_metadata}")
                result = ValidationResult(is_valid=True, errors=None, details={**log_metadata, 'final_status': 'successful', 'validated_symbol': clean_symbol}) # MODIFIED
            
            with self._cache_lock:
                self._add_to_cache(f"symbol:{clean_symbol}", result)
            return result
            
        except SecurityValidationError as sve: # Catch specific security errors first
            self._validation_stats['validation_errors'] += 1
            logger.error(f"Security validation error for symbol '{original_symbol_for_logging}': {sve}", exc_info=True)
            # Do not return potentially unsafe data in validated_data for security issues
            return ValidationResult(is_valid=False, errors=[f"Security error: {str(sve)}"], details={'original_symbol': original_symbol_for_logging, 'error_type': 'SecurityValidationError'}) # MODIFIED
        except Exception as e:
            self._validation_stats['validation_errors'] += 1
            validation_time = time.time() - start_time
            logger.error(f"Unexpected symbol validation error for '{original_symbol_for_logging}': {e}. Time: {validation_time:.4f}s", exc_info=True)
            return ValidationResult(
                is_valid=False,
                errors=[f"Unexpected validation error: {str(e)}"],
                details={'original_symbol': original_symbol_for_logging, 'error_type': 'Exception'} # MODIFIED
            ) 
    
    def validate_symbols(self, symbols_input: str, max_symbols: int = 50, check_api: Optional[bool] = None) -> ValidationResult: # MODIFIED
        """
        Validate multiple comma-separated symbols with batch processing.
        
        Args:
            symbols_input: Comma-separated list of symbols
            max_symbols: Maximum number of symbols allowed
            check_api: Override API checking for individual symbols
            
        Returns:
            ValidationResult with list of valid symbols in validated_data, and aggregated errors.
        """
        start_time = time.time()
        self._validation_stats['symbol_validations'] += len(symbols_input.split(',')) # Approximate count

        if not symbols_input or not isinstance(symbols_input, str) or not symbols_input.strip():
            logger.info("No symbols provided for batch validation.")
            return ValidationResult(is_valid=False, errors=["No symbols provided"], details={'input_symbols': symbols_input}) # MODIFIED
        
        raw_symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]
        
        if not raw_symbols: # Handles case of " , , "
             logger.info("No valid symbols found after stripping input.")
             return ValidationResult(is_valid=False, errors=["No symbols provided after cleaning input"], details={'input_symbols': symbols_input}) # MODIFIED

        if len(raw_symbols) > max_symbols:
            err_msg = f"Too many symbols: {len(raw_symbols)} (maximum {max_symbols})"
            logger.warning(err_msg)
            return ValidationResult(is_valid=False, errors=[err_msg], details={'input_symbols': symbols_input, 'raw_symbol_count': len(raw_symbols)}) # MODIFIED

        valid_symbols: List[str] = []
        all_errors: List[str] = []
        
        # Deduplicate symbols to avoid redundant validation, but preserve original order for user feedback if needed.
        # For simplicity here, we process unique symbols and report based on them.
        unique_symbols_to_validate = sorted(list(set(raw_symbols))) # Process unique symbols
        
        logger.info(f"Starting batch validation for {len(unique_symbols_to_validate)} unique symbols (from {len(raw_symbols)} raw).")

        for sym_idx, symbol_str in enumerate(unique_symbols_to_validate):
            logger.debug(f"Batch validating symbol {sym_idx+1}/{len(unique_symbols_to_validate)}: '{symbol_str}'")
            # Pass the check_api override down
            result = self.validate_symbol(symbol_str, check_api=check_api)
            if result.is_valid and result.details and isinstance(result.details.get('validated_symbol'), str): # MODIFIED: Check details
                valid_symbols.append(result.details['validated_symbol'])
            elif result.is_valid and not (result.details and isinstance(result.details.get('validated_symbol'), str)): # MODIFIED
                logger.error(f"validate_symbol was valid but 'validated_symbol' not found or not string in details: {result.details} for symbol {symbol_str}")
                all_errors.append(f"{symbol_str}: Internal error - validated_symbol missing in details from successful validation.")
            elif result.errors:
                for error_message in result.errors:
                    all_errors.append(f"{symbol_str}: {error_message}")
            else: # Should have errors if not valid
                 all_errors.append(f"{symbol_str}: Unknown validation failure.")


        validation_time = time.time() - start_time
        # Filter valid_symbols to be unique again, in case duplicates were valid under different original forms
        # but cleaned to the same valid symbol.
        final_valid_symbols = sorted(list(set(valid_symbols)))

        log_metadata = {
            'validation_time_seconds': round(validation_time, 4),
            'total_symbols_input': len(raw_symbols),
            'unique_symbols_processed': len(unique_symbols_to_validate),
            'valid_symbols_count': len(final_valid_symbols),
            'error_count': len(all_errors)
        }

        if all_errors:
            logger.info(f"Batch symbol validation completed with errors. Valid: {len(final_valid_symbols)}. Errors: {len(all_errors)}. Metadata: {log_metadata}")
            # If some symbols are valid, we can still return them. is_valid reflects if ALL were valid.
            # The definition of is_valid for a batch can vary. Here, True if no errors at all.
            return ValidationResult(
                is_valid=False, # Or: is_valid = not all_errors and bool(final_valid_symbols)
                errors=all_errors,
                details={**log_metadata, 'final_valid_symbols': final_valid_symbols} # MODIFIED
            ) 
        elif not final_valid_symbols: # No errors, but also no valid symbols (e.g. empty input that wasn't caught earlier)
            logger.info(f"Batch symbol validation completed. No valid symbols found. Metadata: {log_metadata}")
            return ValidationResult(is_valid=False, errors=["No valid symbols found."], details=log_metadata) # MODIFIED
        else:
            logger.info(f"Batch symbol validation successful. All {len(final_valid_symbols)} symbols valid. Metadata: {log_metadata}")
            return ValidationResult(
                is_valid=True, 
                errors=None,
                details={**log_metadata, 'final_valid_symbols': final_valid_symbols} # MODIFIED
            ) 
            
    def validate_dates(self, start_date: date, end_date: date, interval: str = "1d") -> ValidationResult: # MODIFIED
        """
        Validate start and end dates with interval-specific rules.
        
        Args:
            start_date: The start date (datetime.date object)
            end_date: The end date (datetime.date object)
            interval: Data interval (e.g., "1d", "1h", "5m")
            
        Returns:
            ValidationResult with validation outcome. Validated data contains a dict of dates.
        """
        start_time = time.time()
        self._validation_stats['date_validations'] += 1
        errors: List[str] = []
        warnings_log: List[str] = []

        if not isinstance(start_date, date) or not isinstance(end_date, date):
            errors.append("Start and end dates must be valid date objects.")
            # Fall through for further checks if types are wrong, or return early:
            # logger.warning(f"Invalid date types: start_date ({type(start_date)}), end_date ({type(end_date)})")
            # return ValidationResult(is_valid=False, errors=errors, validated_data=None)

        # Basic checks
        if start_date > end_date:
            errors.append("Start date cannot be after end date.")
        
        today = date.today()
        if start_date > today:
            warnings_log.append("Start date is in the future.") # Often a warning, not an error
        if end_date > today:
            # Depending on use case, future end date might be an error or warning
            warnings_log.append("End date is in the future. Data may not be available.")


        # Historical data limits (example)
        min_historical_date = getattr(ValidationConfig, 'MIN_HISTORICAL_DATE', date(1970, 1, 1))
        if start_date < min_historical_date:
            errors.append(f"Start date {start_date} is before the allowed minimum historical date {min_historical_date}.")

        # Interval-specific limits (example)
        # Use INTERVAL_LIMITS from ValidationConfig directly
        interval_limits_map = getattr(ValidationConfig, 'INTERVAL_LIMITS', {})
        max_days = interval_limits_map.get(interval) # This will be None if interval not in map
        
        if max_days is not None and (end_date - start_date).days > max_days:
            errors.append(f"Date range exceeds maximum of {max_days} days for interval '{interval}'.")
            suggested_end = start_date + timedelta(days=max_days)
            warnings_log.append(f"Consider reducing end date to {suggested_end} or earlier for interval '{interval}'.")
        elif max_days is None: # Interval not found in config, could be a warning or default behavior
            warnings_log.append(f"Interval '{interval}' not found in ValidationConfig.INTERVAL_LIMITS. No specific day limit applied.")

        # Check against max lookback period from today
        max_lookback_days = getattr(ValidationConfig, 'MAX_LOOKBACK_PERIOD_DAYS', 365*20) 
        if (today - start_date).days > max_lookback_days:
             warnings_log.append(f"Start date is further than {max_lookback_days} days in the past. Data availability might be limited.")


        validation_time = time.time() - start_time
        log_metadata = {
            'validation_time_seconds': round(validation_time, 4),
            'interval': interval,
            'start_date': str(start_date),
            'end_date': str(end_date),
            'days_in_range': (end_date - start_date).days if isinstance(start_date, date) and isinstance(end_date, date) else 'N/A'
        }
        if warnings_log:
            log_metadata['warnings'] = warnings_log

        if errors:
            logger.info(f"Date validation failed. Metadata: {log_metadata}. Errors: {errors}")
            return ValidationResult(
                is_valid=False, 
                errors=errors, 
                details={**log_metadata, 'input_start_date': start_date, 'input_end_date': end_date, 'input_interval': interval} # MODIFIED
            ) 
        else:
            logger.info(f"Date validation successful. Metadata: {log_metadata}")
            return ValidationResult(
                is_valid=True, 
                errors=None, 
                details={**log_metadata, 'validated_start_date': start_date, 'validated_end_date': end_date, 'validated_interval': interval} # MODIFIED
            ) 

    def validate_dataframe(self, 
                         df: pd.DataFrame, 
                         required_cols: Optional[List[str]] = None, # MODIFIED
                         check_ohlcv: bool = True,
                         min_rows: Optional[int] = 1, # MODIFIED
                         max_rows: Optional[int] = None, # MODIFIED
                         detect_anomalies_level: Optional[str] = None, # e.g., "basic", "advanced"
                         max_null_percentage: float = 0.1 # Max 10% nulls per column by default
                        ) -> DataFrameValidationResult:
        """
        Comprehensive DataFrame validation for financial time series data.
        
        Args:
            df: Pandas DataFrame to validate.
            required_cols: List of required column names. If None, uses default OHLCV.
            check_ohlcv: Whether to perform OHLCV-specific checks.
            min_rows: Minimum required rows.
            max_rows: Maximum allowed rows.
            detect_anomalies_level: Level of anomaly detection ('basic', 'advanced', None).
            max_null_percentage: Maximum allowable percentage of nulls per column.

        Returns:
            DataFrameValidationResult with validation outcome and details.
        """
        # start_time = time.time() # Removed unused variable
        self._validation_stats['dataframe_validations'] += 1
        # start_time is now managed by this calling function for overall duration
        # The core logic is moved to perform_dataframe_validation_logic

        # Delegate to the extracted logic function
        # Note: The original start_time is implicitly handled by the caller if needed for overall stats.
        # The perform_dataframe_validation_logic function will log its own execution time for its specific block.
        return perform_dataframe_validation_logic(
            df=df,
            required_cols=required_cols,
            check_ohlcv=check_ohlcv,
            min_rows=min_rows,
            max_rows=max_rows,
            detect_anomalies_level=detect_anomalies_level,
            max_null_percentage=max_null_percentage
        )

    def validate_price(self, price: Union[float, str], 
                       min_price: Optional[float] = 0.0001,  
                       max_price: Optional[float] = getattr(ValidationConfig, 'MAX_PRICE', 1000000.0) # Added getattr
                      ) -> ValidationResult:
        """Validate a single price value."""
        start_time = time.time()
        errors: List[str] = []
        warnings_log: List[str] = []
        validated_price: Optional[float] = None
        original_input_for_details = price # Store original input for details

        try:
            if isinstance(price, str):
                price_str = price.strip().replace(',', '') # Remove commas for float conversion
                if not price_str: # Empty string after strip
                    errors.append("Price string is empty.")
                else:
                    try:
                        validated_price = float(price_str)
                    except ValueError:
                        errors.append(f"Invalid price format: '{price}'. Cannot convert to float.")
            elif isinstance(price, (int, float)):
                validated_price = float(price)
            else:
                errors.append(f"Invalid price type: {type(price)}. Expected float or string.")

            if validated_price is not None: # Only proceed if conversion was successful
                if validated_price <= 0 and min_price is not None and min_price > 0: # Special check for non-positive if min_price expects positive
                     errors.append(f"Price must be positive. Got: {validated_price}")
                if min_price is not None and validated_price < min_price:
                    errors.append(f"Price {validated_price} is below minimum allowed {min_price}.")
                if max_price is not None and validated_price > max_price:
                    errors.append(f"Price {validated_price} exceeds maximum allowed {max_price}.")
                
                # Precision check (example)
                price_precision = getattr(ValidationConfig, 'PRICE_PRECISION', 4) # Added getattr
                if abs(validated_price - round(validated_price, price_precision)) > 1e-9: 
                    original_precision_price = validated_price
                    validated_price = round(validated_price, price_precision)
                    warnings_log.append(f"Price precision adjusted from {original_precision_price} to {validated_price} (max {price_precision} decimals).")

            validation_time = time.time() - start_time
            log_metadata = {
                'validation_time_seconds': round(validation_time, 4),
                'original_input': original_input_for_details,
                'min_price': min_price,
                'max_price': max_price,
                'price_precision': getattr(ValidationConfig, 'PRICE_PRECISION', 4) # Added getattr
            }
            if warnings_log:
                log_metadata['warnings'] = warnings_log

            if errors:
                logger.warning(f"Price validation failed for '{original_input_for_details}'. Metadata: {log_metadata}. Errors: {errors}")
                return ValidationResult(
                    is_valid=False, 
                    errors=errors, 
                    details={**log_metadata, 'converted_value': validated_price} # MODIFIED
                )
            else:
                logger.info(f"Price validation successful for '{original_input_for_details}'. Validated: {validated_price}. Metadata: {log_metadata}")
                return ValidationResult(
                    is_valid=True, 
                    errors=None, 
                    details={**log_metadata, 'validated_price': validated_price} # MODIFIED
                )
        
        except Exception as e:
            logger.error(f"Unexpected error during price validation for '{original_input_for_details}': {e}", exc_info=True)
            return ValidationResult(
                is_valid=False, 
                errors=[f"Unexpected price validation error: {str(e)}"], 
                details={'original_input': original_input_for_details, 'error_type': 'Exception'} # MODIFIED
            )


    def validate_quantity(self, quantity: Union[int, float, str], 
                          min_quantity: Optional[Union[int, float]] = 0, # MODIFIED
                          max_quantity: Optional[Union[int, float]] = ValidationConfig.MAX_VOLUME, # Assuming volume can be quantity
                          allow_fractional: bool = False
                         ) -> ValidationResult:
        """Validate a single quantity value."""
        start_time = time.time()
        errors: List[str] = []
        warnings_log: List[str] = []
        validated_quantity: Optional[Union[int, float]] = None
        original_input_for_details = quantity # Store original input

        try:
            if isinstance(quantity, str):
                qty_str = quantity.strip().replace(',', '')
                if not qty_str:
                    errors.append("Quantity string is empty.")
                else:
                    try:
                        if allow_fractional:
                            validated_quantity = float(qty_str)
                        else:
                            # Try float first to catch "1.0", then int
                            temp_float = float(qty_str)
                            if temp_float == int(temp_float):
                                validated_quantity = int(temp_float)
                            else:
                                errors.append(f"Fractional quantity '{quantity}' not allowed.")
                    except ValueError:
                        errors.append(f"Invalid quantity format: '{quantity}'. Cannot convert.")
            elif isinstance(quantity, (int, float)):
                if not allow_fractional and isinstance(quantity, float) and quantity != int(quantity):
                    errors.append(f"Fractional quantity {quantity} not allowed.")
                else:
                    validated_quantity = quantity if allow_fractional else int(quantity)
            else:
                errors.append(f"Invalid quantity type: {type(quantity)}. Expected number or string.")

            if validated_quantity is not None:
                if min_quantity is not None and validated_quantity < min_quantity:
                    errors.append(f"Quantity {validated_quantity} is below minimum allowed {min_quantity}.")
                if max_quantity is not None and validated_quantity > max_quantity:
                    errors.append(f"Quantity {validated_quantity} exceeds maximum allowed {max_quantity}.")
            
            validation_time = time.time() - start_time
            log_metadata = {
                'validation_time_seconds': round(validation_time, 4),
                'original_input': original_input_for_details,
                'min_quantity': min_quantity,
                'max_quantity': max_quantity,
                'allow_fractional': allow_fractional
            }
            if warnings_log: # Though none are added in current logic
                log_metadata['warnings'] = warnings_log

            if errors:
                logger.warning(f"Quantity validation failed for '{original_input_for_details}'. Metadata: {log_metadata}. Errors: {errors}")
                return ValidationResult(
                    is_valid=False, 
                    errors=errors, 
                    details={**log_metadata, 'converted_value': validated_quantity} # MODIFIED
                )
            else:
                logger.info(f"Quantity validation successful for '{original_input_for_details}'. Validated: {validated_quantity}. Metadata: {log_metadata}")
                return ValidationResult(
                    is_valid=True, 
                    errors=None, 
                    details={**log_metadata, 'validated_quantity': validated_quantity} # MODIFIED
                )

        except Exception as e:
            logger.error(f"Unexpected error during quantity validation for '{original_input_for_details}': {e}", exc_info=True)
            return ValidationResult(
                is_valid=False, 
                errors=[f"Unexpected quantity validation error: {str(e)}"], 
                details={'original_input': original_input_for_details, 'error_type': 'Exception'} # MODIFIED
            )

    def validate_percentage(self, percentage: Union[float, str], 
                            min_pct: Optional[float] = 0.0, # MODIFIED
                            max_pct: Optional[float] = 1.0 # MODIFIED (assuming 0.0 to 1.0 scale)
                           ) -> ValidationResult:
        """Validate a percentage value (expected as a decimal, e.g., 0.75 for 75%)."""
        start_time = time.time()
        errors: List[str] = []
        validated_percentage: Optional[float] = None
        original_input_for_details = percentage # Store original input

        try:
            if isinstance(percentage, str):
                pct_str = percentage.strip().replace('%', '') # Remove % if present
                if not pct_str:
                    errors.append("Percentage string is empty.")
                else:
                    try:
                        # If user enters "75" for 75%, convert to 0.75
                        val = float(pct_str)
                        if val > 1.0 and max_pct is not None and max_pct <=1.0 : # Heuristic: if they entered 75 instead of 0.75
                            # Check if it makes sense to divide by 100
                            if (val / 100 >= (min_pct or 0.0)) and (val / 100 <= (max_pct or 1.0)):
                                validated_percentage = val / 100.0
                                logger.info(f"Interpreted percentage input '{percentage}' as {validated_percentage*100}% (i.e. {validated_percentage})")
                            else: # Doesn't make sense, treat as error or direct value
                                validated_percentage = val # Or error: errors.append("Percentage seems to be out of 0-100 range if interpreted as direct percentage points")
                        else:
                             validated_percentage = val
                    except ValueError:
                        errors.append(f"Invalid percentage format: '{percentage}'.")
            elif isinstance(percentage, (int, float)):
                validated_percentage = float(percentage)
            else:
                errors.append(f"Invalid percentage type: {type(percentage)}. Expected float or string.")

            if validated_percentage is not None:
                if min_pct is not None and validated_percentage < min_pct:
                    errors.append(f"Percentage {validated_percentage:.2%} is below minimum {min_pct:.0%}.")
                if max_pct is not None and validated_percentage > max_pct:
                    errors.append(f"Percentage {validated_percentage:.2%} exceeds maximum {max_pct:.0%}.")

            validation_time = time.time() - start_time
            log_metadata = {
                'validation_time_seconds': round(validation_time, 4),
                'original_input': original_input_for_details,
                'min_percentage': min_pct,
                'max_percentage': max_pct,
                'as_percentage_string': f"{validated_percentage*100:.2f}%" if validated_percentage is not None else "N/A"
            }

            if errors:
                logger.warning(f"Percentage validation failed for '{original_input_for_details}'. Metadata: {log_metadata}. Errors: {errors}")
                return ValidationResult(
                    is_valid=False, 
                    errors=errors, 
                    details={**log_metadata, 'converted_value': validated_percentage} # MODIFIED
                )
            else:
                logger.info(f"Percentage validation successful for '{original_input_for_details}'. Validated: {validated_percentage}. Metadata: {log_metadata}")
                return ValidationResult(
                    is_valid=True, 
                    errors=None, 
                    details={**log_metadata, 'validated_percentage': validated_percentage} # MODIFIED
                )
        
        except Exception as e:
            logger.error(f"Unexpected error during percentage validation for '{original_input_for_details}': {e}", exc_info=True)
            return ValidationResult(
                is_valid=False, 
                errors=[f"Unexpected percentage validation error: {str(e)}"], 
                details={'original_input': original_input_for_details, 'error_type': 'Exception'} # MODIFIED
            )


    def sanitize_input(self, input_str: str, 
                       max_length: Optional[int] = ValidationConfig.MAX_INPUT_LENGTH, # MODIFIED
                       allow_html: bool = False
                      ) -> str:
        """
        Sanitize user input strings to prevent XSS and other injection attacks.
        This is a basic sanitizer. For robust HTML sanitization, use a dedicated library.
        """
        if not isinstance(input_str, str):
            logger.warning(f"Sanitize input called with non-string type: {type(input_str)}. Returning empty string.")
            return ""

        # Truncate if too long
        if max_length is not None and len(input_str) > max_length:
            input_str = input_str[:max_length]
            logger.debug(f"Input string truncated to {max_length} characters.")

        if allow_html:
            # Basic HTML sanitization (very limited, consider a proper library like bleach)
            # This example only escapes common XSS vectors.
            sanitized_str = input_str.replace('<', '&lt;').replace('>', '&gt;')
            # Potentially more rules here or use a library
            logger.debug("HTML characters escaped for allowed HTML input.")
        else:
            # Strip all HTML tags if not allowed
            sanitized_str = re.sub(r'<[^>]*>', '', input_str)
            # Escape other potentially dangerous characters
            sanitized_str = sanitized_str.replace('&', '&amp;').replace('"', '&quot;').replace("'", '&#x27;').replace('/', '&#x2F;')
            logger.debug("HTML tags stripped and special characters escaped for non-HTML input.")
        
        # Remove or replace dangerous characters based on config (already used in symbol validation)
        # This might be redundant if _dangerous_chars pattern is comprehensive
        # sanitized_str = self._dangerous_chars.sub('', sanitized_str) # Example: remove them

        return sanitized_str.strip()


    def validate_file_path(self, file_path: Union[str, Path], 
                           base_directory: Optional[Path] = None, 
                           must_exist: bool = False, 
                           allowed_extensions: Optional[List[str]] = None, 
                           check_read_access: bool = False 
                          ) -> ValidationResult:
        """Validate a file path for security and existence."""
        start_time = time.time()
        errors: List[str] = []
        warnings_log: List[str] = []
        original_input_path_str = str(file_path) # Store original input for details
        
        try:
            path = Path(file_path)
            # Initialize resolved_path to None or original path string for cases where resolution fails early
            resolved_path_str: Optional[str] = None 

            # Path traversal check
            if base_directory:
                base_directory = Path(base_directory).resolve()
                # Resolve path carefully, it might not exist yet
                try:
                    # Attempt to resolve, but catch if it doesn't exist and must_exist is false
                    # For security, we want to work with absolute paths if possible
                    absolute_path_attempt = Path(os.path.abspath(path))
                except Exception: # Broad exception for os.path.abspath issues
                    absolute_path_attempt = path # Fallback to original path object

                if absolute_path_attempt.is_absolute():
                    resolved_path_obj = absolute_path_attempt
                else: # If still not absolute (e.g. empty path string), make it relative to CWD for consistent checks
                    resolved_path_obj = Path.cwd() / absolute_path_attempt
                
                resolved_path_str = str(resolved_path_obj.resolve()) # Final resolved string

                if base_directory not in resolved_path_obj.resolve().parents and base_directory != resolved_path_obj.resolve() :
                    try:
                        resolved_path_obj.resolve().relative_to(base_directory)
                    except ValueError:
                         errors.append("Path traversal attempt detected or path is outside the allowed base directory.")
                         logger.error(f"Path traversal attempt: {original_input_path_str} (resolved: {resolved_path_str}) is outside base: {base_directory}")
            else: # If no base_directory, resolve normally
                try:
                    resolved_path_obj = path.resolve()
                    resolved_path_str = str(resolved_path_obj)
                except Exception as resolve_err: # Catch errors if path is invalid (e.g. contains null bytes on some OS)
                    errors.append(f"Invalid file path provided: {original_input_path_str}. Error: {resolve_err}")
                    # resolved_path_str remains None or its initial value if path object was created

            current_resolved_path_for_checks = Path(resolved_path_str) if resolved_path_str else path

            if not errors: # Continue if no traversal or resolution error
                if must_exist and not current_resolved_path_for_checks.exists():
                    errors.append(f"File or directory does not exist: {current_resolved_path_for_checks}")
                elif current_resolved_path_for_checks.exists() and check_read_access and not os.access(current_resolved_path_for_checks, os.R_OK):
                    errors.append(f"No read access to path: {current_resolved_path_for_checks}")

                if allowed_extensions and current_resolved_path_for_checks.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                    errors.append(f"Invalid file extension: '{current_resolved_path_for_checks.suffix}'. Allowed: {', '.join(allowed_extensions)}")
            
            max_fp_len = getattr(ValidationConfig, 'MAX_FILE_PATH_LENGTH', 1024) 
            if len(str(current_resolved_path_for_checks)) > max_fp_len:
                errors.append(f"File path is too long (max {max_fp_len} chars). Path: {str(current_resolved_path_for_checks)[:200]}...")


            validation_time = time.time() - start_time
            log_metadata = {
                'validation_time_seconds': round(validation_time, 4),
                'original_path': original_input_path_str,
                'resolved_path_attempt': resolved_path_str, # This is the best effort resolved path
                'base_directory': str(base_directory) if base_directory else 'N/A',
                'must_exist': must_exist,
                'allowed_extensions': allowed_extensions
            }
            if warnings_log:
                log_metadata['warnings'] = warnings_log

            if errors:
                logger.warning(f"File path validation failed for '{original_input_path_str}'. Metadata: {log_metadata}. Errors: {errors}")
                return ValidationResult(
                    is_valid=False, 
                    errors=errors, 
                    details=log_metadata # MODIFIED
                )
            else:
                logger.info(f"File path validation successful for '{original_input_path_str}'. Metadata: {log_metadata}")
                # Ensure 'resolved_path' key exists with the successfully validated path
                log_metadata['resolved_path'] = resolved_path_str 
                return ValidationResult(
                    is_valid=True, 
                    errors=None, 
                    details=log_metadata # MODIFIED
                )
        
        except Exception as e:
            logger.error(f"Unexpected error during file path validation for '{original_input_path_str}': {e}", exc_info=True)
            # Try to get resolved_path_str if it was set before exception
            current_resolved_str = resolved_path_str if 'resolved_path_str' in locals() and resolved_path_str is not None else original_input_path_str
            return ValidationResult(
                is_valid=False, 
                errors=[f"Unexpected path validation error: {str(e)}"], 
                details={'original_input': original_input_path_str, 'resolved_path_attempt': current_resolved_str, 'error_type': 'Exception'} # MODIFIED
            )

    # ========================================
    # INTERNAL HELPER METHODS
    # ========================================

    def _get_from_cache(self, key: str) -> Optional[ValidationResult]:
        """Retrieve an item from the validation cache if not expired."""
        with self._cache_lock:
            cached_item = self._validation_cache.get(key)
            if cached_item:
                # Assuming ValidationResult itself doesn't have a timestamp for expiry.
                # The _symbol_cache example had (bool, float) for (is_valid, timestamp).
                # For _validation_cache, we might need to store ValidationResult along with a timestamp.
                # For simplicity, let's assume the _validation_cache stores (ValidationResult, timestamp)
                # This part needs refinement if different TTLs or cache structures are used.
                
                # Simplified: if it's in cache, it's good for now.
                # A more robust cache would have its own timestamp.
                # Let's assume the cache stores ValidationResult directly and has no TTL here.
                # This is a simplification from the original _symbol_cache logic.
                if key in self._validation_cache: # Check again due to potential race if lock is finer
                    logger.debug(f"Cache hit for {key}")
                    return self._validation_cache[key]
            return None

    def _add_to_cache(self, key: str, result: ValidationResult):
        """Add an item to the validation cache, managing cache size."""
        with self._cache_lock:
            if len(self._validation_cache) >= self.cache_size:
                # Simple FIFO eviction if cache is full
                try:
                    oldest_key = next(iter(self._validation_cache))
                    del self._validation_cache[oldest_key]
                    logger.debug(f"Cache full. Evicted oldest item: {oldest_key}")
                except StopIteration: # Should not happen if len > 0
                    pass 
            self._validation_cache[key] = result
            # If storing with timestamp: self._validation_cache[key] = (result, time.time())
            logger.debug(f"Added to cache: {key}")
            
    # _validate_ohlc_data and _detect_anomalies are now standalone functions
    # in dataframe_validation_logic.py and are named _validate_ohlc_logic
    # and _detect_anomalies_logic respectively.
    # They are called by perform_dataframe_validation_logic.

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get a copy of the current validation statistics."""
        with self._cache_lock:
            return {k: v for k, v in self._validation_stats.items()}

# ========================================
# GLOBAL VALIDATOR INSTANCE & FUNCTIONS (Optional)
# ========================================
_global_validator_instance: Optional[DataValidator] = None
_validator_lock = threading.Lock()

def get_global_validator(enable_api_validation: bool = True, 
                         cache_size: Optional[int] = None # MODIFIED
                        ) -> DataValidator:
    """Get a global instance of DataValidator (thread-safe)."""
    global _global_validator_instance
    if _global_validator_instance is None:
        with _validator_lock:
            if _global_validator_instance is None: # Double-check locking
                logger.info("Creating global DataValidator instance.")
                _global_validator_instance = DataValidator(
                    enable_api_validation=enable_api_validation,
                    cache_size=cache_size
                )
    # Potentially update config if called with different params - or make it truly singleton
    # For now, it returns the first created instance or creates one.
    return _global_validator_instance

# Standalone validation functions using the global validator (examples)
# These might be useful for direct calls without managing a DataValidator instance.

def validate_symbol(symbol: str, check_api: Optional[bool] = None) -> ValidationResult: # MODIFIED
    """Standalone symbol validation using the global validator."""
    validator = get_global_validator(check_api if check_api is not None else True) # Ensure API check is enabled if specified
    return validator.validate_symbol(symbol, check_api=check_api)

def validate_dates(start_date: date, end_date: date, interval: str = "1d") -> ValidationResult: # MODIFIED
    """Standalone date validation using the global validator."""
    validator = get_global_validator()
    return validator.validate_dates(start_date, end_date, interval)

def validate_dataframe(df: pd.DataFrame, 
                       required_cols: Optional[List[str]] = None, # MODIFIED
                       check_ohlcv: bool = True,
                       min_rows: Optional[int] = 1 # MODIFIED
                      ) -> DataFrameValidationResult:
    """Standalone DataFrame validation using the global validator."""
    validator = get_global_validator()
    # Note: validate_dataframe has more params, this is a simplified wrapper
    return validator.validate_dataframe(df, required_cols=required_cols, check_ohlcv=check_ohlcv, min_rows=min_rows)

def validate_symbols(symbols_input: str, max_symbols: int = 50, check_api: Optional[bool] = None) -> ValidationResult:
    """Standalone batch symbol validation using the global validator."""
    validator = get_global_validator(check_api if check_api is not None else True)
    return validator.validate_symbols(symbols_input, max_symbols=max_symbols, check_api=check_api)


# Expose main classes and functions for import
__all__ = [
    'DataValidator',
    'ValidationResult', # From .validation_results
    'DataFrameValidationResult', # From .validation_results
    'FinancialData', # From .validation_models
    'MarketDataPoint', # From .validation_models
    'ValidationConfig', # From .validation_config
    'get_global_validator',
    'validate_symbol', # Standalone function
    'validate_symbols', # Standalone function - ADDED
    'validate_dates',  # Standalone function
    'validate_dataframe', # Standalone function
    'BaseValidationError',
    'SymbolValidationError',
    'DateValidationError',
    'DataIntegrityError',
    'SecurityValidationError',
    'PerformanceValidationError'
]

# Example usage (for testing or demonstration)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    validator = DataValidator(enable_api_validation=False) # API validation off for local test
    
    # Symbol validation
    logger.info("--- Symbol Validation ---")
    res_sym_valid = validator.validate_symbol("AAPL")
    logger.info(f"AAPL: valid={res_sym_valid.is_valid}, details={res_sym_valid.details}, errors={res_sym_valid.errors}") 
    res_sym_invalid = validator.validate_symbol("INVALID$SYMBOL")
    logger.info(f"INVALID$SYMBOL: valid={res_sym_invalid.is_valid}, details={res_sym_invalid.details}, errors={res_sym_invalid.errors}") 
    res_sym_empty = validator.validate_symbol("")
    logger.info(f"Empty Symbol: valid={res_sym_empty.is_valid}, details={res_sym_empty.details}, errors={res_sym_empty.errors}") 

    # Batch symbol validation
    logger.info("--- Batch Symbol Validation ---")
    res_batch = validator.validate_symbols("MSFT, GOOG, BAD!,TSLA, ,NVDA")
    logger.info(f"Batch: valid={res_batch.is_valid}, details={res_batch.details}, errors={res_batch.errors}") 

    # Date validation
    logger.info("--- Date Validation ---")
    d_start = date(2023, 1, 1)
    d_end = date(2023, 1, 30)
    res_date_valid = validator.validate_dates(d_start, d_end, "1d")
    logger.info(f"Dates {d_start}-{d_end} (1d): valid={res_date_valid.is_valid}, details={res_date_valid.details}, errors={res_date_valid.errors}") 
    
    d_start_invalid = date(2023, 2, 1)
    d_end_invalid = date(2023, 1, 1)
    res_date_invalid = validator.validate_dates(d_start_invalid, d_end_invalid)
    logger.info(f"Dates {d_start_invalid}-{d_end_invalid}: valid={res_date_invalid.is_valid}, details={res_date_invalid.details}, errors={res_date_invalid.errors}") 

    # DataFrame validation
    logger.info("--- DataFrame Validation ---")
    good_df_data = {
        'Open': [10, 11, 12], 'High': [10.5, 11.5, 12.5], 'Low': [9.5, 10.5, 11.5], 
        'Close': [10.2, 11.2, 12.2], 'Volume': [1000, 1100, 1200]
    }
    good_df = pd.DataFrame(good_df_data)
    res_df_valid = validator.validate_dataframe(good_df, required_cols=['Open', 'High', 'Low', 'Close', 'Volume'])
    logger.info(f"Good DF: valid={res_df_valid.is_valid}, errors={res_df_valid.errors}, error_details={res_df_valid.error_details}")
    # Access validated data: good_df_validated = res_df_valid.validated_data

    bad_df_data = {'Open': [10, 9], 'High': [9.5, 10], 'Low': [10.1, 8], 'Close': [9.8, 9], 'Volume': [None, 500]} # High < Low, Null
    bad_df = pd.DataFrame(bad_df_data)
    res_df_invalid = validator.validate_dataframe(bad_df, max_null_percentage=0.0) # Stricter null check
    logger.info(f"Bad DF: valid={res_df_invalid.is_valid}, errors={res_df_invalid.errors}, error_details={res_df_invalid.error_details}")

    empty_df = pd.DataFrame()
    res_empty_df = validator.validate_dataframe(empty_df, min_rows=0)
    logger.info(f"Empty DF (min_rows=0): valid={res_empty_df.is_valid}, errors={res_empty_df.errors}")
    res_empty_df_fail = validator.validate_dataframe(empty_df, min_rows=1)
    logger.info(f"Empty DF (min_rows=1): valid={res_empty_df_fail.is_valid}, errors={res_empty_df_fail.errors}")
    
    # Price validation
    logger.info("--- Price Validation ---")
    res_price_valid = validator.validate_price("123.45")
    logger.info(f"Price '123.45': valid={res_price_valid.is_valid}, validated_price={res_price_valid.details.get('validated_price') if res_price_valid.details else 'N/A'}, errors={res_price_valid.errors}, warnings={res_price_valid.details.get('warnings') if res_price_valid.details else 'N/A'}")
    res_price_invalid = validator.validate_price("-5.0")
    logger.info(f"Price '-5.0': valid={res_price_invalid.is_valid}, converted_value={res_price_invalid.details.get('converted_value') if res_price_invalid.details else 'N/A'}, errors={res_price_invalid.errors}")
    res_price_toolong = validator.validate_price("123.456789") # Assuming default precision is less
    logger.info(f"Price '123.456789': valid={res_price_toolong.is_valid}, validated_price={res_price_toolong.details.get('validated_price') if res_price_toolong.details else 'N/A'}, errors={res_price_toolong.errors}, warnings={res_price_toolong.details.get('warnings') if res_price_toolong.details else 'N/A'}")


    # File path validation
    logger.info("--- File Path Validation ---")
    # Create a dummy file for testing 'must_exist'
    dummy_file_path = Path("dummy_test_file.txt")
    try:
        with open(dummy_file_path, "w") as f:
            f.write("test")
        
        res_path_valid = validator.validate_file_path(str(dummy_file_path), must_exist=True, allowed_extensions=[".txt"])
        logger.info(f"Path '{dummy_file_path}': valid={res_path_valid.is_valid}, resolved_path={res_path_valid.details.get('resolved_path') if res_path_valid.details else 'N/A'}, errors={res_path_valid.errors}")
        
        res_path_noexist = validator.validate_file_path("non_existent_file.dat", must_exist=True)
        logger.info(f"Path 'non_existent_file.dat': valid={res_path_noexist.is_valid}, resolved_path_attempt={res_path_noexist.details.get('resolved_path_attempt') if res_path_noexist.details else 'N/A'}, errors={res_path_noexist.errors}")
    finally:
        # Clean up dummy file
        if dummy_file_path.exists():
            os.remove(dummy_file_path)

    logger.info(f"Validator Stats: {validator._validation_stats}")
