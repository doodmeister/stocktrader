"""
Validation utilities for the Stock Trader application.
Provides functions for validating user inputs and API responses.
"""

import re
import time
import logging
from typing import Callable, Any, Optional, TypeVar, List, Dict
from functools import wraps
from train.deeplearning_trainer import TrainingConfig
from train.config import TrainingConfig

logger = logging.getLogger(__name__)

# Common stock market symbols regex (basic validation)
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9.]{1,10}$')

T = TypeVar('T')  # Generic type for return values

def sanitize_input(text: str) -> str:
    """Sanitize stock symbol input (basic alphanumeric)."""
    sanitized = re.sub(r'[^A-Za-z0-9\.\-]', '', text)
    return sanitized.upper()

def validate_symbol(symbol: str) -> str:
    """
    Validate a stock ticker symbol.
    
    Args:
        symbol: The symbol to validate
        
    Returns:
        The validated symbol (uppercase)
        
    Raises:
        ValueError: If the symbol is invalid
    """
    if not symbol:
        raise ValueError("Symbol cannot be empty")
        
    symbol = symbol.strip().upper()
    
    if not SYMBOL_PATTERN.match(symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")
    
    return symbol

def validate_training_params(cfg: TrainingConfig) -> None:
    if cfg.epochs <= 0: raise ValueError("epochs must be > 0")
    if not (0 < cfg.validation_split < 1): raise ValueError("validation_split must be in (0,1)")
    # …any other checks…

def safe_request(
    func: Callable[[], T],
    max_retries: int = 3,
    retry_delay: int = 1,
    timeout: Optional[int] = None,
    error_handler: Optional[Callable[[Exception], T]] = None
) -> Optional[T]:
    """
    Execute a function with retry logic and error handling.
    
    Args:
        func: The function to execute
        max_retries: Maximum number of retry attempts
        retry_delay: Seconds to wait between retries
        timeout: Maximum seconds to wait before giving up
        error_handler: Optional function to handle exceptions
        
    Returns:
        The result of the function call, or None if all attempts fail
    """
    start_time = time.time()
    attempts = 0
    last_error = None
    
    while attempts < max_retries:
        try:
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                logger.warning(f"Request timed out after {timeout} seconds")
                break
                
            # Execute the function
            result = func()
            return result
            
        except Exception as e:
            last_error = e
            attempts += 1
            
            if attempts < max_retries:
                wait_time = retry_delay * (2 ** (attempts - 1))  # Exponential backoff
                logger.warning(f"Request failed (attempt {attempts}/{max_retries}): {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Request failed after {max_retries} attempts: {str(e)}")
    
    # All attempts failed
    if error_handler and last_error:
        return error_handler(last_error)
    
    return None
