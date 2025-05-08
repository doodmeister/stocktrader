"""
Stock validation utility for verifying ticker symbols and providing common stock options.
This module intentionally avoids importing torch or other ML libraries to prevent circular imports.
"""
from utils.logger import setup_logger
import time
from typing import Dict, List, Optional, Set, Tuple
from functools import lru_cache

# Configure module logger
logger = setup_logger(__name__)

# Cache for validated tickers with timestamp to enable expiry
_TICKER_CACHE: Dict[str, Tuple[bool, float]] = {}
# Cache expiration time in seconds (4 hours)
CACHE_EXPIRY = 4 * 60 * 60
# API call timeout in seconds
API_TIMEOUT = 5
# Maximum number of API calls per batch
MAX_VALIDATION_BATCH = 15


@lru_cache(maxsize=1)
def get_popular_tickers() -> List[str]:
    """
    Returns a curated list of popular stock tickers across different sectors.
    
    Returns:
        List[str]: Common stock ticker symbols that are likely to be valid
    """
    return [
        # Technology
        "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "INTC", "AMD", 
        # Finance
        "JPM", "V", "MA", "BAC", "WFC", "C", 
        # Consumer
        "WMT", "DIS", "NFLX", "KO", "PEP", "MCD", "NKE", "SBUX", 
        # Healthcare
        "JNJ", "PFE", "UNH", "ABBV", "MRK",
        # Energy
        "XOM", "CVX", "COP"
    ]


def _is_ticker_cached_valid(ticker: str) -> Optional[bool]:
    """
    Check if a ticker is in cache and still valid.
    
    Args:
        ticker: Stock ticker symbol to check
        
    Returns:
        bool or None: True if valid, False if invalid, None if not in cache or expired
    """
    if ticker not in _TICKER_CACHE:
        return None
        
    valid, timestamp = _TICKER_CACHE[ticker]
    if time.time() - timestamp > CACHE_EXPIRY:
        # Cache entry expired
        return None
        
    return valid


def validate_ticker(ticker: str) -> bool:
    """
    Validate if a ticker symbol exists by querying an external API.
    
    Args:
        ticker: Stock ticker symbol to validate
        
    Returns:
        bool: True if ticker is valid, False otherwise
    """
    # Check cache first
    cached_result = _is_ticker_cached_valid(ticker)
    if cached_result is not None:
        return cached_result
    
    try:
        # Import yfinance here to avoid circular imports
        import yfinance as yf
        import requests.exceptions
        
        # Try to get basic info with timeout
        stock = yf.Ticker(ticker)
        
        # Use a context manager to handle timeouts properly
        try:
            # Access info property with implicit API call
            info = stock.info
            is_valid = bool(info and 'symbol' in info)
        except (requests.exceptions.RequestException, 
                ValueError, 
                KeyError,
                TypeError) as e:
            logger.debug(f"API error validating {ticker}: {str(e)}")
            is_valid = False
            
        # Cache the result with current timestamp
        _TICKER_CACHE[ticker] = (is_valid, time.time())
        return is_valid
        
    except ImportError:
        logger.warning("yfinance not installed, assuming ticker is valid")
        return True
    except Exception as e:
        logger.warning(f"Unexpected error validating {ticker}: {str(e)}")
        return False


def get_valid_tickers(user_tickers: Optional[List[str]] = None) -> List[str]:
    """
    Get a list of valid stock ticker symbols.
    
    Combines popular tickers with user-provided ones and validates them against
    an external API, with caching to minimize API calls.
    
    Args:
        user_tickers: Optional list of user-provided ticker symbols
        
    Returns:
        List[str]: List of validated ticker symbols, sorted alphabetically
    """
    popular = get_popular_tickers()
    
    # Combine and deduplicate tickers
    all_tickers: Set[str] = set(popular)
    if user_tickers:
        # Strip whitespace and convert to uppercase for consistency
        cleaned_tickers = [t.strip().upper() for t in user_tickers if t and t.strip()]
        all_tickers.update(cleaned_tickers)
    
    # First return any cached valid tickers
    valid_tickers = [t for t in all_tickers if _is_ticker_cached_valid(t) is True]
    
    # For remaining tickers, validate up to MAX_VALIDATION_BATCH
    remaining = [t for t in all_tickers if _is_ticker_cached_valid(t) is None]
    
    # Limit API calls to avoid rate limiting
    for ticker in remaining[:MAX_VALIDATION_BATCH]:
        if validate_ticker(ticker):
            valid_tickers.append(ticker)
    
    # If no valid tickers found, provide fallback
    if not valid_tickers:
        logger.warning("No valid tickers found, using AAPL as fallback")
        return ["AAPL"]
        
    return sorted(valid_tickers)


def clear_ticker_cache() -> None:
    """Clear the ticker validation cache."""
    _TICKER_CACHE.clear()