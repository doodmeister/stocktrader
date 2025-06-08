import re

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
