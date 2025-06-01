"""
Reusable exception classes for StockTrader core modules.
"""

class StockTraderException(Exception):
    """Base exception for all StockTrader related errors."""
    pass

class ValidationError(StockTraderException):
    """Base validation exception."""
    pass

class SymbolValidationError(ValidationError):
    """Symbol-specific validation error."""
    pass

class DateValidationError(ValidationError):
    """Date-specific validation error."""
    pass

class DataIntegrityError(ValidationError):
    """Data integrity validation error."""
    pass

class SecurityValidationError(ValidationError):
    """Security-related validation error."""
    pass

class PerformanceValidationError(ValidationError):
    """Performance-related validation error."""
    pass
