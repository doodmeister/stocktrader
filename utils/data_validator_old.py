"""
Data validation utilities for stock data and user inputs.

Provides validation for symbol formats, date ranges, and data integrity.
"""
import re
from datetime import date
from typing import List, Optional


class DataValidator:
    """
    Validates user inputs and data integrity for the stock trading application.
    """
    
    def __init__(self):
        # Regex for basic symbol validation (allowing dots and hyphens)
        self._symbol_pattern = re.compile(r'^[A-Z0-9.-]{1,10}$')
    
    def sanitize_symbol(self, text: str) -> str:
        """
        Sanitize stock symbol input (basic alphanumeric with . and -).
        
        Args:
            text: Raw text input
            
        Returns:
            Sanitized uppercase text
        """
        sanitized = re.sub(r'[^A-Za-z0-9.\-]', '', text)
        return sanitized.upper()
    
    def validate_symbol(self, symbol: str) -> str:
        """
        Validate a single stock ticker symbol.
        
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
        
        if not self._symbol_pattern.match(symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        
        return symbol
    
    def validate_symbols(self, symbols_input: str) -> List[str]:
        """
        Validate and parse comma-separated stock symbols.
        
        Args:
            symbols_input: Comma-separated list of stock ticker symbols
            
        Returns:
            List of valid symbols
            
        Raises:
            ValueError: If input contains invalid symbols
        """
        if not symbols_input or not symbols_input.strip():
            return []
            
        # Split on commas, clean up whitespace, and sanitize each symbol
        symbols = [self.sanitize_symbol(s.strip()) for s in symbols_input.split(',')]
        
        # Filter out empty strings
        symbols = [s for s in symbols if s]
        
        # Validate each symbol
        invalid_symbols = [s for s in symbols if not self._symbol_pattern.match(s)]
        if invalid_symbols:
            raise ValueError(
                f"Invalid symbols: {', '.join(invalid_symbols)}. "
                "Symbols must be 1-10 alphanumeric characters with optional . or -"
            )
            
        return symbols
    
    def validate_dates(self, start_date: date, end_date: date) -> bool:
        """
        Validate a date range for data fetching.
        
        Args:
            start_date: The starting date
            end_date: The ending date
            
        Returns:
            True if dates are valid, False otherwise
        """
        # Ensure dates are properly ordered
        if start_date > end_date:
            return False
            
        # Prevent fetching future data
        today = date.today()
        if end_date > today:
            return False
            
        return True
    
    def validate_dataframe(self, df, required_cols: Optional[List[str]] = None) -> bool:
        """
        Validate a DataFrame has required columns and sufficient data.
        
        Args:
            df: Pandas DataFrame to validate
            required_cols: List of column names that must be present
            
        Returns:
            True if valid, False otherwise
        """
        if df is None or df.empty:
            return False
            
        if required_cols:
            # Check that all required columns exist
            if not all(col in df.columns for col in required_cols):
                return False
                
        return True