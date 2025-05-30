"""
Patterns package for candlestick pattern detection.

This package provides comprehensive candlestick pattern detection
functionality for the StockTrader application.
"""

from .patterns import CandlestickPatterns, PatternType, PatternStrength, PatternResult

__all__ = ['CandlestickPatterns', 'PatternType', 'PatternStrength', 'PatternResult']
