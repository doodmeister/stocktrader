"""
DEPRECATED: This module is deprecated. Use utils.technicals.analysis instead.

technical_analysis.py

This module has been refactored and its functionality moved to:
- utils.technicals.analysis.py (for TechnicalAnalysis class and high-level analysis)
- core.technical_indicators.py (for core calculation functions)

This file is kept for backward compatibility but will be removed in a future version.
"""

import warnings
warnings.warn(
    "utils.technicals.technical_analysis is deprecated. Use utils.technicals.analysis instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export the TechnicalAnalysis class for backward compatibility
from utils.technicals.analysis import TechnicalAnalysis
