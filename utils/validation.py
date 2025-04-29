"""Validation Utilities

Helper functions to validate and sanitize user input.
"""

import re

def sanitize_input(text: str) -> str:
    """Sanitize stock symbol input (basic alphanumeric)."""
    sanitized = re.sub(r'[^A-Za-z0-9\.\-]', '', text)
    return sanitized.upper()