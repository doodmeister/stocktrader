"""
Security Utils Module for StockTrader Security Package

Provides input validation, sanitization, and general security utilities.
Handles common security operations like path validation and input cleaning.
"""

import re
import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Any
import html
import mimetypes

logger = logging.getLogger(__name__)


def sanitize_input(user_input: str, 
                  max_length: Optional[int] = None,
                  allow_html: bool = False,
                  allowed_chars: Optional[str] = None) -> str:
    """
    Sanitize user input to prevent injection attacks and clean data.
    
    Args:
        user_input: Raw user input to sanitize
        max_length: Maximum allowed length (None for no limit)
        allow_html: Whether to allow HTML tags
        allowed_chars: Optional regex pattern of allowed characters
        
    Returns:
        str: Sanitized input
    """
    if not isinstance(user_input, str):
        return ""
    
    # Trim whitespace
    sanitized = user_input.strip()
    
    # Apply length limit
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Handle HTML
    if not allow_html:
        # Escape HTML characters
        sanitized = html.escape(sanitized)
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`', '$']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
    
    # Apply character whitelist if specified
    if allowed_chars:
        try:
            # Keep only allowed characters
            pattern = f"[^{allowed_chars}]"
            sanitized = re.sub(pattern, '', sanitized)
        except re.error as e:
            logger.warning(f"Invalid regex pattern for allowed_chars: {e}")
    
    return sanitized


def sanitize_user_input(input_text: str, max_length: int = 1000, allow_html: bool = False) -> str:
    """
    Sanitize user input to prevent XSS and other security issues.
    
    Args:
        input_text: The input text to sanitize
        max_length: Maximum allowed length
        allow_html: Whether to allow HTML tags
    
    Returns:
        Sanitized input text
    """
    if not isinstance(input_text, str):
        return str(input_text)
    
    # Truncate if too long
    if len(input_text) > max_length:
        input_text = input_text[:max_length]
    
    if not allow_html:
        # Remove HTML tags
        input_text = re.sub(r'<[^>]+>', '', input_text)
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<script', '</script', 'javascript:', 'data:', 'vbscript:']
    for char in dangerous_chars:
        input_text = input_text.replace(char.lower(), '').replace(char.upper(), '')
    
    return input_text.strip()


def validate_input_length(input_str: str, 
                         min_length: int = 0, 
                         max_length: int = 1000) -> bool:
    """
    Validate input string length.
    
    Args:
        input_str: String to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        bool: True if length is valid, False otherwise
    """
    if not isinstance(input_str, str):
        return False
    
    length = len(input_str)
    return min_length <= length <= max_length


def validate_file_path(file_path: Union[str, Path], 
                      allowed_extensions: Optional[List[str]] = None,
                      base_directory: Optional[Path] = None) -> bool:
    """
    Validate if a file path is safe and allowed.
    
    Args:
        file_path: Path to validate
        allowed_extensions: List of allowed file extensions (e.g., ['.csv', '.json'])
        base_directory: Base directory to restrict access to (prevents path traversal)
    
    Returns:
        True if path is valid and safe
    """
    try:
        path = Path(file_path).resolve()
        
        # Check if path exists and is a file
        if not path.exists() or not path.is_file():
            return False
        
        # Check extension if specified
        if allowed_extensions:
            if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                return False
        
        # Ensure path doesn't escape base directory (prevent path traversal)
        if base_directory:
            try:
                path.relative_to(base_directory.resolve())
            except ValueError:
                logger.warning(f"Path traversal attempt detected: {file_path}")
                return False
        else:
            # Default to project root if no base directory specified
            project_root = Path(__file__).parent.parent.resolve()
            try:
                path.relative_to(project_root)
            except ValueError:
                logger.warning(f"Path outside project root detected: {file_path}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"File path validation error: {e}")
        return False


def prevent_path_traversal(file_path: Union[str, Path]) -> bool:
    """
    Quick first-pass check for path traversal attempts in file paths.

    This is a fast, heuristic filter to catch obvious traversal attempts (e.g., '..', encoded variants),
    but it is NOT a security guarantee. Always use Path(path).resolve().relative_to(base.resolve())
    for actual security enforcement (see validate_file_path).

    Args:
        file_path: Path to check
    Returns:
        bool: True if path is likely safe, False if traversal detected
    """
    path_str = str(file_path)

    # Normalize slashes and decode percent-encoded values for better detection
    import urllib.parse
    normalized = path_str.replace('\\', '/').replace('//', '/').lower()
    decoded = urllib.parse.unquote(normalized)

    # Check for common path traversal patterns (quick filter only)
    dangerous_patterns = [
        '..',      # Parent directory
        '~',       # Home directory
        '///',     # Multiple slashes
        '%2e%2e',  # URL encoded ..
        '%252e',   # Double URL encoded .
        '..%2f',   # Mixed encoding
        '..%5c',   # Mixed encoding (backslash)
    ]
    for pattern in dangerous_patterns:
        if pattern in decoded:
            return False

    # Check for absolute paths when they shouldn't be allowed
    if os.path.isabs(path_str):
        # This might be OK depending on context, but flag for review
        logger.info(f"Absolute path detected: {file_path}")

    return True


def validate_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        bool: True if valid symbol format, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Remove whitespace and convert to uppercase
    symbol = symbol.strip().upper()
    
    # Check length (typically 1-5 characters for most exchanges)
    if not (1 <= len(symbol) <= 10):
        return False
    
    # Check format (letters, numbers, and limited special characters)
    pattern = r'^[A-Z0-9.-]+$'
    return bool(re.match(pattern, symbol))


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if valid email format, False otherwise
    """
    if not email or not isinstance(email, str):
        return False
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def validate_numeric_range(value: Union[int, float], 
                          min_value: Optional[Union[int, float]] = None,
                          max_value: Optional[Union[int, float]] = None) -> bool:
    """
    Validate that a numeric value is within specified range.
    
    Args:
        value: Numeric value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        
    Returns:
        bool: True if value is in range, False otherwise
    """
    try:
        num_value = float(value)
        
        if min_value is not None and num_value < min_value:
            return False
        
        if max_value is not None and num_value > max_value:
            return False
        
        return True
        
    except (ValueError, TypeError):
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal and invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    if not filename:
        return "unnamed_file"
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        max_name_length = 255 - len(ext)
        filename = name[:max_name_length] + ext
    
    # Ensure filename is not empty
    if not filename.strip():
        filename = "unnamed_file"
    
    return filename


def generate_secure_filename(original_name: str, max_length: int = 255) -> str:
    """
    Generate a secure filename by removing dangerous characters.
    
    Args:
        original_name: Original filename
        max_length: Maximum allowed filename length
    
    Returns:
        Sanitized filename
    """
    # Remove path separators and dangerous characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', original_name)
    
    # Remove control characters
    safe_name = re.sub(r'[\x00-\x1f\x7f]', '', safe_name)
    
    # Limit length
    if len(safe_name) > max_length:
        name, ext = Path(safe_name).stem, Path(safe_name).suffix
        safe_name = name[:max_length - len(ext)] + ext
    
    return safe_name.strip()


def validate_json_keys(data: Any, required_keys: Optional[List[str]] = None) -> bool:
    """
    Validate JSON data structure.
    
    Args:
        data: JSON data to validate
        required_keys: Optional list of required keys
        
    Returns:
        bool: True if structure is valid, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    if required_keys:
        for key in required_keys:
            if key not in data:
                logger.warning(f"Missing required key in JSON: {key}")
                return False
    
    return True


def validate_json_structure(data: Any, required_fields: Optional[List[str]] = None) -> bool:
    """
    Validate JSON data structure for security and completeness.
    
    Args:
        data: JSON data to validate
        required_fields: List of required field names
    
    Returns:
        True if structure is valid
    """
    try:
        if not isinstance(data, dict):
            return False
        
        if required_fields:
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                logger.warning(f"Missing required fields: {missing_fields}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"JSON structure validation error: {e}")
        return False


def escape_sql_like(query: str) -> str:
    """
    Escape special characters for SQL LIKE queries.
    
    Args:
        query: Query string to escape
        
    Returns:
        str: Escaped query string
    """
    # Escape SQL LIKE special characters
    escaped = query.replace('\\', '\\\\')  # Escape backslashes first
    escaped = escaped.replace('%', '\\%')   # Escape wildcards
    escaped = escaped.replace('_', '\\_')   # Escape single character wildcards
    
    return escaped


def rate_limit_key(identifier: str, action: str = "default") -> str:
    """
    Generate a rate limiting key for an identifier and action.
    
    Args:
        identifier: Unique identifier (e.g., IP, user ID)
        action: Action being rate limited
        
    Returns:
        str: Rate limiting key
    """
    from .encryption import generate_session_hash
    return generate_session_hash(f"{identifier}:{action}")


def is_safe_redirect_url(url: str, allowed_domains: Optional[List[str]] = None) -> bool:
    """
    Check if a redirect URL is safe to prevent open redirect attacks.
    
    Args:
        url: URL to validate
        allowed_domains: Optional list of allowed domains
        
    Returns:
        bool: True if URL is safe for redirect, False otherwise
    """
    if not url:
        return False
    
    # Check for protocol
    if url.startswith(('javascript:', 'data:', 'vbscript:')):
        return False
    
    # Allow relative URLs
    if url.startswith('/'):
        return True
    
    # For absolute URLs, check domain if specified
    if allowed_domains:
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            return domain in [d.lower() for d in allowed_domains]
        except Exception:
            return False
    
    return True


def validate_file_size(file_path: Union[str, Path], max_size_mb: int = 100) -> bool:
    """
    Validate file size limits to prevent resource exhaustion.
    
    Args:
        file_path: Path to the file to check
        max_size_mb: Maximum allowed file size in megabytes
    
    Returns:
        True if file size is within limits
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return True  # File doesn't exist yet
        
        size_mb = path.stat().st_size / (1024 * 1024)
        return size_mb <= max_size_mb
    except Exception as e:
        logger.error(f"File size validation error: {e}")
        return False


def validate_mime_type(file_path: Union[str, Path], allowed_types: Optional[List[str]] = None) -> bool:
    """
    Validate file MIME type for additional security.
    
    Args:
        file_path: Path to the file to check
        allowed_types: List of allowed MIME types (e.g., ['text/csv', 'application/json'])
    
    Returns:
        True if MIME type is allowed
    """
    if not allowed_types:
        return True
    
    try:
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type in allowed_types if mime_type else False
    except Exception as e:
        logger.error(f"MIME type validation error: {e}")
        return False
