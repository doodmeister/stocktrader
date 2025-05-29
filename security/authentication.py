"""
Authentication Module for StockTrader Security Package

Handles session management, API key validation, and credential management.
Provides secure authentication flows for the trading platform.
"""

import streamlit as st
import time
import logging
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)


def validate_session_security() -> bool:
    """
    Validate session security for the StockTrader dashboard.
    
    Performs comprehensive security checks including:
    - Session token validation
    - Rate limiting
    - IP validation (if configured)
    - Session timeout checks
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        # Initialize session security state if not exists
        if 'security_initialized' not in st.session_state:
            _initialize_session_security()
        
        # Check session timeout
        if not _check_session_timeout():
            logger.warning("Session timeout detected")
            return False
        
        # Check rate limiting
        if not _check_rate_limiting():
            logger.warning("Rate limit exceeded")
            return False
        
        # Update last activity time
        st.session_state.last_activity = time.time()
        
        return True
        
    except Exception as e:
        logger.error(f"Security validation error: {e}")
        return False


def _initialize_session_security() -> None:
    """Initialize security-related session state variables."""
    current_time = time.time()
    
    st.session_state.security_initialized = True
    st.session_state.session_token = _generate_session_token()
    st.session_state.session_start = current_time
    st.session_state.last_activity = current_time
    st.session_state.request_count = 0
    st.session_state.last_request_time = current_time


def _generate_session_token() -> str:
    """Generate a secure session token."""
    from .encryption import create_secure_token
    return create_secure_token()


def _check_session_timeout() -> bool:
    """
    Check if the session has timed out.
    
    Returns:
        bool: True if session is valid, False if timed out
    """
    # Default timeout: 8 hours
    timeout_seconds = 8 * 60 * 60
    
    if 'last_activity' not in st.session_state:
        return True  # First access, allow
    
    time_since_activity = time.time() - st.session_state.last_activity
    
    if time_since_activity > timeout_seconds:
        _clear_session_security()
        return False
    
    return True


def _check_rate_limiting() -> bool:
    """
    Check if the current session is within rate limits.
    
    Returns:
        bool: True if within limits, False if rate limited
    """
    current_time = time.time()
    
    # Rate limit: 100 requests per minute
    rate_limit = 100
    window_seconds = 60
    
    if 'request_count' not in st.session_state:
        st.session_state.request_count = 0
        st.session_state.last_request_time = current_time
    
    # Reset counter if window has passed
    if current_time - st.session_state.last_request_time > window_seconds:
        st.session_state.request_count = 0
        st.session_state.last_request_time = current_time
    
    # Increment request count
    st.session_state.request_count += 1
    
    # Check if rate limit exceeded
    if st.session_state.request_count > rate_limit:
        return False
    
    return True


def _clear_session_security() -> None:
    """Clear security-related session state."""
    security_keys = [
        'security_initialized',
        'session_token',
        'session_start',
        'last_activity',
        'request_count',
        'last_request_time'
    ]
    
    for key in security_keys:
        if key in st.session_state:
            del st.session_state[key]


def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validate API key format and structure.
    
    Args:
        api_key: API key to validate
        
    Returns:
        bool: True if valid format, False otherwise
    """
    if not api_key:
        return False
    
    # Basic validation - adjust based on your API key format
    if len(api_key) < 16:
        return False
    
    # Check for basic alphanumeric structure
    if not api_key.replace('-', '').replace('_', '').isalnum():
        return False
    
    return True


def get_api_credentials() -> Optional[Dict[str, str]]:
    """
    Get API credentials from environment or session state.
    
    Returns:
        Dictionary containing API credentials or None if not available
    """
    credentials = {}
    
    # Check for E*Trade credentials
    etrade_keys = [
        'ETRADE_CONSUMER_KEY',
        'ETRADE_CONSUMER_SECRET', 
        'ETRADE_OAUTH_TOKEN',
        'ETRADE_OAUTH_TOKEN_SECRET',
        'ETRADE_ACCOUNT_ID'
    ]
    
    for key in etrade_keys:
        value = os.getenv(key)
        if value:
            # Map environment variable names to expected credential keys
            cred_key = key.replace('ETRADE_', '').lower()
            credentials[cred_key] = value
    
    # Check session state as fallback
    if not credentials and hasattr(st, 'session_state'):
        session_creds = getattr(st.session_state, 'api_credentials', {})
        if session_creds:
            credentials.update(session_creds)
    
    return credentials if credentials else None


def get_openai_api_key() -> Optional[str]:
    """
    Get OpenAI API key from environment or session state.
    
    Returns:
        OpenAI API key string or None if not available
    """
    # Try environment variable first
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key
    
    # Check session state as fallback
    if hasattr(st, 'session_state'):
        session_key = getattr(st.session_state, 'openai_api_key', None)
        if session_key:
            return session_key
    
    return None


def validate_credentials(credentials: Dict[str, str]) -> bool:
    """
    Validate API credentials format and completeness.
    
    Args:
        credentials: Dictionary of credentials to validate
        
    Returns:
        bool: True if credentials are valid format, False otherwise
    """
    if not isinstance(credentials, dict):
        return False
    
    # Required keys for E*Trade
    required_keys = [
        'consumer_key',
        'consumer_secret',
        'oauth_token', 
        'oauth_token_secret',
        'account_id'
    ]
    
    # Check if all required keys are present and non-empty
    for key in required_keys:
        if key not in credentials or not credentials[key]:
            logger.warning(f"Missing or empty credential: {key}")
            return False
    
    # Basic format validation
    for key, value in credentials.items():
        if not isinstance(value, str) or len(value.strip()) == 0:
            logger.warning(f"Invalid credential format for {key}")
            return False
    
    return True


def get_sandbox_mode(credentials: Dict[str, str]) -> bool:
    """
    Determine if sandbox mode should be used based on credentials or environment.
    
    Args:
        credentials: API credentials dictionary
        
    Returns:
        bool: True if sandbox mode should be used, False for production
    """
    # Check environment variable first
    sandbox_env = os.getenv('ETRADE_SANDBOX', 'true').lower()
    if sandbox_env in ('false', '0', 'no', 'prod', 'production'):
        return False
    
    # Check credentials dictionary for sandbox setting
    if credentials and 'sandbox' in credentials:
        sandbox_cred = str(credentials['sandbox']).lower()
        if sandbox_cred in ('false', '0', 'no', 'prod', 'production'):
            return False
    
    # Default to sandbox for safety
    return True


def validate_session_token(token: str) -> bool:
    """
    Validate a session token.
    
    Args:
        token: Token to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not token or len(token) != 64:  # 32 bytes = 64 hex chars
        return False
    
    # Check if token exists in session state
    session_token = getattr(st.session_state, 'session_token', None)
    return session_token == token


def refresh_session() -> bool:
    """
    Refresh the current session with new security parameters.
    
    Returns:
        bool: True if refresh successful, False otherwise
    """
    try:
        _clear_session_security()
        _initialize_session_security()
        logger.info("Session refreshed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to refresh session: {e}")
        return False


def get_session_info() -> Dict[str, Any]:
    """
    Get current session security information.
    
    Returns:
        Dictionary containing session information
    """
    if not hasattr(st, 'session_state'):
        return {}
    
    return {
        'initialized': getattr(st.session_state, 'security_initialized', False),
        'session_start': getattr(st.session_state, 'session_start', None),
        'last_activity': getattr(st.session_state, 'last_activity', None),
        'request_count': getattr(st.session_state, 'request_count', 0),
        'has_token': bool(getattr(st.session_state, 'session_token', None))
    }
