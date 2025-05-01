"""
Security utilities for the Stock Trader application.
Handles credential management and secure access to API keys.
"""
import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def get_api_credentials() -> Dict[str, str]:
    """
    Retrieve API credentials from secure storage (env variables or .env file).
    
    Returns:
        Dictionary containing API credentials
    
    Raises:
        ValueError: If required credentials are missing
    """
    # Map of environment variable names to credential keys
    credential_map = {
        'ETRADE_CONSUMER_KEY': 'api_key',
        'ETRADE_CONSUMER_SECRET': 'api_secret', 
        'ETRADE_ACCOUNT_ID': 'account_id',
        'ETRADE_USE_SANDBOX': 'use_sandbox'
    }
    
    # Extract credentials from environment
    credentials = {}
    missing_keys = []
    
    for env_var, cred_key in credential_map.items():
        value = os.getenv(env_var)
        if value is None and env_var != 'ETRADE_USE_SANDBOX':
            missing_keys.append(env_var)
        elif value is not None:
            credentials[cred_key] = value
    
    # Set default for sandbox if not provided
    if 'use_sandbox' not in credentials:
        credentials['use_sandbox'] = 'true'
    
    # Validate required credentials
    if missing_keys:
        error_msg = f"Missing required API credentials: {', '.join(missing_keys)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return credentials