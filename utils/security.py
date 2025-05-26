"""
Security utilities for the Stock Trader application.
Handles credential management and secure access to API keys.
"""
import os
from utils.logger import setup_logger
from typing import Dict, Any
from dotenv import load_dotenv

logger = setup_logger(__name__)

# Load environment variables from .env file
load_dotenv()

def get_api_credentials() -> Dict[str, str]:
    """
    Get E*Trade API credentials from environment variables.
    Returns consistent key names for use throughout the application.
    """
    return {
        'consumer_key': os.getenv('ETRADE_CONSUMER_KEY', ''),
        'consumer_secret': os.getenv('ETRADE_CONSUMER_SECRET', ''),
        'oauth_token': os.getenv('ETRADE_OAUTH_TOKEN', ''),
        'oauth_token_secret': os.getenv('ETRADE_OAUTH_TOKEN_SECRET', ''),
        'account_id': os.getenv('ETRADE_ACCOUNT_ID', ''),
        'sandbox': os.getenv('ETRADE_USE_SANDBOX', 'true')
    }

def get_openai_api_key() -> str:
    """
    Retrieve the OpenAI API key from environment variables.
    """
    return os.getenv('OPENAI_API_KEY', '')

def validate_credentials(creds: Dict[str, str]) -> bool:
    """
    Validate that all required credentials are present and non-empty.
    
    Args:
        creds: Dictionary containing API credentials
        
    Returns:
        bool: True if all required credentials are present
    """
    required_fields = ['consumer_key', 'consumer_secret', 'oauth_token', 'oauth_token_secret', 'account_id']
    return all(creds.get(field, '').strip() for field in required_fields)

def get_sandbox_mode(creds: Dict[str, str]) -> bool:
    """
    Determine if sandbox mode should be used based on credentials.
    
    Args:
        creds: Dictionary containing API credentials
        
    Returns:
        bool: True for sandbox mode, False for live trading
    """
    sandbox_value = creds.get('sandbox', creds.get('use_sandbox', 'true'))
    if isinstance(sandbox_value, str):
        return sandbox_value.lower() == 'true'
    return bool(sandbox_value)