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
    """
    load_dotenv()  # Make sure to import this if not already
    
    return {
        'consumer_key': os.getenv('ETRADE_CONSUMER_KEY', ''),
        'consumer_secret': os.getenv('ETRADE_CONSUMER_SECRET', ''),
        'oauth_token': os.getenv('ETRADE_OAUTH_TOKEN', ''),
        'oauth_token_secret': os.getenv('ETRADE_OAUTH_TOKEN_SECRET', ''),
        'account_id': os.getenv('ETRADE_ACCOUNT_ID', ''),
        'use_sandbox': os.getenv('ETRADE_USE_SANDBOX', 'true')
    }