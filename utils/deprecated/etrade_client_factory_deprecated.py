from utils.logger import setup_logger
from typing import Optional
from core.etrade_candlestick_bot import ETradeClient

logger = setup_logger(__name__)

def create_etrade_client(creds: dict) -> Optional[ETradeClient]:
    """
    Initialize and return an E*TradeClient using provided credentials.
    Returns None if initialization fails.
    """
    try:
        client = ETradeClient(
            consumer_key=creds.get('consumer_key', ''),
            consumer_secret=creds.get('consumer_secret', ''),
            oauth_token=creds.get('oauth_token', ''),
            oauth_token_secret=creds.get('oauth_token_secret', ''),
            account_id=creds.get('account_id', ''),
            sandbox=creds.get('sandbox', True)
        )
        logger.info("E*Trade client initialized successfully")
        return client
    except Exception as e:
        logger.exception(f"E*Trade client initialization failed: {str(e)}")
        return None