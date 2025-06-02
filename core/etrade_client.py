"""
E*Trade API Client - Following Official Examples

This module implements E*Trade API integration following the official examples
from the E*Trade Python SDK. It uses the correct OAuth flow, URL patterns,
and response handling as shown in the source of truth examples.

Based on: examples/etrade/etrade_python_client.py
"""

import json
import logging
import webbrowser
import configparser
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
from rauth import OAuth1Service
import requests

# Configure logging to match E*Trade examples
logger = logging.getLogger(__name__)


class ETradeAuthenticationError(Exception):
    """Exception for E*Trade authentication issues"""
    pass


class ETradeAPIError(Exception):
    """Exception for E*Trade API errors"""
    pass


class ETradeClient:
    """
    E*Trade API Client following official examples.
    
    This implementation follows the patterns from examples/etrade/ files:
    - Uses rauth.OAuth1Service for OAuth flow
    - Follows exact URL patterns from examples
    - Uses response structures matching official examples
    - Implements simple error handling as shown in examples
    """
    
    def __init__(self, consumer_key: str, consumer_secret: str, sandbox: bool = True):
        """
        Initialize E*Trade client with consumer credentials.
        
        Args:
            consumer_key: E*Trade consumer key
            consumer_secret: E*Trade consumer secret
            sandbox: Use sandbox environment (default True)
        """
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.sandbox = sandbox
        
        # Set base URLs following examples/etrade/sample_config.ini
        if sandbox:
            self.base_url = "https://apisb.etrade.com"
        else:
            self.base_url = "https://api.etrade.com"
        
        # Initialize OAuth service following etrade_python_client.py
        self.etrade = OAuth1Service(
            name="etrade",
            consumer_key=self.consumer_key,
            consumer_secret=self.consumer_secret,
            request_token_url="https://api.etrade.com/oauth/request_token",
            access_token_url="https://api.etrade.com/oauth/access_token",
            authorize_url="https://us.etrade.com/e/t/etws/authorize?key={}&token={}",
            base_url="https://api.etrade.com"
        )
        
        self.session = None
        self.accounts = []
        
        logger.info(f"ETradeClient initialized ({'SANDBOX' if sandbox else 'LIVE'} mode)")
    
    def authenticate(self, verification_code: Optional[str] = None) -> bool:
        """
        Perform OAuth authentication following etrade_python_client.py flow.
        
        Args:
            verification_code: Manual verification code if not using browser
            
        Returns:
            bool: True if authentication successful
        """
        try:
            # Step 1: Get OAuth 1 request token and secret
            logger.info("Getting request token...")
            request_token, request_token_secret = self.etrade.get_request_token(
                params={"oauth_callback": "oob", "format": "json"}
            )
            
            # Step 2: Get authorization from user
            authorize_url = self.etrade.authorize_url.format(
                self.etrade.consumer_key, request_token
            )
            
            if verification_code is None:
                # Open browser for authorization (following examples)
                logger.info(f"Opening browser for authorization: {authorize_url}")
                webbrowser.open(authorize_url)
                
                # This would need to be handled in Streamlit UI
                raise ETradeAuthenticationError(
                    f"Please authorize the application at: {authorize_url}\n"
                    "Then call authenticate() again with the verification code."
                )
            
            # Step 3: Exchange authorized request token for authenticated session
            logger.info("Exchanging tokens for authenticated session...")
            self.session = self.etrade.get_auth_session(
                request_token,
                request_token_secret,
                params={"oauth_verifier": verification_code}
            )
            
            # Verify authentication by getting account list
            self._load_accounts()
            
            logger.info("Authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise ETradeAuthenticationError(f"Authentication failed: {e}")
    
    def _load_accounts(self) -> None:
        """Load account list following accounts.py example."""
        if not self.session:
            raise ETradeAuthenticationError("Not authenticated")
        
        # URL pattern from accounts.py
        url = f"{self.base_url}/v1/accounts/list.json"
        
        try:
            # Make request following accounts.py pattern
            response = self.session.get(url)
            logger.debug(f"Request Header: {response.request.headers}")
            
            # Handle response following accounts.py pattern
            if response is not None and response.status_code == 200:
                parsed = json.loads(response.text)
                logger.debug(f"Response Body: {json.dumps(parsed, indent=4, sort_keys=True)}")
                
                data = response.json()
                if (data is not None and 
                    "AccountListResponse" in data and 
                    "Accounts" in data["AccountListResponse"] and
                    "Account" in data["AccountListResponse"]["Accounts"]):
                    
                    accounts = data["AccountListResponse"]["Accounts"]["Account"]
                    # Filter closed accounts following accounts.py
                    self.accounts = [acc for acc in accounts if acc.get('accountStatus') != 'CLOSED']
                    
                    logger.info(f"Loaded {len(self.accounts)} active accounts")
                else:
                    raise ETradeAPIError("Invalid account list response structure")
            else:
                # Error handling following accounts.py pattern
                error_msg = "Unknown error"
                if (response is not None and 
                    response.headers.get('Content-Type') == 'application/json'):
                    try:
                        error_data = response.json()
                        if "Error" in error_data and "message" in error_data["Error"]:
                            error_msg = error_data["Error"]["message"]
                    except:
                        pass
                
                raise ETradeAPIError(f"Account list error: {error_msg}")
                
        except ETradeAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to load accounts: {e}")
            raise ETradeAPIError(f"Failed to load accounts: {e}")
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get quote for a symbol following market.py example.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Quote data dictionary
        """
        if not self.session:
            raise ETradeAuthenticationError("Not authenticated")
        
        symbol = symbol.upper().strip()
        
        # URL pattern from market.py
        url = f"{self.base_url}/v1/market/quote/{symbol}.json"
        
        try:
            # Make request following market.py pattern
            response = self.session.get(url)
            logger.debug(f"Request Header: {response.request.headers}")
            
            if response is not None and response.status_code == 200:
                parsed = json.loads(response.text)
                logger.debug(f"Response Body: {json.dumps(parsed, indent=4, sort_keys=True)}")
                
                # Handle response following market.py pattern
                data = response.json()
                if (data is not None and 
                    "QuoteResponse" in data and 
                    "QuoteData" in data["QuoteResponse"]):
                    
                    quote_data = data["QuoteResponse"]["QuoteData"]
                    if quote_data and len(quote_data) > 0:
                        return quote_data[0]  # Return first quote
                    else:
                        raise ETradeAPIError("No quote data returned")
                else:
                    # Error handling following market.py pattern
                    if (data is not None and 
                        'QuoteResponse' in data and 
                        'Messages' in data["QuoteResponse"] and
                        'Message' in data["QuoteResponse"]["Messages"]):
                        
                        messages = data["QuoteResponse"]["Messages"]["Message"]
                        error_msgs = [msg["description"] for msg in messages]
                        raise ETradeAPIError(f"Quote error: {'; '.join(error_msgs)}")
                    else:
                        raise ETradeAPIError("Quote API service error")
            else:
                raise ETradeAPIError(f"Quote request failed: {response.status_code}")
                
        except ETradeAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            raise ETradeAPIError(f"Failed to get quote: {e}")
    
    def get_candles(self, symbol: str, interval: str = "5min", days: int = 1) -> pd.DataFrame:
        """
        Get candlestick data. This would need to be implemented based on 
        actual E*Trade historical data API endpoints.
        
        Note: The examples don't show historical data endpoints, so this
        would need to be implemented based on E*Trade API documentation.
        """
        # Placeholder implementation - would need actual E*Trade endpoint
        raise NotImplementedError(
            "Candlestick data endpoint not available in E*Trade examples. "
            "This would need to be implemented based on E*Trade API documentation."
        )
    
    def get_account_balance(self, account_id: str) -> Dict[str, Any]:
        """
        Get account balance. This follows the pattern from examples but
        would need the actual balance endpoint from E*Trade API docs.
        """
        if not self.session:
            raise ETradeAuthenticationError("Not authenticated")
        
        # This URL pattern would need to be confirmed with E*Trade API docs
        url = f"{self.base_url}/v1/accounts/{account_id}/balance.json"
        
        try:
            response = self.session.get(url)
            
            if response is not None and response.status_code == 200:
                return response.json()
            else:
                raise ETradeAPIError(f"Balance request failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            raise ETradeAPIError(f"Failed to get account balance: {e}")
    
    def place_order(self, symbol: str, side: str, quantity: int, 
                   order_type: str = "MARKET", limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place an order following order.py example patterns.
        
        This is a simplified version - the full implementation would need
        to follow the complete XML payload structure from order.py.
        """
        if not self.session:
            raise ETradeAuthenticationError("Not authenticated")
        
        if not self.accounts:
            raise ETradeAPIError("No accounts available")
        
        # Use first account for now
        account = self.accounts[0]
        account_id_key = account.get("accountIdKey")
        
        if not account_id_key:
            raise ETradeAPIError("Invalid account ID key")
        
        # URL pattern from order.py
        url = f"{self.base_url}/v1/accounts/{account_id_key}/orders/preview.json"
        
        # Headers from order.py
        headers = {
            "Content-Type": "application/xml",
            "consumerKey": self.consumer_key
        }
        
        # XML payload following order.py pattern (simplified)
        client_order_id = f"ORDER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        payload = f"""<PreviewOrderRequest>
                       <orderType>EQ</orderType>
                       <clientOrderId>{client_order_id}</clientOrderId>
                       <Order>
                           <allOrNone>false</allOrNone>
                           <priceType>{order_type}</priceType>
                           <orderTerm>GOOD_FOR_DAY</orderTerm>
                           <marketSession>REGULAR</marketSession>
                           <stopPrice></stopPrice>
                           <limitPrice>{limit_price or ''}</limitPrice>
                           <Instrument>
                               <Product>
                                   <securityType>EQ</securityType>
                                   <symbol>{symbol}</symbol>
                               </Product>
                               <orderAction>{side}</orderAction>
                               <quantityType>QUANTITY</quantityType>
                               <quantity>{quantity}</quantity>
                           </Instrument>
                       </Order>
                   </PreviewOrderRequest>"""
        
        try:
            # Make request following order.py pattern
            response = self.session.post(url, headers=headers, data=payload)
            logger.debug(f"Request Header: {response.request.headers}")
            logger.debug(f"Request payload: {payload}")
            
            if response is not None and response.status_code == 200:
                parsed = json.loads(response.text)
                logger.debug(f"Response Body: {json.dumps(parsed, indent=4, sort_keys=True)}")
                return parsed
            else:
                # Error handling following order.py pattern
                try:
                    error_data = response.json()
                    if 'Error' in error_data and 'message' in error_data["Error"]:
                        error_msg = error_data["Error"]["message"]
                    else:
                        error_msg = "Order API service error"
                except:
                    error_msg = "Order API service error"
                
                raise ETradeAPIError(f"Order failed: {error_msg}")
                
        except ETradeAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise ETradeAPIError(f"Failed to place order: {e}")


def create_etrade_client_from_config() -> ETradeClient:
    """
    Create E*Trade client from configuration following the examples pattern.
    
    This function looks for credentials in environment variables following
    the pattern from sample_config.ini.
    """
    import os
    
    consumer_key = os.getenv('ETRADE_CONSUMER_KEY')
    consumer_secret = os.getenv('ETRADE_CONSUMER_SECRET')
    sandbox = os.getenv('ETRADE_SANDBOX', 'true').lower() == 'true'
    
    if not consumer_key or not consumer_secret:
        raise ETradeAuthenticationError(
            "Missing E*Trade credentials. Please set ETRADE_CONSUMER_KEY and ETRADE_CONSUMER_SECRET environment variables."
        )
    
    return ETradeClient(consumer_key, consumer_secret, sandbox)
