"""
ETrade Candlestick Bot

Production-grade trading engine that connects to E*Trade's API for automated candlestick pattern trading.
Features comprehensive risk management, ML pattern detection, and enterprise-grade error handling.

Key Components:
- ETradeClient: Robust API client with authentication, rate limiting, and error recovery
- StrategyEngine: Core trading logic with pattern detection and position management
- RiskManager: Position sizing, portfolio limits, and stop-loss management
- PerformanceTracker: Trade analytics and performance metrics

Author: Trading Bot Team
License: MIT
"""

import asyncio
import os
import sys
import time
import signal
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import datetime as dt

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests_oauthlib import OAuth1Session
from urllib3.util.retry import Retry

# Project imports
from utils.logger import setup_logger
from utils.security import get_api_credentials
from patterns.patterns_nn import PatternNN
from utils.notifier import Notifier
from utils.technicals.indicators import TechnicalIndicators
from core.risk_manager_v2 import RiskManager


# Configure structured logging
logger = setup_logger(__name__)


class OrderStatus(Enum):
    """Order execution status enumeration"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"


class MarketState(Enum):
    """Market session state enumeration"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PRE_MARKET = "PRE_MARKET"
    AFTER_HOURS = "AFTER_HOURS"


@dataclass(frozen=True)
class TradeConfig:
    """
    Immutable configuration for trading parameters with validation.
    
    All percentage values are in decimal format (e.g., 0.02 = 2%).
    """
    # Position management
    max_positions: int = field(default=5)
    max_loss_percent: float = field(default=0.02)
    profit_target_percent: float = field(default=0.03)
    max_daily_loss: float = field(default=0.05)
    
    # Execution parameters
    polling_interval: int = field(default=300)  # seconds
    risk_per_trade_pct: float = field(default=0.01)
    max_position_size_pct: float = field(default=0.10)
    trailing_stop_activation_pct: float = field(default=0.01)
    
    # Strategy controls
    use_market_hours: bool = field(default=True)
    enable_notifications: bool = field(default=False)
    pattern_confidence_threshold: float = field(default=0.6)
    require_indicator_confirmation: bool = field(default=True)
    
    # API settings
    max_retries: int = field(default=3)
    retry_delay: float = field(default=1.0)
    request_timeout: float = field(default=30.0)
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not (1 <= self.max_positions <= 50):
            raise ValueError("max_positions must be between 1 and 50")
        if not (0.001 <= self.max_loss_percent <= 0.20):
            raise ValueError("max_loss_percent must be between 0.1% and 20%")
        if not (0.005 <= self.profit_target_percent <= 1.0):
            raise ValueError("profit_target_percent must be between 0.5% and 100%")
        if not (0.01 <= self.max_daily_loss <= 0.50):
            raise ValueError("max_daily_loss must be between 1% and 50%")
        if not (10 <= self.polling_interval <= 3600):
            raise ValueError("polling_interval must be between 10 and 3600 seconds")
        if not (0.0001 <= self.risk_per_trade_pct <= 0.10):
            raise ValueError("risk_per_trade_pct must be between 0.01% and 10%")
        if not (0.01 <= self.max_position_size_pct <= 0.50):
            raise ValueError("max_position_size_pct must be between 1% and 50%")
        if not (0.0 <= self.pattern_confidence_threshold <= 1.0):
            raise ValueError("pattern_confidence_threshold must be between 0 and 1")


@dataclass
class Position:
    """
    Represents an open trading position with comprehensive tracking.
    """
    symbol: str
    quantity: int
    entry_price: Decimal
    entry_time: dt.datetime
    patterns: List[str]
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop: Optional[Decimal] = None
    unrealized_pnl: Decimal = field(default=Decimal('0'))
    
    def __post_init__(self):
        """Validate position data"""
        if self.quantity <= 0:
            raise ValueError("Position quantity must be positive")
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        
        # Ensure decimal precision
        self.entry_price = Decimal(str(self.entry_price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        if self.stop_loss:
            self.stop_loss = Decimal(str(self.stop_loss)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        if self.take_profit:
            self.take_profit = Decimal(str(self.take_profit)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def update_unrealized_pnl(self, current_price: Decimal) -> None:
        """Update unrealized P&L based on current market price"""
        current_price = Decimal(str(current_price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
    
    def update_trailing_stop(self, current_price: Decimal, trail_percent: float) -> bool:
        """
        Update trailing stop if price moves favorably.
        
        Returns:
            bool: True if trailing stop was updated
        """
        current_price = Decimal(str(current_price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        trail_amount = current_price * Decimal(str(trail_percent))
        new_stop = current_price - trail_amount
        
        if self.trailing_stop is None or new_stop > self.trailing_stop:
            self.trailing_stop = new_stop
            return True
        return False


class APIException(Exception):
    """Custom exception for API-related errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class AuthenticationError(APIException):
    """Exception for authentication failures"""
    pass


class RateLimitError(APIException):
    """Exception for rate limiting"""
    pass


class ETradeClient:
    """
    Production-grade E*Trade API client with comprehensive error handling,
    automatic retries, rate limiting, and session management.
    """
    
    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        oauth_token: str,
        oauth_token_secret: str,
        account_id: str,
        config: TradeConfig
    ):
        """
        Initialize E*Trade API client with enhanced error handling.
        
        Args:
            consumer_key: E*Trade consumer key
            consumer_secret: E*Trade consumer secret  
            oauth_token: OAuth access token
            oauth_token_secret: OAuth token secret
            account_id: E*Trade account ID
            config: Trading configuration object
            
        Raises:
            ValueError: If required credentials are missing
            AuthenticationError: If credential validation fails
        """
        self._validate_credentials(consumer_key, consumer_secret, oauth_token, oauth_token_secret, account_id)
        
        self.config = config
        self.account_id = account_id
        
        # Determine API endpoints based on sandbox mode
        self.sandbox = os.getenv('ETRADE_SANDBOX', 'true').lower() == 'true'
        host = "https://apisb.etrade.com" if self.sandbox else "https://api.etrade.com"
        self.oauth_host = host
        self.base_url = f"{host}/v1"
        
        # Initialize session with retry strategy
        self.session = self._create_session(consumer_key, consumer_secret, oauth_token, oauth_token_secret)
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests
        
        # Account value cache
        self._account_value_cache = None
        self._account_value_cache_time = None
        self._account_cache_ttl = 300  # 5 minutes
        
        # Validate connection
        self._validate_connection()
        
        logger.info(f"ETradeClient initialized successfully ({'SANDBOX' if self.sandbox else 'PRODUCTION'} mode)")
    
    @staticmethod
    def _validate_credentials(consumer_key: str, consumer_secret: str, oauth_token: str, 
                            oauth_token_secret: str, account_id: str) -> None:
        """Validate that all required credentials are provided"""
        required_creds = {
            'consumer_key': consumer_key,
            'consumer_secret': consumer_secret,
            'oauth_token': oauth_token,
            'oauth_token_secret': oauth_token_secret,
            'account_id': account_id
        }
        
        missing = [name for name, value in required_creds.items() if not value or not str(value).strip()]
        if missing:
            raise ValueError(f"Missing required E*Trade API credentials: {', '.join(missing)}")
    
    def _create_session(self, consumer_key: str, consumer_secret: str, 
                       oauth_token: str, oauth_token_secret: str) -> OAuth1Session:
        """Create OAuth session with retry strategy"""
        try:
            session = OAuth1Session(
                client_key=consumer_key,
                client_secret=consumer_secret,
                resource_owner_key=oauth_token,
                resource_owner_secret=oauth_token_secret
            )
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.retry_delay,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to create OAuth session: {e}")
            raise AuthenticationError("Failed to initialize API session") from e
    
    def _validate_connection(self) -> None:
        """Validate API connection and credentials"""
        try:
            response = self._make_request("GET", f"{self.base_url}/accounts/list")
            if not response.get("AccountListResponse", {}).get("Accounts"):
                raise AuthenticationError("No accounts found - invalid credentials")
                
        except APIException:
            raise
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            raise AuthenticationError("Failed to validate API credentials") from e
    
    def _rate_limit(self) -> None:
        """Implement rate limiting to avoid API throttling"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request with comprehensive error handling and retries.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            Dict containing response data
            
        Raises:
            APIException: For various API errors
            AuthenticationError: For authentication issues
            RateLimitError: For rate limiting
        """
        self._rate_limit()
        
        # Set default timeout
        kwargs.setdefault('timeout', self.config.request_timeout)
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.session.request(method, url, **kwargs)
                
                # Handle different HTTP status codes
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    self._handle_authentication_error(response)
                elif response.status_code == 429:
                    self._handle_rate_limit(response, attempt)
                elif response.status_code >= 500:
                    self._handle_server_error(response, attempt)
                else:
                    self._handle_client_error(response)
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt == self.config.max_retries:
                    raise APIException("Request timeout after all retries")
                time.sleep(self.config.retry_delay * (2 ** attempt))
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries:
                    raise APIException("Connection failed after all retries")
                time.sleep(self.config.retry_delay * (2 ** attempt))
                
            except Exception as e:
                logger.error(f"Unexpected error in request: {e}")
                if attempt == self.config.max_retries:
                    raise APIException(f"Request failed: {str(e)}")
                time.sleep(self.config.retry_delay)
        
        raise APIException("Request failed after all retries")
    
    def _handle_authentication_error(self, response: requests.Response) -> None:
        """Handle 401 authentication errors with token renewal"""
        logger.warning("Authentication failed, attempting token renewal...")
        try:
            self.renew_access_token()
            logger.info("Token renewed successfully")
        except Exception as e:
            logger.error(f"Token renewal failed: {e}")
            raise AuthenticationError("Authentication failed and token renewal unsuccessful")
    
    def _handle_rate_limit(self, response: requests.Response, attempt: int) -> None:
        """Handle 429 rate limiting errors"""
        retry_after = int(response.headers.get("Retry-After", self.config.retry_delay * (2 ** attempt)))
        logger.warning(f"Rate limited, waiting {retry_after} seconds...")
        
        if retry_after > 300:  # Don't wait more than 5 minutes
            raise RateLimitError("Rate limit wait time too long")
        
        time.sleep(retry_after)
    
    def _handle_server_error(self, response: requests.Response, attempt: int) -> None:
        """Handle 5xx server errors"""
        logger.warning(f"Server error {response.status_code} on attempt {attempt + 1}")
        if attempt < self.config.max_retries:
            time.sleep(self.config.retry_delay * (2 ** attempt))
        else:
            raise APIException(f"Server error {response.status_code}: {response.text}")
    
    def _handle_client_error(self, response: requests.Response) -> None:
        """Handle 4xx client errors"""
        error_msg = f"Client error {response.status_code}: {response.text}"
        logger.error(error_msg)
        raise APIException(error_msg, response.status_code, response.text)
    
    def renew_access_token(self) -> Dict[str, Any]:
        """
        Renew access token when expired.
        
        Returns:
            Dict containing token renewal response
        """
        renew_url = f"{self.oauth_host}/oauth/renew_access_token"
        
        try:
            response = self.session.get(renew_url, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            result = response.json()
            logger.info("Access token renewed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Token renewal failed: {e}")
            raise AuthenticationError("Failed to renew access token") from e
    
    def get_account_value(self, force_refresh: bool = False) -> Decimal:
        """
        Get current account value with caching.
        
        Args:
            force_refresh: Skip cache and fetch fresh data
            
        Returns:
            Current account value as Decimal
        """
        current_time = time.time()
        
        # Return cached value if still valid
        if (not force_refresh and 
            self._account_value_cache is not None and
            self._account_value_cache_time is not None and
            current_time - self._account_value_cache_time < self._account_cache_ttl):
            return self._account_value_cache
        
        try:
            url = f"{self.base_url}/accounts/{self.account_id}/balance"
            response = self._make_request("GET", url)
            
            balance_data = response.get("BalanceResponse", {})
            account_value = Decimal(str(balance_data.get("accountValue", "0")))
            
            # Update cache
            self._account_value_cache = account_value
            self._account_value_cache_time = current_time
            
            logger.debug(f"Account value retrieved: ${account_value}")
            return account_value
            
        except Exception as e:
            logger.error(f"Failed to get account value: {e}")
            raise APIException("Failed to retrieve account value") from e
    
    def get_candles(self, symbol: str, interval: str = "5min", days: int = 1) -> pd.DataFrame:
        """
        Fetch historical candlestick data with enhanced error handling.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Candle interval ('1min', '5min', '15min', '30min', '1hour', '1day')
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data indexed by datetime
            
        Raises:
            APIException: For API-related errors
            ValueError: For invalid parameters
        """
        # Validate inputs
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        symbol = symbol.strip().upper()
        if not symbol.isalpha() or len(symbol) > 5:
            raise ValueError("Invalid symbol format")
        
        valid_intervals = {'1min', '5min', '15min', '30min', '1hour', '1day'}
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of: {valid_intervals}")
        
        if not isinstance(days, int) or days < 1 or days > 365:
            raise ValueError("Days must be an integer between 1 and 365")
        
        url = f"{self.base_url}/market/productlookup"
        params = {
            "company": symbol,
            "type": "EQ"
        }
        
        try:
            # First verify symbol exists
            response = self._make_request("GET", url, params=params)
            products = response.get("ProductLookupResponse", {}).get("Data", {}).get("Products", [])
            
            if not products:
                logger.warning(f"Symbol {symbol} not found")
                return pd.DataFrame()
            
            # Fetch candle data
            candle_url = f"{self.base_url}/market/productlookup"
            candle_params = {"company": symbol, "type": "EQ"}
            
            response = self._make_request("GET", candle_url, params=candle_params)
            
            # Parse candle data (Note: E*Trade API structure may vary)
            # This is a simplified implementation - adjust based on actual API response
            candles = response.get("Data", {}).get("candles", [])
            
            if not candles:
                logger.warning(f"No candle data available for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            # Standardize column names and data types
            df.rename(columns={
                'dateTime': 'datetime',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }, inplace=True)
            
            # Convert datetime
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', errors='coerce')
                df = df.set_index('datetime')
            
            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            # Sort by datetime
            df = df.sort_index()
            
            logger.debug(f"Retrieved {len(df)} candles for {symbol}")
            return df
            
        except APIException:
            raise
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            raise APIException(f"Failed to fetch candle data for {symbol}") from e
    
    def place_market_order(self, symbol: str, quantity: int, instruction: str = "BUY") -> Dict[str, Any]:
        """
        Place a market order with comprehensive validation and error handling.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares (must be positive)
            instruction: Order side ('BUY' or 'SELL')
            
        Returns:
            Dict containing order response data
            
        Raises:
            ValueError: For invalid parameters
            APIException: For API-related errors
        """
        # Validate inputs
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        symbol = symbol.strip().upper()
        if not symbol.isalpha() or len(symbol) > 5:
            raise ValueError("Invalid symbol format")
        
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("Quantity must be a positive integer")
        
        if quantity > 10000:  # Reasonable limit
            raise ValueError("Quantity exceeds maximum allowed (10,000 shares)")
        
        if instruction not in {"BUY", "SELL"}:
            raise ValueError("Instruction must be 'BUY' or 'SELL'")
        
        # Create order payload
        client_order_id = f"bot-{int(time.time())}-{symbol}"
        order_payload = {
            "PlaceOrderRequest": {
                "orderType": "MARKET",
                "clientOrderId": client_order_id,
                "allOrNone": False,
                "orderTerm": "GOOD_FOR_DAY",
                "priceType": "MARKET",
                "quantity": str(quantity),
                "symbol": symbol,
                "instruction": instruction
            }
        }
        
        url = f"{self.base_url}/accounts/{self.account_id}/orders/place"
        
        try:
            response = self._make_request("POST", url, json=order_payload)
            
            order_id = response.get("PlaceOrderResponse", {}).get("orderId")
            if order_id:
                logger.info(f"Order placed successfully: {instruction} {quantity} {symbol} (ID: {order_id})")
            else:
                logger.warning(f"Order placed but no order ID returned: {instruction} {quantity} {symbol}")
            
            return response
            
        except APIException:
            raise
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            raise APIException(f"Failed to place order for {symbol}") from e
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status by order ID.
        
        Args:
            order_id: E*Trade order ID
            
        Returns:
            Dict containing order status information
        """
        if not order_id:
            raise ValueError("Order ID is required")
        
        url = f"{self.base_url}/accounts/{self.account_id}/orders/{order_id}"
        
        try:
            response = self._make_request("GET", url)
            return response
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            raise APIException(f"Failed to get order status") from e
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current account positions.
        
        Returns:
            List of position dictionaries
        """
        url = f"{self.base_url}/accounts/{self.account_id}/portfolio"
        
        try:
            response = self._make_request("GET", url)
            positions = response.get("PortfolioResponse", {}).get("AccountPortfolio", [])
            return positions if isinstance(positions, list) else []
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise APIException("Failed to get positions") from e


class PerformanceTracker:
    """
    Comprehensive performance tracking and analytics for trading operations.
    """
    
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.daily_returns: Dict[str, Decimal] = {}
        self.start_time = dt.datetime.now()
        self.peak_account_value: Optional[Decimal] = None
        self.max_drawdown: Decimal = Decimal('0')
    
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Add a completed trade to the performance tracker.
        
        Args:
            trade_data: Dict containing trade information
        """
        required_fields = ['symbol', 'entry_price', 'exit_price', 'quantity', 'entry_time', 'exit_time']
        missing_fields = [field for field in required_fields if field not in trade_data]
        
        if missing_fields:
            logger.warning(f"Trade data missing fields: {missing_fields}")
            return
        
        # Calculate trade metrics
        entry_price = Decimal(str(trade_data['entry_price']))
        exit_price = Decimal(str(trade_data['exit_price']))
        quantity = int(trade_data['quantity'])
        
        profit_loss = (exit_price - entry_price) * quantity
        profit_pct = (exit_price - entry_price) / entry_price
        
        trade_data.update({
            'profit_loss': profit_loss,
            'profit_pct': profit_pct,
            'trade_duration': trade_data['exit_time'] - trade_data['entry_time'],
            'timestamp': dt.datetime.now()
        })
        
        self.trades.append(trade_data)
        
        # Update daily returns
        trade_date = trade_data['exit_time'].date().isoformat()
        if trade_date not in self.daily_returns:
            self.daily_returns[trade_date] = Decimal('0')
        self.daily_returns[trade_date] += profit_loss
        
        logger.info(f"Trade recorded: {trade_data['symbol']} P&L: ${profit_loss:.2f} ({profit_pct:.2%})")
    
    def update_account_value(self, current_value: Decimal) -> None:
        """Update account value for drawdown calculation"""
        if self.peak_account_value is None or current_value > self.peak_account_value:
            self.peak_account_value = current_value
        
        if self.peak_account_value and current_value < self.peak_account_value:
            drawdown = (self.peak_account_value - current_value) / self.peak_account_value
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
    
    def calculate_metrics(self) -> Dict[str, Union[int, float, Decimal]]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dict containing various performance metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit_pct': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': float(self.max_drawdown),
                'sharpe_ratio': 0.0,
                'total_pnl': Decimal('0')
            }
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['profit_pct'] > 0]
        losing_trades = [t for t in self.trades if t['profit_pct'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_profit = sum(t['profit_loss'] for t in winning_trades)
        total_loss = abs(sum(t['profit_loss'] for t in losing_trades))
        profit_factor = float(total_profit / total_loss) if total_loss > 0 else float('inf')
        
        # Average returns
        avg_profit_pct = sum(t['profit_pct'] for t in self.trades) / total_trades
        
        # Sharpe ratio (simplified - using daily returns)
        if len(self.daily_returns) > 1:
            returns = list(self.daily_returns.values())
            avg_return = sum(returns) / len(returns)
            return_std = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = float(avg_return / return_std) if return_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_profit_pct': float(avg_profit_pct),
            'profit_factor': profit_factor,
            'max_drawdown': float(self.max_drawdown),
            'sharpe_ratio': sharpe_ratio,
            'total_pnl': sum(t['profit_loss'] for t in self.trades),
            'avg_win': sum(t['profit_loss'] for t in winning_trades) / len(winning_trades) if winning_trades else Decimal('0'),
            'avg_loss': sum(t['profit_loss'] for t in losing_trades) / len(losing_trades) if losing_trades else Decimal('0'),
            'largest_win': max((t['profit_loss'] for t in winning_trades), default=Decimal('0')),
            'largest_loss': min((t['profit_loss'] for t in losing_trades), default=Decimal('0'))
        }
    
    def get_daily_summary(self, date: Optional[dt.date] = None) -> Dict[str, Any]:
        """Get performance summary for a specific date"""
        if date is None:
            date = dt.date.today()
        
        date_str = date.isoformat()
        daily_trades = [t for t in self.trades if t['exit_time'].date() == date]
        daily_pnl = self.daily_returns.get(date_str, Decimal('0'))
        
        return {
            'date': date_str,
            'trades': len(daily_trades),
            'pnl': daily_pnl,
            'winning_trades': len([t for t in daily_trades if t['profit_pct'] > 0]),
            'symbols_traded': list(set(t['symbol'] for t in daily_trades))
        }


class MarketHours:
    """
    Utility class for market hours and trading session management.
    """
    
    @staticmethod
    def get_market_state() -> MarketState:
        """
        Determine current market state based on Eastern Time.
        
        Returns:
            MarketState enum value
        """
        # Convert to Eastern Time
        et_tz = dt.timezone(dt.timedelta(hours=-5))  # EST (adjust for DST if needed)
        now = dt.datetime.now(et_tz)
        
        # Check if it's a weekend
        if now.weekday() > 4:  # Saturday = 5, Sunday = 6
            return MarketState.CLOSED
        
        # Define market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Pre-market: 4:00 AM - 9:30 AM ET
        pre_market_start = now.replace(hour=4, minute=0, second=0, microsecond=0)
        
        # After-hours: 4:00 PM - 8:00 PM ET
        after_hours_end = now.replace(hour=20, minute=0, second=0, microsecond=0)
        
        if market_open <= now <= market_close:
            return MarketState.OPEN
        elif pre_market_start <= now < market_open:
            return MarketState.PRE_MARKET
        elif market_close < now <= after_hours_end:
            return MarketState.AFTER_HOURS
        else:
            return MarketState.CLOSED
    
    @staticmethod
    def is_trading_allowed(config: TradeConfig) -> bool:
        """Check if trading should be allowed based on configuration"""
        if not config.use_market_hours:
            return True
        
        market_state = MarketHours.get_market_state()
        return market_state == MarketState.OPEN
    
    @staticmethod
    def time_until_market_open() -> Optional[dt.timedelta]:
        """Calculate time until next market open"""
        et_tz = dt.timezone(dt.timedelta(hours=-5))
        now = dt.datetime.now(et_tz)
        
        # Find next trading day
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # If market opening time has passed today, move to next trading day
        if now.time() >= dt.time(9, 30):
            next_open += dt.timedelta(days=1)
        
        # Skip weekends
        while next_open.weekday() > 4:
            next_open += dt.timedelta(days=1)
        
        return next_open - now


class StrategyEngine:
    """
    Main trading strategy engine with comprehensive pattern detection,
    risk management, and position monitoring.
    """
    
    def __init__(self, client: ETradeClient, symbols: List[str], config: TradeConfig):
        """
        Initialize the trading strategy engine.
        
        Args:
            client: E*Trade API client
            symbols: List of symbols to monitor
            config: Trading configuration
        """
        self.client = client
        self.config = config
        self.symbols = self._validate_symbols(symbols)
        
        # Core components
        self.positions: Dict[str, Position] = {}
        self.risk_manager = RiskManager(config)
        self.performance_tracker = PerformanceTracker()
        self.pattern_model = PatternNN()
        self.notifier = Notifier() if config.enable_notifications else None
        
        # State management
        self.running = False
        self.daily_pl = Decimal('0')
        self.start_account_value: Optional[Decimal] = None
        
        # Performance optimization
        self._last_symbol_update: Dict[str, float] = {}
        self._symbol_data_cache: Dict[str, pd.DataFrame] = {}
        self._cache_ttl = 60  # 1 minute cache for symbol data
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info(f"StrategyEngine initialized with {len(self.symbols)} symbols")
    
    @staticmethod
    def _validate_symbols(symbols: List[str]) -> List[str]:
        """Validate and clean symbol list"""
        if not symbols:
            raise ValueError("At least one symbol must be provided")
        
        validated_symbols = []
        for symbol in symbols:
            if isinstance(symbol, str) and symbol.strip():
                clean_symbol = symbol.strip().upper()
                if clean_symbol.isalpha() and len(clean_symbol) <= 5:
                    validated_symbols.append(clean_symbol)
                else:
                    logger.warning(f"Invalid symbol format: {symbol}")
            else:
                logger.warning(f"Invalid symbol: {symbol}")
        
        if not validated_symbols:
            raise ValueError("No valid symbols provided")
        
        return validated_symbols
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self) -> None:
        """Start the trading engine"""
        if self.running:
            logger.warning("Trading engine is already running")
            return
        
        logger.info("Starting trading engine...")
        self.running = True
        
        # Initialize account value
        try:
            self.start_account_value = self.client.get_account_value()
            self.performance_tracker.update_account_value(self.start_account_value)
            logger.info(f"Starting account value: ${self.start_account_value}")
        except Exception as e:
            logger.error(f"Failed to get initial account value: {e}")
            self.start_account_value = Decimal('10000')  # Default fallback
        
        # Main trading loop
        try:
            self._run_trading_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Unexpected error in trading loop: {e}")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the trading engine gracefully"""
        if not self.running:
            return
        
        logger.info("Stopping trading engine...")
        self.running = False
        
        # Close all positions if in production mode
        if not self.client.sandbox:
            self._close_all_positions("Engine shutdown")
        
        # Send final performance report
        if self.notifier:
            self._send_final_report()
        
        logger.info("Trading engine stopped successfully")
    
    def _run_trading_loop(self) -> None:
        """Main trading loop with error handling"""
        last_daily_summary = dt.date.today()
        
        while self.running:
            loop_start_time = time.time()
            
            try:
                # Check if trading is allowed
                if not MarketHours.is_trading_allowed(self.config):
                    self._handle_market_closed()
                    continue
                
                # Process all symbols
                self._process_symbols()
                
                # Monitor existing positions
                self._monitor_positions()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Send daily summary if needed
                current_date = dt.date.today()
                if current_date > last_daily_summary:
                    self._send_daily_summary()
                    last_daily_summary = current_date
                
                # Calculate sleep time to maintain polling interval
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, self.config.polling_interval - loop_duration)
                
                if sleep_time > 0:
                    logger.debug(f"Sleeping for {sleep_time:.1f} seconds")
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(10)  # Brief pause before retry
    
    def _handle_market_closed(self) -> None:
        """Handle operations when market is closed"""
        time_until_open = MarketHours.time_until_market_open()
        
        if time_until_open:
            sleep_time = min(3600, time_until_open.total_seconds())  # Max 1 hour
            logger.info(f"Market closed. Sleeping for {sleep_time/60:.1f} minutes")
            time.sleep(sleep_time)
        else:
            time.sleep(300)  # 5 minutes default
    
    def _process_symbols(self) -> None:
        """Process all symbols for entry opportunities"""
        for symbol in self.symbols:
            try:
                # Skip if already have position
                if symbol in self.positions:
                    continue
                
                # Check cache first
                df = self._get_symbol_data(symbol)
                if df.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Evaluate for entry
                self._evaluate_symbol_for_entry(symbol, df)
                
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
    
    def _get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Get symbol data with caching"""
        current_time = time.time()
        last_update = self._last_symbol_update.get(symbol, 0)
        
        # Use cache if data is fresh
        if (current_time - last_update < self._cache_ttl and 
            symbol in self._symbol_data_cache):
            return self._symbol_data_cache[symbol]
        
        # Fetch fresh data
        try:
            df = self.client.get_candles(symbol, interval="5min", days=1)
            if not df.empty:
                self._symbol_data_cache[symbol] = df
                self._last_symbol_update[symbol] = current_time
            return df
        except Exception as e:
            logger.error(f"Failed to get data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _evaluate_symbol_for_entry(self, symbol: str, df: pd.DataFrame) -> None:
        """Evaluate symbol for potential entry"""
        if len(df) < 20:  # Need minimum data for indicators
            return
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Pattern detection
        patterns = self._detect_patterns(df)
        if not patterns:
            return
        
        # Indicator confirmation
        if self.config.require_indicator_confirmation:
            if not self._check_indicator_confirmation(df):
                logger.debug(f"Indicators do not confirm pattern for {symbol}")
                return
        
        # Risk management checks
        current_price = Decimal(str(df['close'].iloc[-1]))
        if not self._check_entry_conditions(symbol, current_price):
            return
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            account_value=self.client.get_account_value(),
            price=current_price,
            config=self.config
        )
        
        if position_size <= 0:
            logger.debug(f"Position size calculation returned 0 for {symbol}")
            return
        
        # Enter position
        self._enter_position(symbol, df, patterns, position_size)
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        try:
            df = TechnicalIndicators.add_rsi(df, length=14)
            df = TechnicalIndicators.add_macd(df, fast=12, slow=26, signal=9)
            df = TechnicalIndicators.add_atr(df, length=14)
            # Add more indicators as needed
            return df
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect candlestick patterns using ML model"""
        try:
            patterns = self.pattern_model.predict(df)
            
            # Filter by confidence threshold
            filtered_patterns = []
            for pattern in patterns:
                if isinstance(pattern, dict) and pattern.get('confidence', 0) >= self.config.pattern_confidence_threshold:
                    filtered_patterns.append(pattern.get('name', 'unknown'))
                elif isinstance(pattern, str):
                    filtered_patterns.append(pattern)
            
            return filtered_patterns
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            return []
    
    def _check_indicator_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if technical indicators confirm the signal"""
        try:
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2] if len(df) > 1 else last_row
            
            confirmations = []
            
            # RSI confirmation (oversold for buy signals)
            if 'rsi' in df.columns:
                rsi_oversold = last_row['rsi'] < 35
                rsi_recovering = last_row['rsi'] > prev_row['rsi']
                confirmations.append(rsi_oversold and rsi_recovering)
            
            # MACD confirmation (bullish crossover)
            if all(col in df.columns for col in ['macd', 'signal']):
                macd_bullish = (last_row['macd'] > last_row['signal'] and 
                              prev_row['macd'] <= prev_row['signal'])
                confirmations.append(macd_bullish)
            
            # Volume confirmation (above average)
            if 'volume' in df.columns and len(df) >= 20:
                avg_volume = df['volume'].tail(20).mean()
                volume_confirmation = last_row['volume'] > avg_volume * 1.5
                confirmations.append(volume_confirmation)
            
            # At least one confirmation required
            return any(confirmations)
            
        except Exception as e:
            logger.error(f"Error checking indicator confirmation: {e}")
            return False
    
    def _check_entry_conditions(self, symbol: str, price: Decimal) -> bool:
        """Check if entry conditions are met"""
        # Maximum positions check
        if len(self.positions) >= self.config.max_positions:
            logger.debug("Maximum positions reached")
            return False
        
        # Daily loss limit check
        if self.daily_pl <= -self.config.max_daily_loss * (self.start_account_value or Decimal('10000')):
            logger.info("Daily loss limit reached")
            return False
        
        # Risk manager validation
        if not self.risk_manager.can_enter_position(
            current_positions=len(self.positions),
            account_value=self.client.get_account_value(),
            daily_pl=self.daily_pl
        ):
            return False
        
        return True
    
    def _enter_position(self, symbol: str, df: pd.DataFrame, patterns: List[str], quantity: int) -> None:
        """Enter a new position"""
        try:
            current_price = Decimal(str(df['close'].iloc[-1]))
            
            # Calculate stop loss and take profit
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * Decimal('0.02')
            stop_loss = current_price - (Decimal(str(atr)) * Decimal('2'))
            take_profit = current_price + (Decimal(str(atr)) * Decimal('3'))
            
            # Place market order
            order_response = self.client.place_market_order(symbol, quantity, "BUY")
            
            # Create position object
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=current_price,
                entry_time=dt.datetime.now(),
                patterns=patterns,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            self.positions[symbol] = position
            
            # Log and notify
            logger.info(f"Entered position: {symbol} @ ${current_price} (qty: {quantity}, patterns: {patterns})")
            
            if self.notifier:
                self.notifier.send_order_notification({
                    'symbol': symbol,
                    'side': 'BUY',
                    'quantity': quantity,
                    'filled_price': float(current_price),
                    'patterns': patterns,
                    'timestamp': dt.datetime.now().isoformat()
                })
        
        except Exception as e:
            logger.error(f"Failed to enter position for {symbol}: {e}")
    
    def _monitor_positions(self) -> None:
        """Monitor all open positions"""
        for symbol in list(self.positions.keys()):
            try:
                self._monitor_position(symbol)
            except Exception as e:
                logger.error(f"Error monitoring position {symbol}: {e}")
    
    def _monitor_position(self, symbol: str) -> None:
        """Monitor individual position for exit conditions"""
        position = self.positions[symbol]
        
        # Get current market data
        df = self._get_symbol_data(symbol)
        if df.empty:
            logger.warning(f"No market data for position monitoring: {symbol}")
            return
        
        current_price = Decimal(str(df['close'].iloc[-1]))
        
        # Update unrealized P&L
        position.update_unrealized_pnl(current_price)
        
        # Update trailing stop
        trail_updated = position.update_trailing_stop(current_price, self.config.max_loss_percent)
        if trail_updated:
            logger.info(f"Updated trailing stop for {symbol}: ${position.trailing_stop}")
        
        # Check exit conditions
        exit_reason = self._check_exit_conditions(position, current_price)
        if exit_reason:
            self._exit_position(symbol, exit_reason)
    
    def _check_exit_conditions(self, position: Position, current_price: Decimal) -> Optional[str]:
        """Check if position should be exited"""
        # Stop loss
        if position.stop_loss and current_price <= position.stop_loss:
            return "Stop loss triggered"
        
        # Trailing stop
        if position.trailing_stop and current_price <= position.trailing_stop:
            return "Trailing stop triggered"
        
        # Take profit
        if position.take_profit and current_price >= position.take_profit:
            return "Take profit triggered"
        
        # Time-based exit (e.g., hold for max 24 hours)
        max_hold_time = dt.timedelta(hours=24)
        if dt.datetime.now() - position.entry_time > max_hold_time:
            return "Maximum hold time reached"
        
        return None
    
    def _exit_position(self, symbol: str, reason: str) -> None:
        """Exit a position"""
        try:
            position = self.positions[symbol]
            
            # Place sell order
            order_response = self.client.place_market_order(symbol, position.quantity, "SELL")
            
            # Get current price for P&L calculation
            df = self._get_symbol_data(symbol)
            exit_price = Decimal(str(df['close'].iloc[-1])) if not df.empty else position.entry_price
            
            # Calculate final P&L
            realized_pnl = (exit_price - position.entry_price) * position.quantity
            self.daily_pl += realized_pnl
            
            # Record trade
            trade_data = {
                'symbol': symbol,
                'entry_price': float(position.entry_price),
                'exit_price': float(exit_price),
                'quantity': position.quantity,
                'entry_time': position.entry_time,
                'exit_time': dt.datetime.now(),
                'patterns': position.patterns,
                'exit_reason': reason,
                'realized_pnl': realized_pnl
            }
            
            self.performance_tracker.add_trade(trade_data)
            
            # Remove position
            del self.positions[symbol]
            
            # Log and notify
            logger.info(f"Exited position: {symbol} @ ${exit_price} (P&L: ${realized_pnl:.2f}, reason: {reason})")
            
            if self.notifier:
                self.notifier.send_order_notification({
                    'symbol': symbol,
                    'side': 'SELL',
                    'quantity': position.quantity,
                    'filled_price': float(exit_price),
                    'pnl': float(realized_pnl),
                    'reason': reason,
                    'timestamp': dt.datetime.now().isoformat()
                })
        
        except Exception as e:
            logger.error(f"Failed to exit position {symbol}: {e}")
    
    def _close_all_positions(self, reason: str) -> None:
        """Close all open positions"""
        for symbol in list(self.positions.keys()):
            try:
                self._exit_position(symbol, reason)
            except Exception as e:
                logger.error(f"Failed to close position {symbol}: {e}")
    
    def _update_performance_metrics(self) -> None:
        """Update performance tracking metrics"""
        try:
            current_account_value = self.client.get_account_value()
            self.performance_tracker.update_account_value(current_account_value)
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    def _send_daily_summary(self) -> None:
        """Send daily performance summary"""
        if not self.notifier:
            return
        
        try:
            metrics = self.performance_tracker.calculate_metrics()
            daily_summary = self.performance_tracker.get_daily_summary()
            
            summary_message = f"""
Daily Trading Summary - {daily_summary['date']}

Trades Today: {daily_summary['trades']}
Daily P&L: ${daily_summary['pnl']:.2f}
Win Rate: {metrics['win_rate']:.1%}
Total Trades: {metrics['total_trades']}
Profit Factor: {metrics['profit_factor']:.2f}
Max Drawdown: {metrics['max_drawdown']:.1%}

Open Positions: {len(self.positions)}
Symbols Traded: {', '.join(daily_summary['symbols_traded'])}
            """.strip()
            
            self.notifier.send_order_notification({
                'symbol': 'DAILY_SUMMARY',
                'side': 'INFO',
                'message': summary_message,
                'timestamp': dt.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
    
    def _send_final_report(self) -> None:
        """Send final performance report on shutdown"""
        try:
            metrics = self.performance_tracker.calculate_metrics()
            
            final_report = f"""
Final Trading Report

Session Duration: {dt.datetime.now() - self.start_time}
Total Trades: {metrics['total_trades']}
Win Rate: {metrics['win_rate']:.1%}
Total P&L: ${metrics['total_pnl']:.2f}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Max Drawdown: {metrics['max_drawdown']:.1%}

The trading session has ended.
            """.strip()
            
            self.notifier.send_order_notification({
                'symbol': 'FINAL_REPORT',
                'side': 'INFO',
                'message': final_report,
                'timestamp': dt.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to send final report: {e}")


def create_config_from_env() -> TradeConfig:
    """
    Create TradeConfig from environment variables with validation.
    
    Returns:
        TradeConfig object with validated parameters
    """
    try:
        config = TradeConfig(
            max_positions=int(os.getenv('MAX_POSITIONS', '5')),
            max_loss_percent=float(os.getenv('MAX_LOSS_PERCENT', '0.02')),
            profit_target_percent=float(os.getenv('PROFIT_TARGET_PERCENT', '0.03')),
            max_daily_loss=float(os.getenv('MAX_DAILY_LOSS', '0.05')),
            polling_interval=int(os.getenv('POLLING_INTERVAL', '300')),
            risk_per_trade_pct=float(os.getenv('RISK_PER_TRADE_PCT', '0.01')),
            max_position_size_pct=float(os.getenv('MAX_POSITION_SIZE_PCT', '0.10')),
            trailing_stop_activation_pct=float(os.getenv('TRAILING_STOP_ACTIVATION_PCT', '0.01')),
            use_market_hours=os.getenv('USE_MARKET_HOURS', 'true').lower() == 'true',
            enable_notifications=os.getenv('ENABLE_NOTIFICATIONS', 'false').lower() == 'true',
            pattern_confidence_threshold=float(os.getenv('PATTERN_CONFIDENCE_THRESHOLD', '0.6')),
            require_indicator_confirmation=os.getenv('REQUIRE_INDICATOR_CONFIRMATION', 'true').lower() == 'true'
        )
        
        logger.info("Configuration loaded from environment variables")
        return config
        
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid configuration in environment variables: {e}")
        logger.info("Using default configuration")
        return TradeConfig()


def main() -> None:
    """
    Main entry point for the trading bot with comprehensive error handling.
    """
    logger.info("Starting ETrade Candlestick Trading Bot")
    
    try:
        # Load configuration
        config = create_config_from_env()
        
        # Get API credentials
        try:
            creds = get_api_credentials()
        except Exception as e:
            logger.error(f"Failed to load API credentials: {e}")
            logger.error("Please ensure .env file is properly configured")
            sys.exit(1)
        
        # Validate required credentials
        required_creds = ['consumer_key', 'consumer_secret', 'oauth_token', 'oauth_token_secret', 'account_id']
        missing_creds = [cred for cred in required_creds if not creds.get(cred)]
        
        if missing_creds:
            logger.error(f"Missing required credentials: {', '.join(missing_creds)}")
            sys.exit(1)
        
        # Initialize E*Trade client
        try:
            client = ETradeClient(
                consumer_key=creds['consumer_key'],
                consumer_secret=creds['consumer_secret'],
                oauth_token=creds['oauth_token'],
                oauth_token_secret=creds['oauth_token_secret'],
                account_id=creds['account_id'],
                config=config
            )
        except Exception as e:
            logger.error(f"Failed to initialize E*Trade client: {e}")
            sys.exit(1)
        
        # Get symbols to monitor
        symbols_str = os.getenv('SYMBOLS', 'AAPL,MSFT,GOOGL,TSLA,AMZN')
        symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
        
        if not symbols:
            logger.error("No symbols provided in SYMBOLS environment variable")
            sys.exit(1)
        
        # Initialize and start strategy engine
        try:
            engine = StrategyEngine(client, symbols, config)
            logger.info(f"Monitoring symbols: {', '.join(symbols)}")
            engine.start()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Strategy engine error: {e}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        sys.exit(1)
    
    finally:
        logger.info("ETrade Candlestick Trading Bot shutdown complete")


if __name__ == "__main__":
    main()