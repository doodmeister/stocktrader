"""
ETrade Candlestick Bot

Main bot engine that connects to E*Trade's API, monitors selected stock symbols for bullish candlestick patterns,
and automatically places trades based on defined risk management rules.

Pattern detection is separated into a 'patterns_nn.py' module.
"""

import os
import time
import logging
import datetime as dt
import pandas as pd
import requests
from requests_oauthlib import OAuth1Session
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import signal
import sys

from patterns_nn import PatternNN

# Configure logging to both file and console for traceability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradeConfig:
    """Configuration dataclass for trading parameters"""
    max_positions: int = 5
    max_loss_percent: float = 0.02
    profit_target_percent: float = 0.03
    max_daily_loss: float = 0.05
    polling_interval: int = 300  # seconds

class ETradeClient:
    """
    Handles authentication, data retrieval, and order placement with E*Trade API.
    Implements robust error handling, credential validation, and token renewal.
    """
    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        oauth_token: str,
        oauth_token_secret: str,
        account_id: str,
        sandbox: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        if not all([consumer_key, consumer_secret, oauth_token, oauth_token_secret, account_id]):
            logger.error("Missing required E*Trade API credentials.")
            raise ValueError("All E*Trade API credentials must be provided.")

        # Determine OAuth host (for renew/revoke) and API base URL
        host = "https://apisb.etrade.com" if sandbox else "https://api.etrade.com"
        self.oauth_host = host
        self.base_url = f"{host}/v1"
        self.account_id = account_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize session and validate
        self._initialize_session(consumer_key, consumer_secret, oauth_token, oauth_token_secret)
        self._validate_credentials()

    def renew_access_token(self):
        """
        Reactivate an access token after 2 hours of inactivity or past midnight ET.
        """
        renew_url = f"{self.oauth_host}/oauth/renew_access_token"
        resp = self.session.get(renew_url)
        resp.raise_for_status()
        logger.info("Access token renewed successfully.")
        return resp.json()

    def _initialize_session(self, consumer_key, consumer_secret, oauth_token, oauth_token_secret):
        try:
            self.session = OAuth1Session(
                client_key=consumer_key,
                client_secret=consumer_secret,
                resource_owner_key=oauth_token,
                resource_owner_secret=oauth_token_secret
            )
        except Exception as e:
            logger.error(f"Failed to initialize OAuth session: {e}")
            raise

    def _validate_credentials(self):
        try:
            r = self.session.get(f"{self.base_url}/accounts/list")
            r.raise_for_status()
        except Exception as e:
            logger.error("Failed to validate API credentials")
            raise ValueError("Invalid API credentials") from e

    def get_candles(self, symbol: str, interval: str = "5min", days: int = 1) -> pd.DataFrame:
        """
        Fetch historical candlestick data for a given symbol.
        """
        url = f"{self.base_url}/market/quote/{symbol}/candles"
        params = {"interval": interval, "days": days}

        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params)
                r.raise_for_status()
                data = r.json().get("candlesResponse", {}).get("candles", [])
                if not data:
                    logger.warning(f"No candle data returned for {symbol}.")
                    return pd.DataFrame()
                df = pd.DataFrame(data)
                df["datetime"] = pd.to_datetime(df["dateTime"], unit="ms")
                df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
                return df
            except requests.exceptions.HTTPError as e:
                status = getattr(r, "status_code", None)
                if status == 401:
                    logger.warning("Session expired. Attempting token renewal...")
                    try:
                        self.renew_access_token()
                    except Exception as renew_err:
                        logger.error(f"Token renew failed: {renew_err}")
                    # re-init session with latest env vars
                    self._initialize_session(
                        os.getenv("ETRADE_CONSUMER_KEY"),
                        os.getenv("ETRADE_CONSUMER_SECRET"),
                        os.getenv("ETRADE_OAUTH_TOKEN"),
                        os.getenv("ETRADE_OAUTH_TOKEN_SECRET")
                    )
                elif status == 429:
                    retry_after = int(r.headers.get("Retry-After", 1))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                elif status and status >= 500:
                    logger.warning(f"Server error {status}. Retrying...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"HTTP error {status}: {r.text if r is not None else str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Error fetching candles for {symbol}: {e}")
                time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to fetch candles for {symbol} after {self.max_retries} retries")

    def place_market_order(self, symbol: str, quantity: int, instruction: str = "BUY") -> Dict:
        """
        Place a market order for a given symbol.
        """
        if not symbol or quantity <= 0 or instruction not in {"BUY", "SELL"}:
            logger.error("Invalid order parameters.")
            raise ValueError("Invalid order parameters.")

        url = f"{self.base_url}/accounts/{self.account_id}/orders/place"
        order = {
            "orderType": "MARKET",
            "clientOrderId": f"cli-{int(time.time())}",
            "allOrNone": False,
            "orderTerm": "GOOD_FOR_DAY",
            "priceType": "MARKET",
            "quantity": quantity,
            "symbol": symbol,
            "instruction": instruction
        }
        payload = {"PlaceOrderRequest": order}

        for attempt in range(self.max_retries):
            try:
                r = self.session.post(url, json=payload)
                r.raise_for_status()
                logger.info(f"Order placed: {instruction} {quantity} {symbol}")
                return r.json()
            except requests.exceptions.HTTPError as e:
                status = getattr(r, "status_code", None)
                if status == 401:
                    logger.warning("Session expired. Attempting token renewal...")
                    try:
                        self.renew_access_token()
                    except Exception as renew_err:
                        logger.error(f"Token renew failed: {renew_err}")
                    self._initialize_session(
                        os.getenv("ETRADE_CONSUMER_KEY"),
                        os.getenv("ETRADE_CONSUMER_SECRET"),
                        os.getenv("ETRADE_OAUTH_TOKEN"),
                        os.getenv("ETRADE_OAUTH_TOKEN_SECRET")
                    )
                elif status == 429:
                    retry_after = int(r.headers.get("Retry-After", 1))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                elif status and status >= 500:
                    logger.warning(f"Server error {status}. Retrying...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"HTTP error {status}: {r.text if r is not None else str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Error placing order for {symbol}: {e}")
                time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to place order for {symbol} after {self.max_retries} retries")

class StrategyEngine:
    """
    Main trading strategy engine. Monitors symbols, detects patterns, manages positions, and enforces risk controls.
    """
    def __init__(self, client: ETradeClient, symbols: List[str], config: TradeConfig):
        self.client = client
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        self.config = config
        self.positions: Dict[str, Dict] = {}
        self.daily_pl = 0.0
        self.running = True
        self.pattern_model = PatternNN()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received, closing positions...")
        self.running = False
        self._close_all_positions()
        sys.exit(0)

    def _close_all_positions(self):
        for symbol in list(self.positions.keys()):
            try:
                self.client.place_market_order(symbol,
                                               self.positions[symbol]['quantity'],
                                               instruction="SELL")
                logger.info(f"Closed position in {symbol}")
            except Exception as e:
                logger.error(f"Failed to close position in {symbol}: {e}")

    def _check_risk_limits(self, symbol: str, price: float) -> bool:
        if len(self.positions) >= self.config.max_positions:
            logger.info("Max positions reached. Skipping new entry.")
            return False
        if self.daily_pl <= -self.config.max_daily_loss:
            logger.info("Max daily loss reached. Skipping new entry.")
            return False
        return True

    def run(self):
        logger.info("Strategy engine started.")
        while self.running:
            try:
                self._process_symbols()
                self._monitor_positions()
                time.sleep(self.config.polling_interval)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)

    def _process_symbols(self):
        for symbol in self.symbols:
            try:
                df = self.client.get_candles(symbol)
                if df.empty:
                    logger.warning(f"No data for {symbol}, skipping.")
                    continue
                self._evaluate_symbol(symbol, df)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    def _evaluate_symbol(self, symbol: str, df: pd.DataFrame):
        if symbol in self.positions:
            return

        detected_patterns = self.pattern_model.predict(df)
        if detected_patterns:
            price = df['close'].iloc[-1]
            if self._check_risk_limits(symbol, price):
                self._enter_position(symbol, df, detected_patterns)

    def _enter_position(self, symbol: str, df: pd.DataFrame, patterns: List[str]):
        try:
            quantity = 1  # TODO: Integrate with risk manager for dynamic sizing
            self.client.place_market_order(symbol, quantity, instruction="BUY")
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': df['close'].iloc[-1],
                'pattern': patterns
            }
            logger.info(f"Entered position in {symbol} due to pattern(s): {', '.join(patterns)}")
        except Exception as e:
            logger.error(f"Failed to enter position in {symbol}: {e}")

    def _monitor_positions(self):
        for symbol, position in list(self.positions.items()):
            try:
                df = self.client.get_candles(symbol)
                if df.empty:
                    logger.warning(f"No data for {symbol} while monitoring position.")
                    continue
                current_price = df['close'].iloc[-1]
                entry_price = position['entry_price']
                pnl_percent = (current_price - entry_price) / entry_price
                pnl_absolute = pnl_percent * entry_price * position['quantity']

                if (
                    pnl_percent >= self.config.profit_target_percent or
                    pnl_percent <= -self.config.max_loss_percent
                ):
                    self.client.place_market_order(symbol,
                                                   position['quantity'],
                                                   instruction="SELL")
                    self.daily_pl += pnl_absolute
                    logger.info(
                        f"Exited position in {symbol} with PnL: {pnl_percent:.2%}. "
                        f"Entry pattern(s): {', '.join(position.get('pattern', []))}"
                    )
                    del self.positions[symbol]
            except Exception as e:
                logger.error(f"Error monitoring position in {symbol}: {e}")

def main():
    try:
        load_dotenv()
        config = TradeConfig(
            max_positions=int(os.getenv('MAX_POSITIONS', '5')),
            max_loss_percent=float(os.getenv('MAX_LOSS_PERCENT', '0.02')),
            profit_target_percent=float(os.getenv('PROFIT_TARGET_PERCENT', '0.03')),
            max_daily_loss=float(os.getenv('MAX_DAILY_LOSS', '0.05')),
            polling_interval=int(os.getenv('POLLING_INTERVAL', '300'))
        )

        client = ETradeClient(
            consumer_key=os.getenv('ETRADE_CONSUMER_KEY'),
            consumer_secret=os.getenv('ETRADE_CONSUMER_SECRET'),
            oauth_token=os.getenv('ETRADE_OAUTH_TOKEN'),
            oauth_token_secret=os.getenv('ETRADE_OAUTH_TOKEN_SECRET'),
            account_id=os.getenv('ETRADE_ACCOUNT_ID'),
            sandbox=os.getenv('ETRADE_USE_SANDBOX', 'true').lower() == 'true'
        )

        symbols = os.getenv('SYMBOLS', 'AAPL,MSFT,GOOG').split(',')
        engine = StrategyEngine(client, symbols, config)

        logger.info("Starting trading engine...")
        engine.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
