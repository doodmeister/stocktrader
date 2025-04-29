""""
ETrade Candlestick Bot

Main bot engine that connects to E*Trade's API, monitors selected stock symbols for bullish candlestick patterns,
and automatically places trades based on defined risk management rules.

Pattern detection is separated into a 'patterns.py' module.
"""

import time
import logging
import datetime as dt
import pandas as pd
import requests
from requests_oauthlib import OAuth1Session
from typing import List, Dict
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import signal
import sys

# Import patterns module
from patterns import CandlestickPatterns

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
    polling_interval: int = 300

class ETradeClient:
    """Handles authentication, data retrieval, and order placement with E*Trade API"""
    def __init__(self, consumer_key, consumer_secret, oauth_token, oauth_token_secret, account_id, sandbox=True, max_retries=3, retry_delay=1.0):
        self.base_url = f"https://api.etrade.com{'/sandbox' if sandbox else ''}/v1"
        self.account_id = account_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._initialize_session(consumer_key, consumer_secret, oauth_token, oauth_token_secret)
        self._validate_credentials()

    def _initialize_session(self, consumer_key, consumer_secret, oauth_token, oauth_token_secret):
        try:
            self.session = OAuth1Session(
                client_key=consumer_key,
                client_secret=consumer_secret,
                resource_owner_key=oauth_token,
                resource_owner_secret=oauth_token_secret
            )
        except Exception as e:
            logger.error(f"Failed to initialize OAuth session: {str(e)}")
            raise

    def _validate_credentials(self):
        try:
            r = self.session.get(f"{self.base_url}/accounts/list")
            r.raise_for_status()
        except Exception as e:
            logger.error("Failed to validate API credentials")
            raise ValueError("Invalid API credentials") from e

    def get_candles(self, symbol, interval="5min", days=1) -> pd.DataFrame:
        url = f"{self.base_url}/market/quote/{symbol}/candles"
        params = {"interval": interval, "days": days}

        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params)
                r.raise_for_status()
                data = r.json()["candlesResponse"]["candles"]
                df = pd.DataFrame(data)
                df["datetime"] = pd.to_datetime(df["dateTime"], unit="ms")
                df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
                return df
            except requests.exceptions.HTTPError as e:
                if r.status_code == 401:
                    logger.warning("Session expired. Refreshing session...")
                    self._initialize_session(
                        os.getenv("ETRADE_CONSUMER_KEY"),
                        os.getenv("ETRADE_CONSUMER_SECRET"),
                        os.getenv("ETRADE_OAUTH_TOKEN"),
                        os.getenv("ETRADE_OAUTH_TOKEN_SECRET")
                    )
                elif r.status_code == 429:
                    retry_after = int(r.headers.get("Retry-After", 1))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                elif r.status_code >= 500:
                    logger.warning(f"Server error {r.status_code}. Retrying...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"HTTP error {r.status_code}: {r.text}")
                    raise
            except Exception as e:
                logger.error(f"Error fetching candles: {str(e)}")
                time.sleep(self.retry_delay)
        raise RuntimeError("Failed to fetch candles after retries")

    def place_market_order(self, symbol, quantity, instruction="BUY") -> Dict:
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

        for attempt in range(self.max_retries):
            try:
                r = self.session.post(url, json=order)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.HTTPError as e:
                if r.status_code == 401:
                    logger.warning("Session expired. Refreshing session...")
                    self._initialize_session(
                        os.getenv("ETRADE_CONSUMER_KEY"),
                        os.getenv("ETRADE_CONSUMER_SECRET"),
                        os.getenv("ETRADE_OAUTH_TOKEN"),
                        os.getenv("ETRADE_OAUTH_TOKEN_SECRET")
                    )
                elif r.status_code == 429:
                    retry_after = int(r.headers.get("Retry-After", 1))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                elif r.status_code >= 500:
                    logger.warning(f"Server error {r.status_code}. Retrying...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"HTTP error {r.status_code}: {r.text}")
                    raise
            except Exception as e:
                logger.error(f"Error placing order: {str(e)}")
                time.sleep(self.retry_delay)
        raise RuntimeError("Failed to place order after retries")

class StrategyEngine:
    def __init__(self, client: ETradeClient, symbols: List[str], config: TradeConfig):
        self.client = client
        self.symbols = symbols
        self.config = config
        self.positions: Dict[str, Dict] = {}
        self.daily_pl = 0.0
        self.running = True
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
                self.client.place_market_order(symbol, self.positions[symbol]['quantity'], instruction="SELL")
                logger.info(f"Closed position in {symbol}")
            except Exception as e:
                logger.error(f"Failed to close position in {symbol}: {str(e)}")

    def _check_risk_limits(self, symbol, price) -> bool:
        if len(self.positions) >= self.config.max_positions:
            return False
        if self.daily_pl <= -self.config.max_daily_loss:
            return False
        return True

    def run(self):
        while self.running:
            try:
                self._process_symbols()
                self._monitor_positions()
                time.sleep(self.config.polling_interval)
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(10)

    def _process_symbols(self):
        for symbol in self.symbols:
            try:
                df = self.client.get_candles(symbol)
                self._evaluate_symbol(symbol, df)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")

    def _evaluate_symbol(self, symbol, df: pd.DataFrame):
        if symbol in self.positions:
            return

        detected_patterns = CandlestickPatterns.detect_patterns(df)
        if detected_patterns:
            if self._check_risk_limits(symbol, df['close'].iloc[-1]):
                self._enter_position(symbol, df, detected_patterns)

    def _enter_position(self, symbol, df: pd.DataFrame, patterns: list):
        try:
            quantity = 1
            self.client.place_market_order(symbol, quantity, instruction="BUY")
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': df['close'].iloc[-1],
                'pattern': patterns
            }
            logger.info(f"Entered position in {symbol} due to pattern(s): {', '.join(patterns)}")
        except Exception as e:
            logger.error(f"Failed to enter position in {symbol}: {str(e)}")

    def _monitor_positions(self):
        for symbol, position in list(self.positions.items()):
            try:
                df = self.client.get_candles(symbol)
                current_price = df['close'].iloc[-1]
                entry_price = position['entry_price']
                pnl_percent = (current_price - entry_price) / entry_price
                pnl_absolute = pnl_percent * entry_price * position['quantity']

                if pnl_percent >= self.config.profit_target_percent or pnl_percent <= -self.config.max_loss_percent:
                    self.client.place_market_order(symbol, position['quantity'], instruction="SELL")
                    self.daily_pl += pnl_absolute
                    logger.info(f"Exited position in {symbol} with PnL: {pnl_percent:.2%}. Entry pattern(s): {', '.join(position.get('pattern', []))}")
                    del self.positions[symbol]
            except Exception as e:
                logger.error(f"Error monitoring position in {symbol}: {str(e)}")

def main():
    try:
        load_dotenv()
        config = TradeConfig(
            max_positions=int(os.getenv('MAX_POSITIONS', '5')),
            max_loss_percent=float(os.getenv('MAX_LOSS_PERCENT', '0.02')),
            profit_target_percent=float(os.getenv('PROFIT_TARGET_PERCENT', '0.03')),
            max_daily_loss=float(os.getenv('MAX_DAILY_LOSS', '0.05')),
        )

        client = ETradeClient(
            consumer_key=os.getenv('ETRADE_CONSUMER_KEY'),
            consumer_secret=os.getenv('ETRADE_CONSUMER_SECRET'),
            oauth_token=os.getenv('ETRADE_OAUTH_TOKEN'),
            oauth_token_secret=os.getenv('ETRADE_OAUTH_TOKEN_SECRET'),
            account_id=os.getenv('ETRADE_ACCOUNT_ID'),
            sandbox=os.getenv('ETRADE_SANDBOX', 'True').lower() == 'true'
        )

        symbols = os.getenv('SYMBOLS', 'AAPL,MSFT,GOOG').split(',')
        engine = StrategyEngine(client, symbols, config)

        logger.info("Starting trading engine...")
        engine.run()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()