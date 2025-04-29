import time
import logging
import datetime as dt
import pandas as pd
from requests_oauthlib import OAuth1Session
from typing import List, Dict, Optional
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import signal
import sys

# Configure logging
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
    """Configuration for trading parameters"""
    max_positions: int = 5
    max_loss_percent: float = 0.02  # 2% max loss per trade
    profit_target_percent: float = 0.03  # 3% profit target
    max_daily_loss: float = 0.05  # 5% max daily loss
    polling_interval: int = 300  # 5 minutes

class ETradeClient:
    """Enhanced E*Trade REST client with better error handling and rate limiting"""
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
        self.base_url = f"https://api.etrade.com{'sandbox' if sandbox else ''}/v1"
        self.account_id = account_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._initialize_session(consumer_key, consumer_secret, oauth_token, oauth_token_secret)
        self._validate_credentials()

    def _initialize_session(self, consumer_key: str, consumer_secret: str, 
                          oauth_token: str, oauth_token_secret: str) -> None:
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

    def _validate_credentials(self) -> None:
        """Validate API credentials on startup"""
        try:
            r = self.session.get(f"{self.base_url}/accounts/list")
            r.raise_for_status()
        except Exception as e:
            logger.error("Failed to validate API credentials")
            raise ValueError("Invalid API credentials") from e

    def get_candles(self, symbol: str, interval: str = "5min", days: int = 1) -> pd.DataFrame:
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
                        os.getenv("OAUTH_TOKEN"),
                        os.getenv("OAUTH_TOKEN_SECRET")
                    )
                else:
                    logger.error(f"HTTP error: {str(e)}")
                    time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Error fetching candles: {str(e)}")
                time.sleep(self.retry_delay)
        raise RuntimeError("Failed to fetch candles after retries")

    def place_market_order(self, symbol: str, quantity: int, instruction: str = "BUY") -> Dict:
        url = f"{self.base_url}/accounts/{self.account_id}/orders/place"
        order = {
            "orderType": "MARKET",
            "clientOrderId": f"cli-{int(time.time())}",
            "order": {
                "allOrNone": "false",
                "limitPrice": 0,
                "markerPrice": 0,
                "orderTerm": "GOOD_FOR_DAY",
                "stopPrice": 0,
                "bidType": instruction,
                "priceType": "MARKET",
                "quantity": quantity,
                "symbol": symbol
            }
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
                        os.getenv("OAUTH_TOKEN"),
                        os.getenv("OAUTH_TOKEN_SECRET")
                    )
                else:
                    logger.error(f"HTTP error: {str(e)}")
                    time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Error placing order: {str(e)}")
                time.sleep(self.retry_delay)
        raise RuntimeError("Failed to place order after retries")

class CandlestickPatterns:
    @staticmethod
    def is_hammer(df: pd.DataFrame) -> bool:
        o, h, l, c = df["open"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
        body = abs(c - o)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        return (lower_wick > 2 * body) and (upper_wick < body)

    @staticmethod
    def is_bullish_engulfing(df: pd.DataFrame) -> bool:
        prev, last = df.iloc[-2], df.iloc[-1]
        prev_body = abs(prev.close - prev.open)
        last_body = abs(last.close - last.open)
        return (prev.close < prev.open) and (last.close > last.open) and \
               (last.open < prev.close) and (last.close > prev.open)

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
        """Handle graceful shutdown"""
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        """Cleanup and exit gracefully"""
        logger.info("Shutdown signal received, closing positions...")
        self.running = False
        self._close_all_positions()
        sys.exit(0)

    def _close_all_positions(self):
        """Close all open positions"""
        for symbol in self.positions:
            try:
                self.client.place_market_order(
                    symbol, 
                    self.positions[symbol]['quantity'],
                    instruction="SELL"
                )
                logger.info(f"Closed position in {symbol}")
            except Exception as e:
                logger.error(f"Failed to close position in {symbol}: {str(e)}")

    def _check_risk_limits(self, symbol: str, price: float) -> bool:
        """Check if trade meets risk management criteria"""
        if len(self.positions) >= self.config.max_positions:
            return False
        if self.daily_pl <= -self.config.max_daily_loss:
            return False
        return True

    def run(self):
        """Main trading loop with improved error handling and risk management"""
        while self.running:
            try:
                self._process_symbols()
                self._monitor_positions()
                time.sleep(self.config.polling_interval)
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(10)  # Back off on error

    def _process_symbols(self):
        """Process each symbol for trading opportunities"""
        for symbol in self.symbols:
            try:
                df = self.client.get_candles(symbol, interval="5min", days=1)
                self._evaluate_symbol(symbol, df)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")

    def _evaluate_symbol(self, symbol: str, df: pd.DataFrame):
        """Evaluate trading signals for a symbol"""
        if symbol in self.positions:
            return

        if CandlestickPatterns.is_hammer(df) or CandlestickPatterns.is_bullish_engulfing(df):
            if self._check_risk_limits(symbol, df['close'].iloc[-1]):
                self._enter_position(symbol, df)

    def _enter_position(self, symbol: str, df: pd.DataFrame):
        """Enter a new position"""
        try:
            quantity = 1  # Example quantity, can be adjusted
            self.client.place_market_order(symbol, quantity, instruction="BUY")
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': df['close'].iloc[-1]
            }
            logger.info(f"Entered position in {symbol}")
        except Exception as e:
            logger.error(f"Failed to enter position in {symbol}: {str(e)}")

    def _monitor_positions(self):
        """Monitor open positions for exit conditions"""
        for symbol, position in list(self.positions.items()):
            try:
                df = self.client.get_candles(symbol, interval="5min", days=1)
                current_price = df['close'].iloc[-1]
                entry_price = position['entry_price']
                pnl_percent = (current_price - entry_price) / entry_price

                if pnl_percent >= self.config.profit_target_percent or pnl_percent <= -self.config.max_loss_percent:
                    self.client.place_market_order(symbol, position['quantity'], instruction="SELL")
                    self.daily_pl += pnl_percent
                    logger.info(f"Exited position in {symbol} with PnL: {pnl_percent:.2%}")
                    del self.positions[symbol]
            except Exception as e:
                logger.error(f"Error monitoring position in {symbol}: {str(e)}")

def main():
    """Application entry point with proper initialization and error handling"""
    try:
        load_dotenv()
        
        config = TradeConfig(
            max_positions=int(os.getenv('MAX_POSITIONS', '5')),
            max_loss_percent=float(os.getenv('MAX_LOSS_PERCENT', '0.02')),
            profit_target_percent=float(os.getenv('PROFIT_TARGET_PERCENT', '0.03')),
            max_daily_loss=float(os.getenv('MAX_DAILY_LOSS', '0.05'))
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
