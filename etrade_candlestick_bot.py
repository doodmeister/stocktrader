import time
import hmac
import hashlib
import base64
import datetime as dt
import requests
import pandas as pd
from requests_oauthlib import OAuth1Session
from typing import List, Dict

# ─────── Requirements ──────────────────────────────────────────────────────────
# pip install requests requests_oauthlib pandas

# ─────── E*TRADE API CLIENT ───────────────────────────────────────────────────

class ETradeClient:
    """
    Minimal E*Trade REST client for market data & order placement.
    """
    def __init__(self,
                 consumer_key: str,
                 consumer_secret: str,
                 oauth_token: str,
                 oauth_token_secret: str,
                 account_id: str,
                 sandbox: bool = True):
        base = "https://api.etrade.com" if not sandbox else "https://api.etrade.com/sandbox"
        self.base_url = base + "/v1"
        self.account_id = account_id

        # OAuth1 session
        self.session = OAuth1Session(
            client_key=consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=oauth_token,
            resource_owner_secret=oauth_token_secret
        )

    def get_candles(self,
                    symbol: str,
                    interval: str = "5min",
                    days: int = 1) -> pd.DataFrame:
        """
        Fetch recent intraday candlestick data.
        interval: '1min','5min','10min','30min','1day'
        days: how many days back
        """
        url = f"{self.base_url}/market/quote/{symbol}/candles"
        params = {
            "interval": interval,
            "days": days
        }
        r = self.session.get(url, params=params)
        r.raise_for_status()
        data = r.json()["candlesResponse"]["candles"]
        df = pd.DataFrame(data)
        # Normalize timestamp
        df["datetime"] = pd.to_datetime(df["dateTime"], unit="ms")
        df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
        return df

    def place_market_order(self,
                           symbol: str,
                           quantity: int,
                           instruction: str = "BUY") -> Dict:
        """
        Place a simple market order.
        instruction: "BUY" or "SELL"
        """
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
        r = self.session.post(url, json=order)
        r.raise_for_status()
        return r.json()

# ─────── CANDLESTICK PATTERNS ──────────────────────────────────────────────────

class CandlestickPatterns:
    @staticmethod
    def is_hammer(df: pd.DataFrame) -> bool:
        """
        Hammer: small body, long lower wick, near the low of the day.
        Checks the last candle.
        """
        o, h, l, c = df["open"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
        body = abs(c - o)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        return (lower_wick > 2 * body) and (upper_wick < body)

    @staticmethod
    def is_bullish_engulfing(df: pd.DataFrame) -> bool:
        """
        Bullish Engulfing: previous candle bearish, last candle bullish,
        and last body engulfs previous body.
        """
        prev, last = df.iloc[-2], df.iloc[-1]
        prev_body = abs(prev.close - prev.open)
        last_body = abs(last.close - last.open)
        return (prev.close < prev.open) and (last.close > last.open) and \
               (last.open < prev.close) and (last.close > prev.open)

# ─────── STRATEGY ENGINE ──────────────────────────────────────────────────────

class StrategyEngine:
    def __init__(self, client: ETradeClient, symbols: List[str], qty: int = 1):
        self.client = client
        self.symbols = symbols
        self.qty = qty

    def run(self):
        """
        Main loop: for each symbol, fetch data, detect patterns,
        and place orders accordingly.
        """
        for symbol in self.symbols:
            df = self.client.get_candles(symbol, interval="5min", days=1)
            print(f"\n[{dt.datetime.now()}] {symbol}")
            if CandlestickPatterns.is_hammer(df):
                print("  → Hammer detected. Placing BUY order.")
                resp = self.client.place_market_order(symbol, self.qty, instruction="BUY")
                print("   Order response:", resp)
            elif CandlestickPatterns.is_bullish_engulfing(df):
                print("  → Bullish Engulfing detected. Placing BUY order.")
                resp = self.client.place_market_order(symbol, self.qty, instruction="BUY")
                print("   Order response:", resp)
            else:
                print("  → No entry signal.")

# ─────── MAIN ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # — fill these in from your E*Trade developer dashboard —
    CONSUMER_KEY = "YOUR_CONSUMER_KEY"
    CONSUMER_SECRET = "YOUR_CONSUMER_SECRET"
    OAUTH_TOKEN = "YOUR_OAUTH_TOKEN"
    OAUTH_TOKEN_SECRET = "YOUR_OAUTH_TOKEN_SECRET"
    ACCOUNT_ID = "YOUR_ACCOUNT_ID"

    client = ETradeClient(
        consumer_key=CONSUMER_KEY,
        consumer_secret=CONSUMER_SECRET,
        oauth_token=OAUTH_TOKEN,
        oauth_token_secret=OAUTH_TOKEN_SECRET,
        account_id=ACCOUNT_ID,
        sandbox=True  # change to False for production
    )

    symbols_to_track = ["AAPL", "MSFT", "GOOG"]
    engine = StrategyEngine(client, symbols_to_track, qty=1)

    # Run every 5 minutes
    while True:
        try:
            engine.run()
        except Exception as e:
            print("Error:", e)
        time.sleep(5 * 60)
