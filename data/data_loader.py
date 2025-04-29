"""stocktrader/data/data_loader.py

Module to fetch daily OHLCV data from Yahoo Finance and save it to CSV.
"""
import os
import time
import logging
from typing import List

import yfinance as yf
import pandas as pd

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_daily_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Yahoo Finance.

    Args:
        symbol (str): Ticker symbol (e.g., 'AAPL').
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing Date, Open, High, Low, Close, Volume.
    """
    logger.info(f"Fetching data for {symbol} from {start} to {end}")
    for attempt in range(3):
        try:
            data = yf.download(symbol, start=start, end=end, progress=False)
            if data.empty:
                raise ValueError("Received empty data.")
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.index.name = 'Date'
            return data
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}: Error fetching data - {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch data for {symbol} after 3 attempts.")

def save_to_csv(data: pd.DataFrame, path: str):
    """
    Save OHLCV data to a CSV file.

    Args:
        data (pd.DataFrame): OHLCV data.
        path (str): Destination path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.to_csv(path)
    logger.info(f"Data saved to {path}")

if __name__ == "__main__":
    # Example usage
    symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    save_path = f"data/{symbol}_daily.csv"

    ohlcv_data = fetch_daily_ohlcv(symbol, start_date, end_date)
    save_to_csv(ohlcv_data, save_path)
