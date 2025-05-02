"""stocktrader/data/data_loader.py

Module to fetch OHLCV data from Yahoo Finance and save it to CSV.
"""
import os
import time
import logging
from typing import Dict, List, Optional
from datetime import date, timedelta
from pathlib import Path
import functools

import yfinance as yf
import pandas as pd
from utils.validation import sanitize_input, validate_symbol, safe_request

# Set up basic + debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Optional in-memory cache for fetched data
_data_cache = {}


def fetch_daily_ohlcv(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Yahoo Finance.
    """
    logger.info(f"Fetching data for {symbol} from {start} to {end} ({interval})")
    
    def fetch_attempt():
        # use the Ticker.history API here
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            start=start,
            end=end,
            interval=interval,
            actions=False,
            auto_adjust=False,
            timeout=30
        )
        if data.empty:
            raise ValueError(f"No data returned for {symbol} ({interval})")
        # select and lowercase columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data.index.name = 'Date'
        data.columns = [c.lower() for c in data.columns]
        return data

    result = safe_request(
        func=fetch_attempt,
        max_retries=3,
        retry_delay=1,
        timeout=30
    )
    
    if result is None:
        raise RuntimeError(f"Failed to fetch data for {symbol}")
    
    return result


def download_stock_data(
    symbols: List[str], 
    start_date: date, 
    end_date: date, 
    interval: str = "1d",
    timeout: int = 30,
    notifier = None
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Download and validate OHLCV stock data for one or more symbols.
    """
    # Generate cache key
    cache_key = f"{','.join(sorted(symbols))}-{start_date}-{end_date}-{interval}"
    if cache_key in _data_cache:
        logger.info(f"Using cached data for {symbols}")
        return _data_cache[cache_key]
    
    # Validate inputs
    if not symbols:
        logger.warning("No valid symbols provided")
        return None
        
    # Create sanitized symbol list using validation utility
    sanitized_symbols = []
    for symbol in symbols:
        try:
            sanitized = validate_symbol(symbol)
            sanitized_symbols.append(sanitized)
        except ValueError as e:
            logger.warning(f"Skipping invalid symbol: {symbol} - {e}")
            
    if not sanitized_symbols:
        logger.warning("No valid symbols after sanitization")
        return None

    try:
        # make end inclusive (yfinance end= is exclusive)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str   = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
        # use ASCII arrow in logs so Windows console won’t choke
        symbol_str = ",".join(sanitized_symbols)
        logger.info(f"Batch fetching: {symbol_str}  {start_str} -> {end_date.isoformat()} ({interval})")
        df = yf.download(
            symbol_str,
            start=start_str,
            end=end_str,
            interval=interval,
            progress=False,
            timeout=timeout,
            auto_adjust=False
        )
        # ——— DEBUG LOGGING ———
        logger.debug(f"[Batch] df.shape={None if df is None else df.shape}")
        if df is not None and not df.empty:
            logger.debug(
                f"[Batch] index {df.index.min()} -> {df.index.max()}, "
                f"cols {df.columns.tolist()}"
            )
        # ————————————————

        if df is None or df.empty:
            logger.warning(f"Batch empty for {symbol_str}, falling back to fetch_daily_ohlcv().")
            result: Dict[str, pd.DataFrame] = {}
            for symbol in sanitized_symbols:
                try:
                    single = fetch_daily_ohlcv(
                        symbol,
                        start=start_str,
                        end=end_str,
                        interval=interval
                    )
                    # fetch_daily_ohlcv already lower-cases its columns
                    result[symbol] = single
                except Exception as e:
                    logger.warning(f"fetch_daily_ohlcv failed for {symbol}: {e}")
            if not result:
                logger.warning("No data returned after fallback for any symbol.")
                return None
            _data_cache[cache_key] = result
            return result

        # Process downloaded data into individual symbol DataFrames
        result = process_downloaded_data(df, sanitized_symbols)
        _data_cache[cache_key] = result
        return result

    except Exception as e:
        logger.exception(f"Error downloading data for {symbols}: {e}")
        if notifier:
            try:
                notifier.send_notification(
                    f"Critical: Failed to download data for {', '.join(symbols)}. Error: {str(e)}"
                )
            except Exception as notify_err:
                logger.error(f"Notifier failed: {notify_err}")
        return None


def process_downloaded_data(df: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Process a downloaded DataFrame into individual symbol DataFrames.
    
    Args:
        df: Raw DataFrame from yfinance
        symbols: List of symbols to extract
        
    Returns:
        Dict of symbol -> DataFrame with standardized columns
    """
    result = {}
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    if isinstance(df.columns, pd.MultiIndex):
        # Handle MultiIndex columns (multiple symbols)
        for symbol in symbols:
            symbol_df = pd.DataFrame(index=df.index)
            missing_cols = []
            
            # Extract each required column
            for col in required_cols:
                col_tuple = (col, symbol)
                if col_tuple in df.columns:
                    symbol_df[col.lower()] = df[col_tuple]
                else:
                    missing_cols.append(col)
                    
            # Log any missing columns
            if missing_cols:
                logger.warning(f"Missing columns for {symbol}: {missing_cols}")
            
            # Include if we have data
            symbol_df.dropna(how="all", inplace=True)
            if not symbol_df.empty and not missing_cols:
                result[symbol] = symbol_df
    else:
        # Handle single-level columns (single symbol)
        symbol_df = pd.DataFrame(index=df.index)
        missing_cols = []
        
        # Extract each required column
        for col in required_cols:
            if col in df.columns:
                symbol_df[col.lower()] = df[col]
            else:
                missing_cols.append(col)
        
        # Log any missing columns
        if missing_cols:
            logger.warning(f"Missing columns for {symbols[0]}: {missing_cols}")
            
        # Include if we have data
        symbol_df.dropna(how="all", inplace=True)
        if not symbol_df.empty and not missing_cols:
            result[symbols[0]] = symbol_df

    return result if result else None


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


def clear_cache():
    """Clear the in-memory data cache."""
    global _data_cache
    _data_cache = {}
    logger.info("Data cache cleared")
