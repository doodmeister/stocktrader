"""stocktrader/data/data_loader.py

Module to fetch OHLCV data from Yahoo Finance and save it to CSV.
"""
import os
import time
import logging
from typing import Dict, List, Optional
from datetime import date, timedelta, datetime
from pathlib import Path
import functools

import yfinance as yf
import pandas as pd
from utils.config.validation import sanitize_input, validate_symbol, safe_request

# Set up basic + debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def fetch_daily_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a symbol between start and end dates (inclusive).
    Uses period-based fetch and slices to exact window.
    """
    symbol = symbol.upper()
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    window_days = (end_date - start_date).days + 1
    ticker = yf.Ticker(symbol)

    hist = ticker.history(period=f"{window_days}d", interval="1d")
    if hist.empty:
        raise ValueError(f"No data returned for {symbol}")

    hist.index = pd.to_datetime(hist.index)
    df = hist.loc[pd.to_datetime(start_date):pd.to_datetime(end_date), ["Open", "High", "Low", "Close", "Volume"]]

    if df.empty:
        raise ValueError(f"No data in requested date range {start} to {end} for {symbol}")

    df.columns = [c.lower() for c in df.columns]
    df.index.name = "Date"
    return df


def _period_from_days(days: int) -> str:
    """
    Map a number of days to one of yfinance’s supported period strings.
    """
    if days <= 7:
        return f"{days}d"
    if days <= 30:
        return "1mo"
    if days <= 90:
        return "3mo"
    if days <= 180:
        return "6mo"
    if days <= 365:
        return "1y"
    if days <= 730:
        return "2y"
    return "max"


def download_stock_data(
    symbols: List[str], 
    start_date: date, 
    end_date: date, 
    interval: str = "1d",
    timeout: int = 30,
    notifier = None
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Fetch OHLCV data for one or more symbols using Ticker.history.
    """
    # Normalize all symbols to uppercase
    symbols = [s.upper() for s in symbols]
    
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

    # Convert date objects to string format for internal functions
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    try:
        # --- Batch via history + slice (more reliable than yf.download in-streamlit) ---
        symbol_str = ",".join(sanitized_symbols)
        logger.info(f"Batch fetching via history(): {symbol_str} ({interval})")

        window_days = (end_date - start_date).days + 1
        period_str  = _period_from_days(window_days)

        df = yf.Ticker(symbol_str).history(
            period=period_str,
            interval=interval,
            auto_adjust=False,
            actions=False,
            timeout=timeout
        )
        # --- DEBUG ---
        logger.debug(f"[Batch] raw.history() shape={df.shape}")
        # Strip timezone from index if present
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # slice to exact window - convert dates to pandas datetime for slicing
        df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
        logger.debug(f"[Batch] after slice shape={df.shape}")

        # If we have data, process it into the result dict
        if df is not None and not df.empty:
            result = process_downloaded_data(df, sanitized_symbols)
            if result:
                return result

        # If we get here, we need the per-symbol fallback
        logger.warning(f"Batch empty or processing failed, falling back per-symbol history+slice.")
        result: Dict[str, pd.DataFrame] = {}
        
        # Add the fallback implementation here
        window_days = (end_date - start_date).days + 1
        period_str = _period_from_days(window_days)
        
        for symbol in sanitized_symbols:
            try:
                hist = yf.Ticker(symbol).history(
                    period=period_str,
                    interval=interval,
                    auto_adjust=False,
                    actions=False,
                    timeout=timeout
                )
                
                # Strip timezone if present
                if hasattr(hist.index, "tz") and hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                    
                if hist.empty:
                    logger.warning(f"No history for {symbol} ({period_str})")
                    continue
                
                # Use pd.to_datetime for consistent date format handling
                sliced = hist.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
                if sliced.empty:
                    logger.warning(f"Sliced empty for {symbol} ({start_date}->{end_date})")
                    continue
                
                df_clean = sliced[["Open","High","Low","Close","Volume"]]
                df_clean.columns = [c.lower() for c in df_clean.columns]
                result[symbol] = df_clean
                
            except Exception as e:
                logger.warning(f"Per-symbol history+slice failed for {symbol}: {e}")
                
        return result if result else None

    except Exception as e:
        logger.exception(f"Error downloading data for {symbols}: {e}")
        if notifier:
            for method in ("send_notification", "notify", "send"):
                if hasattr(notifier, method):
                    getattr(notifier, method)(
                        f"Critical: Failed to download data for {', '.join(symbols)}. Error: {str(e)}"
                    )
                    break
            else:
                logger.warning("No notification method on Notifier; skipping")
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

    if result:
        logger.debug(f"Data dict keys: {list(result.keys())}")
        logger.debug(f"AAPL data shape: {result.get('AAPL', pd.DataFrame()).shape}")

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
    logger.info("Data cache cleared")
