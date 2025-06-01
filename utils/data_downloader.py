"""
stocktrader/data/data_downloader.py

Robust module to fetch OHLCV data from Yahoo Finance and save it to CSV,
with full integration of enterprise-grade data validation from core.data_validator.

Features:
- Symbol validation (API-aware)
- Interval/date validation
- DataFrame/OHLCV integrity checks
- Detailed logging and notification
"""

import os
import time
from typing import Dict, List, Optional
from datetime import date, timedelta, datetime
from pathlib import Path

import yfinance as yf
import pandas as pd

from utils.logger import setup_logger
from core.data_validator import (
    validate_symbol,
    validate_dates,
    validate_dataframe,
    get_global_validator,
)

logger = setup_logger(__name__)

def fetch_daily_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a symbol between start and end dates (inclusive).
    Now with input and output validation.
    """
    # --- Symbol Validation ---
    val_result = validate_symbol(symbol)
    if not val_result.is_valid:
        raise ValueError(f"Invalid symbol: {symbol} - {val_result.errors}")

    symbol = val_result.value
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    # --- Date Validation ---
    date_result = validate_dates(start_date, end_date, interval="1d")
    if not date_result.is_valid:
        raise ValueError(f"Invalid date range: {date_result.errors}")

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

    # --- DataFrame Validation ---
    df_result = validate_dataframe(df, required_cols=["open", "high", "low", "close", "volume"])
    if not df_result.is_valid:
        raise ValueError(f"Data for {symbol} failed validation: {df_result.errors}")

    return df_result.value


def _period_from_days(days: int) -> str:
    """Map a number of days to yfinance’s supported period strings."""
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
    notifier=None
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Fetch OHLCV data for one or more symbols using Ticker.history,
    with comprehensive validation at each stage.
    """
    # --- Symbol Validation (batch) ---
    valid_symbols = []
    for symbol in symbols:
        sym_result = validate_symbol(symbol)
        if sym_result.is_valid:
            valid_symbols.append(sym_result.value)
        else:
            logger.warning(f"Skipping invalid symbol: {symbol} - {sym_result.errors}")

    if not valid_symbols:
        logger.warning("No valid symbols after validation")
        return None

    # --- Date Validation ---
    date_result = validate_dates(start_date, end_date, interval)
    if not date_result.is_valid:
        logger.warning(f"Invalid date range: {date_result.errors}")
        return None

    # --- Fetching Data ---
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    window_days = (end_date - start_date).days + 1
    period_str = _period_from_days(window_days)

    try:
        # --- Batch Fetch ---
        symbol_str = ",".join(valid_symbols)
        logger.info(f"Batch fetching via history(): {symbol_str} ({interval})")
        df = yf.Ticker(symbol_str).history(
            period=period_str,
            interval=interval,
            auto_adjust=False,
            actions=False,
            timeout=timeout
        )
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]

        # --- Process & Validate DataFrames ---
        result = process_downloaded_data(df, valid_symbols)
        cleaned_results = {}
        if result:
            for symbol, symbol_df in result.items():
                df_val = validate_dataframe(symbol_df, required_cols=["open", "high", "low", "close", "volume"])
                if df_val.is_valid:
                    cleaned_results[symbol] = df_val.value
                else:
                    logger.warning(f"Data for {symbol} failed validation: {df_val.errors}")
            if cleaned_results:
                return cleaned_results

        # --- Per-symbol Fallback ---
        logger.warning("Batch empty or processing failed, falling back per-symbol history+slice.")
        fallback_results = {}
        for symbol in valid_symbols:
            try:
                hist = yf.Ticker(symbol).history(
                    period=period_str,
                    interval=interval,
                    auto_adjust=False,
                    actions=False,
                    timeout=timeout
                )
                if hasattr(hist.index, "tz") and hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                hist = hist.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
                if hist.empty:
                    logger.warning(f"No data for {symbol} in fallback mode")
                    continue

                symbol_df = hist[["Open", "High", "Low", "Close", "Volume"]]
                symbol_df.columns = [c.lower() for c in symbol_df.columns]
                # DataFrame Validation
                df_val = validate_dataframe(symbol_df, required_cols=["open", "high", "low", "close", "volume"])
                if df_val.is_valid:
                    fallback_results[symbol] = df_val.value
                else:
                    logger.warning(f"Data for {symbol} failed validation: {df_val.errors}")
            except Exception as e:
                logger.warning(f"Per-symbol history+slice failed for {symbol}: {e}")

        return fallback_results if fallback_results else None

    except Exception as e:
        logger.exception(f"Error downloading data for {valid_symbols}: {e}")
        if notifier:
            for method in ("send_notification", "notify", "send"):
                if hasattr(notifier, method):
                    getattr(notifier, method)(
                        f"Critical: Failed to download data for {', '.join(valid_symbols)}. Error: {str(e)}"
                    )
                    break
            else:
                logger.warning("No notification method on Notifier; skipping")
        return None


def process_downloaded_data(df: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Process a downloaded DataFrame into individual symbol DataFrames.
    DataFrames are NOT assumed valid—validate with validate_dataframe after!
    """
    result = {}
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    if isinstance(df.columns, pd.MultiIndex):
        # Handle MultiIndex columns (multiple symbols)
        for symbol in symbols:
            symbol_df = pd.DataFrame(index=df.index)
            missing_cols = []
            for col in required_cols:
                col_tuple = (col, symbol)
                if col_tuple in df.columns:
                    symbol_df[col.lower()] = df[col_tuple]
                else:
                    missing_cols.append(col)
            if missing_cols:
                logger.warning(f"Missing columns for {symbol}: {missing_cols}")
            symbol_df.dropna(how="all", inplace=True)
            if not symbol_df.empty and not missing_cols:
                result[symbol] = symbol_df
    else:
        # Handle single-level columns (single symbol)
        symbol_df = pd.DataFrame(index=df.index)
        missing_cols = []
        for col in required_cols:
            if col in df.columns:
                symbol_df[col.lower()] = df[col]
            else:
                missing_cols.append(col)
        if missing_cols:
            logger.warning(f"Missing columns for {symbols[0]}: {missing_cols}")
        symbol_df.dropna(how="all", inplace=True)
        if not symbol_df.empty and not missing_cols:
            result[symbols[0]] = symbol_df

    if result:
        logger.debug(f"Data dict keys: {list(result.keys())}")
        logger.debug(f"AAPL data shape: {result.get('AAPL', pd.DataFrame()).shape}")
    return result if result else None


def save_to_csv(data: pd.DataFrame, path: str):
    """Save OHLCV data to a CSV file (directory auto-created)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.to_csv(path)
    logger.info(f"Data saved to {path}")


def clear_cache():
    """Clear the in-memory data cache (dummy function for compatibility)."""
    logger.info("Data cache cleared")
