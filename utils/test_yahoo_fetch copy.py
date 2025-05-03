# test_yahoo_fetch.py

import yfinance as yf
import pandas as pd
from datetime import date

symbol      = "AAPL"
start_date  = date(2024, 4, 1)
end_date    = date(2025, 5, 1)
interval    = "1d"
window_days = (end_date - start_date).days + 1

print(f"Fetching {symbol} via history({window_days}d), slicing {start_date}→{end_date}")

ticker = yf.Ticker(symbol)
hist = ticker.history(period=f"{window_days}d", interval=interval)
print("Raw history shape:", hist.shape)

hist.index = pd.to_datetime(hist.index)
df = hist.loc[pd.to_datetime(start_date) : pd.to_datetime(end_date), 
              ["Open","High","Low","Close","Volume"]]
print("Sliced DF shape:", df.shape)
print(df.head(), df.tail())

if df is not None and not df.empty:
    print(f"\nRows: {len(df)} | Date range: {df.index.min()} to {df.index.max()}")
    close_series = None
    # Handle MultiIndex columns (as returned for multiple tickers or some yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        print("MultiIndex columns:", list(df.columns))
        # Try to find the ('Close', symbol) column
        if ("Close", symbol) in df.columns:
            close_series = df[("Close", symbol)]
        else:
            # Try to find any column where the first level is 'Close'
            for col in df.columns:
                if col[0].lower() == "close":
                    close_series = df[col]
                    break
        if close_series is None:
            print("No 'Close' column found in MultiIndex DataFrame.")
    else:
        # Single-level columns
        if "Close" in df.columns:
            close_series = df["Close"]
        else:
            print("No 'Close' column found in DataFrame.")

    if close_series is not None and len(close_series) > 1:
        first = close_series.iloc[0]
        last = close_series.iloc[-1]
        returns = ((last / first) - 1) * 100
        print(f"Period return: {returns:.2f}% (from {first:.2f} to {last:.2f})")
    else:
        print("No 'Close' column or insufficient data for return calculation.")
else:
    print("No data returned! Check symbol, date range, or Yahoo Finance availability.")