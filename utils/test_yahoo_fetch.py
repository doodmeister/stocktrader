# test_yahoo_fetch.py

import yfinance as yf
import pandas as pd
from datetime import date

# Parameters
symbol = "AAPL"
start_date = "2024-04-01"
end_date = "2025-05-01"
interval = "1d"

print(f"Fetching {symbol} from {start_date} to {end_date} ({interval})...")

df = yf.download(
    symbol,
    start=start_date,
    end=end_date,
    interval=interval,
    progress=True,
    auto_adjust=False
)

print("\nRaw DataFrame:")
print(df)

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