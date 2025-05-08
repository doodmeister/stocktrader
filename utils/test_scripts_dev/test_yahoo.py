# test_yahoo_fetch_msft.py

import yfinance as yf
import pandas as pd

# 1) Parameters
symbol     = "MSFT"
start_date = "2025-05-05"
end_date   = "2025-05-06"   # we want the full trading day of May 5
interval   = "5m"           # 5-minute bars

print(f"Fetching {symbol} from {start_date} to {end_date} ({interval})…")

# 2) Use Ticker.history to avoid the extra MultiIndex header
ticker = yf.Ticker(symbol)
df = ticker.history(
    start=start_date,
    end=end_date,
    interval=interval,
    auto_adjust=False,
    prepost=False 
)

# 3) Convert timestamps from UTC → US/Eastern, then drop tzinfo
if df.index.tz is not None:
    df.index = df.index.tz_convert("America/New_York").tz_localize(None)

print("\nSample of the DataFrame (now in ET):")
print(df.iloc[:5])
print("…")
print(df.iloc[-5:])

# 4) Isolate the 12:45pm bar in Eastern Time
target = pd.to_datetime("2025-05-05 12:45:00")
if target in df.index:
    bar = df.loc[target]
    print(f"\nBar at {target} ET:")
    print(f"  Open:   {bar['Open']:.8f}")
    print(f"  High:   {bar['High']:.8f}")
    print(f"  Low:    {bar['Low']:.8f}")
    print(f"  Close:  {bar['Close']:.8f}")
    print(f"  Volume: {int(bar['Volume'])}")
else:
    print(f"\nNo bar found at {target} ET. Available times:\n", df.index.tolist())
