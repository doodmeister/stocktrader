# indicators.py
"""
Technical indicators module wrapping pandas-ta or similar for chart overlays.

Functions:
    - add_rsi(df, length=14) -> pd.DataFrame
    - add_macd(df, fast=12, slow=26, signal=9) -> pd.DataFrame
    - add_bollinger_bands(df, length=20, std=2) -> pd.DataFrame
"""
import pandas as pd
try:
    import pandas_ta as ta
except ImportError:
    ta = None


def add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """
    Calculate RSI and append as 'rsi' column.
    """
    if ta:
        df['rsi'] = ta.rsi(df['close'], length=length)
    else:
        # Fallback implementation
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(length).mean()
        loss = -delta.clip(upper=0).rolling(length).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD histogram and append 'macd','macd_signal','macd_hist' columns.
    """
    if ta:
        macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
        df[['macd','macd_signal','macd_hist']] = macd
    else:
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
    return df


def add_bollinger_bands(df: pd.DataFrame, length: int = 20, std: int = 2) -> pd.DataFrame:
    """
    Calculate Bollinger Bands and append 'bb_upper','bb_middle','bb_lower'.
    """
    if ta:
        bb = ta.bbands(df['close'], length=length, std=std)
        df[['bb_upper','bb_middle','bb_lower']] = bb[[f'BBU_{length}_{std}', f'BBM_{length}_{std}', f'BBL_{length}_{std}']]
    else:
        ma = df['close'].rolling(length).mean()
        sd = df['close'].rolling(length).std()
        df['bb_middle'] = ma
        df['bb_upper'] = ma + std * sd
        df['bb_lower'] = ma - std * sd
    return df
