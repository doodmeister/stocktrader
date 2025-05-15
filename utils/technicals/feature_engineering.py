import pandas as pd
from .technical_analysis import TechnicalAnalysis

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicator columns (RSI, MACD, Bollinger Bands, ATR, etc.)
    to the DataFrame and returns the enriched DataFrame.
    """
    ta = TechnicalAnalysis(df)
    df = df.copy()
    # Add RSI
    df['rsi_14'] = ta.rsi(period=14)
    # Add MACD and signal
    macd_line, macd_signal = ta.macd()
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    # Add Bollinger Bands
    bb_upper, bb_lower = ta.bollinger_bands()
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    # Add ATR
    df['atr_14'] = ta.atr(period=14)
    # Fill NaNs if desired
    df = df.fillna(0)
    return df