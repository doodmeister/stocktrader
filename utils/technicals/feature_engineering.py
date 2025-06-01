import pandas as pd
from core.technical_indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands, 
    calculate_atr, IndicatorError
)
from utils.logger import get_dashboard_logger

logger = get_dashboard_logger(__name__)

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicator columns (RSI, MACD, Bollinger Bands, ATR, etc.)
    to the DataFrame and returns the enriched DataFrame.
    
    This function has been refactored to use the centralized technical analysis 
    architecture (core.technical_indicators.py) for improved performance, 
    reliability, and maintainability.
    """
    try:
        df = df.copy()
        logger.info("Computing technical features using centralized architecture")
        
        # Use centralized core functions for technical indicators
        logger.debug("Calculating RSI using core.technical_indicators")
        df['rsi_14'] = calculate_rsi(df, length=14)
        
        logger.debug("Calculating MACD using core.technical_indicators")
        macd_line, macd_signal, _ = calculate_macd(df, fast=12, slow=26, signal=9)
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal
        
        logger.debug("Calculating Bollinger Bands using core.technical_indicators")
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df, length=20, std=2)
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        
        logger.debug("Calculating ATR using core.technical_indicators")
        df['atr_14'] = calculate_atr(df, length=14)
        
        # Drop rows with NaN values introduced by indicators
        original_rows = len(df)
        df.dropna(inplace=True)
        dropped_rows = original_rows - len(df)
        
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with NaN values from technical indicators")
        
        # Fill remaining NaNs if desired (optional, keeping the original behavior)
        df = df.fillna(0)
        
        logger.info(f"Successfully computed technical features. Final dataset: {len(df)} rows")
        return df
        
    except IndicatorError as e:
        logger.error(f"Technical indicator calculation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in compute_technical_features: {e}")
        raise