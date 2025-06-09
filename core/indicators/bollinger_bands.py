\
"""
core/indicators/bollinger_bands.py

Bollinger Bands calculation.
"""
import pandas as pd
from pandas import DataFrame # Added for type hint consistency
from typing import Union
from utils.logger import setup_logger
from .base import IndicatorError, validate_input

logger = setup_logger(__name__)

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("pandas_ta not found - using fallback for Bollinger Bands calculations")

@validate_input(['close'])
def calculate_bollinger_bands(df: DataFrame, length: int = 20, std: Union[int, float] = 2, 
                            close_col: str = 'close') -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with price data
        length: Period for moving average
        std: Standard deviation multiplier
        close_col: Name of close price column
        
    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    try:
        if length < 1:
            raise ValueError("Length must be positive")
        if std <= 0:
            raise ValueError("Standard deviation multiplier must be positive")
            
        if TA_AVAILABLE:
            bb = ta.bbands(df[close_col], length=length, std=std)
            if bb is None or not hasattr(bb, "columns") or bb.empty: # Added bb.empty check
                raise ValueError("pandas_ta.bbands returned None or empty. Check input data for sufficient rows and NaNs")
            
            # Try all reasonable column name variants for each band
            std_variants = [str(std), f"{float(std):.1f}", f"{float(std)}"] # pandas_ta can vary column names
            upper_col_name = None
            middle_col_name = None
            lower_col_name = None

            for std_str_variant in std_variants:
                potential_upper = f'BBU_{length}_{std_str_variant}'
                if potential_upper in bb.columns:
                    upper_col_name = potential_upper
                    break
            
            for std_str_variant in std_variants:
                potential_middle = f'BBM_{length}_{std_str_variant}'
                if potential_middle in bb.columns:
                    middle_col_name = potential_middle
                    break

            for std_str_variant in std_variants:
                potential_lower = f'BBL_{length}_{std_str_variant}'
                if potential_lower in bb.columns:
                    lower_col_name = potential_lower
                    break
            
            if not all([upper_col_name, middle_col_name, lower_col_name]):
                # Fallback: try to infer by common prefixes if exact names fail
                if 'BBU' in bb.columns.str.upper() and 'BBM' in bb.columns.str.upper() and 'BBL' in bb.columns.str.upper():
                    upper_col_name = bb.columns[bb.columns.str.upper().str.startswith('BBU')][0]
                    middle_col_name = bb.columns[bb.columns.str.upper().str.startswith('BBM')][0]
                    lower_col_name = bb.columns[bb.columns.str.upper().str.startswith('BBL')][0]
                else:
                    raise KeyError(f"Bollinger Bands columns (e.g., BBU_{length}_{std}, BBM_{length}_{std}, BBL_{length}_{std}) not found in DataFrame. Available: {bb.columns}")
                
            return bb[upper_col_name], bb[middle_col_name], bb[lower_col_name]
        else:
            ma = df[close_col].rolling(window=length, min_periods=1).mean() # min_periods=1 for robustness
            sd = df[close_col].rolling(window=length, min_periods=1).std()
            upper = ma + (std * sd)
            lower = ma - (std * sd)
            
            return upper, ma, lower
            
    except Exception as e:
        logger.error(f"Bollinger Bands calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate Bollinger Bands: {e}") from e
