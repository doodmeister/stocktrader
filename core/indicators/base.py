\
"""
core/indicators/base.py

Base utilities for technical indicator calculations.
"""
import pandas as pd
from pandas import DataFrame
from typing import List
from utils.logger import setup_logger
from core.validation.dataframe_validation_logic import perform_dataframe_validation_logic # Updated import
from core.validation.validation_results import DataFrameValidationResult # Updated import

logger = setup_logger(__name__)

class IndicatorError(Exception):
    """Custom exception for indicator calculation errors."""
    pass

def validate_indicator_data(df: DataFrame, required_columns: List[str], check_ohlcv_coherence: bool = False) -> None: # Added check_ohlcv_coherence
    """
    Validate DataFrame for technical indicator calculations using centralized validation logic.
    
    Args:
        df: Input DataFrame to validate
        required_columns: List of required column names
        check_ohlcv_coherence: Whether to perform OHLCV coherence checks.
        
    Raises:
        TypeError: If input is not a DataFrame
        ValueError: If validation fails (e.g. missing columns, empty DataFrame)
        IndicatorError: For more specific validation failures from the centralized logic.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Validation failed: Expected pandas DataFrame, got {type(df)}")
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

    # Use the centralized validation logic
    # Basic checks like empty or missing columns are handled first for efficiency.
    if df.empty:
        logger.error("Validation failed: DataFrame is empty.")
        raise ValueError("DataFrame is empty")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Validation failed: Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Determine if full OHLCV check is needed based on required columns and parameter
    # This is a simplified check; perform_dataframe_validation_logic handles more detailed scenarios.
    perform_ohlcv_check = check_ohlcv_coherence and all(col in df.columns for col in ['open', 'high', 'low', 'close'])


    validation_result: DataFrameValidationResult = perform_dataframe_validation_logic(
        df=df,
        required_cols=required_columns,
        check_ohlcv=perform_ohlcv_check, # Use the new parameter
        min_rows=1 # Indicators generally need at least one row
        # Not enabling anomaly detection by default for indicators, can be added if specific indicators need it
    )

    if not validation_result.is_valid:
        error_summary = "; ".join(validation_result.errors) if validation_result.errors else "Unknown validation error."
        logger.error(f"Indicator data validation failed: {error_summary}")
        # Optionally, include more details from validation_result.error_details if needed
        raise IndicatorError(f"DataFrame validation failed for indicator: {error_summary}")
    
    logger.debug(f"Centralized validation passed for indicator data: {len(df)} rows, {len(df.columns)} columns")


def validate_input(required_cols, check_ohlcv_coherence: bool = False): # Added check_ohlcv_coherence
    """
    Decorator to validate DataFrame input for indicator functions.
    Usage: @validate_input(['close'])
           @validate_input(['open', 'high', 'low', 'close'], check_ohlcv_coherence=True)
    """
    def decorator(func):
        def wrapper(df, *args, **kwargs):
            # Pass check_ohlcv_coherence to the validation function
            validate_indicator_data(df, required_cols, check_ohlcv_coherence=check_ohlcv_coherence)
            return func(df, *args, **kwargs)
        return wrapper
    return decorator
