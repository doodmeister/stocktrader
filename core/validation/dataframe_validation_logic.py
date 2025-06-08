# filepath: c:\\dev\\stocktrader\\core\\validation\\dataframe_validation_logic.py
"""
Logic for DataFrame validation, extracted from DataValidator.
"""
import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

# Assuming ValidationConfig, ValidationResult, DataFrameValidationResult are in core.validation
from core.validation.validation_config import ValidationConfig
from core.validation.validation_results import ValidationResult, DataFrameValidationResult

logger = logging.getLogger(__name__)

def _validate_ohlc_logic(df_ohlc: pd.DataFrame) -> ValidationResult:
    """Validate OHLC consistency (High >= Open, Low, Close; Low <= Open, High, Close)."""
    # Expects a DataFrame with 'Open', 'High', 'Low', 'Close' columns.
    errors: List[str] = []
    stats = {'rows_checked': len(df_ohlc)}

    if df_ohlc.empty:
        return ValidationResult(is_valid=True, errors=None, validated_data=stats)

    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df_ohlc.columns]
    if missing_cols:
        errors.append(f"Missing OHLC columns for detailed check: {', '.join(missing_cols)}")
        return ValidationResult(is_valid=False, errors=errors, validated_data=stats)

    # Ensure numeric types before comparison
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df_ohlc[col]):
            errors.append(f"OHLC column '{col}' is not numeric.")
    if errors:
            return ValidationResult(is_valid=False, errors=errors, validated_data=stats)

    invalid_high_rows = df_ohlc[ (df_ohlc['High'] < df_ohlc['Open']) | \
                                     (df_ohlc['High'] < df_ohlc['Low'])  | \
                                     (df_ohlc['High'] < df_ohlc['Close']) ].index
    if not invalid_high_rows.empty:
        errors.append(f"High price is not the maximum in {len(invalid_high_rows)} rows (e.g., index {invalid_high_rows[0]}).")

    invalid_low_rows = df_ohlc[ (df_ohlc['Low'] > df_ohlc['Open']) | \
                                (df_ohlc['Low'] > df_ohlc['High']) | \
                                (df_ohlc['Low'] > df_ohlc['Close']) ].index
    if not invalid_low_rows.empty:
        errors.append(f"Low price is not the minimum in {len(invalid_low_rows)} rows (e.g., index {invalid_low_rows[0]}).")

    if errors:
        logger.warning(f"OHLC data validation found inconsistencies. Errors: {errors}")
        return ValidationResult(is_valid=False, errors=errors, validated_data=stats)
    else:
        logger.debug("OHLC data consistency check passed.")
        return ValidationResult(is_valid=True, errors=None, validated_data=stats)

def _detect_anomalies_logic(df: pd.DataFrame, level: str = "basic",
                            target_cols: Optional[List[str]] = None) -> ValidationResult:
    """
    Detect anomalies in DataFrame columns (e.g., outliers using IQR or Z-score).
    'level' can be 'basic' (IQR) or 'advanced' (more complex methods).
    'target_cols': specific columns to check, defaults to numeric columns.
    Returns ValidationResult. Anomalies found are typically logged as warnings or info,
    but can be errors if configured. validated_data contains anomaly stats.
    """
    warnings_log: List[str] = [] 
    errors: List[str] = [] 
    anomaly_stats: Dict[str, Any] = {'level': level, 'anomalies_found': 0, 'details': {}}

    if df.empty:
        return ValidationResult(is_valid=True, errors=None, validated_data=anomaly_stats)

    if target_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in target_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if not numeric_cols:
        warnings_log.append("No numeric columns found or specified for anomaly detection.")
        anomaly_stats['message'] = "No numeric columns for detection."
        if warnings_log: # Log if any warnings were generated
            logger.info(f"Anomaly detection: {warnings_log}")
        return ValidationResult(is_valid=True, errors=None, validated_data=anomaly_stats)

    for col in numeric_cols:
        data_col = df[col].dropna()
        if data_col.empty:
            continue

        if level == "basic": # IQR method
            Q1 = data_col.quantile(0.25)
            Q3 = data_col.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data_col[(data_col < lower_bound) | (data_col > upper_bound)]
            if not outliers.empty:
                msg = f"Column '{col}' has {len(outliers)} potential outliers (IQR method)."
                warnings_log.append(msg)
                anomaly_stats['details'][col] = {
                    'count': len(outliers), 
                    'method': 'IQR',
                    'bounds': (lower_bound, upper_bound),
                    'sample_outliers': outliers.head(3).tolist()
                }
                anomaly_stats['anomalies_found'] += len(outliers)
        
        elif level == "advanced": # Placeholder for Z-score or other methods
            # Example: Z-score
            # mean = data_col.mean()
            # std = data_col.std()
            # z_score_threshold = 3 
            # outliers = data_col[np.abs((data_col - mean) / std) > z_score_threshold]
            # ... (similar logic as above)
            warnings_log.append(f"Advanced anomaly detection for column '{col}' is not fully implemented yet.")
            anomaly_stats['details'][col] = {'message': 'Advanced detection pending implementation.'}

    if warnings_log: # Log all warnings collected during anomaly detection
        logger.info(f"Anomaly detection for level '{level}' completed. Warnings: {warnings_log}")

    # Decide if anomalies constitute an error based on configuration or severity (not implemented here)
    # For now, anomalies are treated as warnings, so is_valid is True unless other errors occur.
    if errors: # If any specific error conditions were added
        return ValidationResult(is_valid=False, errors=errors, validated_data=anomaly_stats)
    
    return ValidationResult(is_valid=True, errors=None, validated_data=anomaly_stats)


def perform_dataframe_validation_logic(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    check_ohlcv: bool = True,
    min_rows: Optional[int] = 1,
    max_rows: Optional[int] = None,
    detect_anomalies_level: Optional[str] = None,
    max_null_percentage: float = 0.1,
    # Removed time import, as start_time is managed by the caller if needed for overall call duration.
    # The validation_time_seconds logged here is specific to this logic block.
) -> DataFrameValidationResult:
    """
    Core logic for DataFrame validation.
    This function is extracted from DataValidator.validate_dataframe.
    """
    # Internal start_time for this specific logic block's duration logging
    logic_start_time = pd.Timestamp.now().timestamp() # Using pandas for consistency if already imported

    errors: List[str] = []
    error_details: Dict[Union[int, str], List[str]] = {} 
    warnings_log: List[str] = []

    if not isinstance(df, pd.DataFrame):
        logger.error("Invalid input: df is not a pandas DataFrame.")
        return DataFrameValidationResult(is_valid=False, errors=["Input is not a DataFrame"], validated_data=None, error_details=None)

    if df.empty:
        if min_rows is not None and min_rows > 0:
            errors.append("DataFrame is empty.")
            logger.warning("DataFrame validation: DataFrame is empty.")
            return DataFrameValidationResult(is_valid=False, errors=errors, validated_data=df.copy(), error_details=error_details)
        else:
            warnings_log.append("DataFrame is empty, but this is acceptable by configuration (min_rows=0 or None).")

    row_count = len(df)
    if min_rows is not None and row_count < min_rows:
        errors.append(f"DataFrame has {row_count} rows, less than minimum required {min_rows}.")
    if max_rows is not None and row_count > max_rows:
        errors.append(f"DataFrame has {row_count} rows, more than maximum allowed {max_rows}.")

    actual_cols = df.columns.tolist()
    default_ohlcv_for_check = getattr(ValidationConfig, 'DEFAULT_OHLCV_COLUMNS', ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    current_required_cols = []
    if required_cols is None and check_ohlcv: 
        current_required_cols = default_ohlcv_for_check
    elif required_cols is not None:
        current_required_cols = required_cols
    
    if current_required_cols:
        missing_cols = [col for col in current_required_cols if col not in actual_cols]
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}.")

    null_counts_series = None
    if not df.empty:
        null_counts_series = df.isnull().sum()
        for col_name, count in null_counts_series.items():
            col_key = str(col_name) 
            if count > 0:
                null_pct = count / row_count
                if null_pct > max_null_percentage:
                    err_msg = f"Column '{col_key}' has {count} nulls ({null_pct:.2%}), exceeding max of {max_null_percentage:.0%}._PERCENTAGE_." # Corrected placeholder
                    errors.append(err_msg)
                    if col_key not in error_details:
                        error_details[col_key] = []
                    error_details[col_key].append(f"Exceeds null threshold ({null_pct:.0%})")
                else:
                    warnings_log.append(f"Column '{col_key}' has {count} nulls ({null_pct:.2%}).")
    
    if check_ohlcv and current_required_cols and not df.empty:
        for col_name in [c for c in default_ohlcv_for_check if c in df.columns]:
            col_key = str(col_name) 
            if not pd.api.types.is_numeric_dtype(df[col_key]):
                errors.append(f"Column '{col_key}' is not numeric as expected for OHLCV data.")
                if col_key not in error_details:
                    error_details[col_key] = []
                error_details[col_key].append(f"Non-numeric data found: {df[col_key].dtype}")
            elif df[col_key].isnull().all():
                warnings_log.append(f"Column '{col_key}' is numeric but all values are null.")

    ohlc_stats = {}
    if check_ohlcv and not df.empty and all(c in df.columns for c in ['Open', 'High', 'Low', 'Close']):
        # Pass only relevant columns that exist to avoid KeyErrors if some are missing
        ohlc_cols_present = [c for c in ['Open', 'High', 'Low', 'Close'] if c in df.columns]
        if len(ohlc_cols_present) == 4: # Only if all 4 primary OHLC are there
            ohlc_result = _validate_ohlc_logic(df[ohlc_cols_present])
            if not ohlc_result.is_valid and ohlc_result.errors:
                errors.extend(ohlc_result.errors)
            if ohlc_result.validated_data and isinstance(ohlc_result.validated_data, dict):
                ohlc_stats = ohlc_result.validated_data
        else:
            warnings_log.append("Skipping detailed OHLC validation due to missing one or more of Open, High, Low, Close columns.")


    anomaly_stats = {}
    if detect_anomalies_level and not df.empty:
        anomaly_result = _detect_anomalies_logic(df, level=detect_anomalies_level) # target_cols can be added if needed
        if not anomaly_result.is_valid and anomaly_result.errors:
            errors.extend(anomaly_result.errors)
        if anomaly_result.validated_data and isinstance(anomaly_result.validated_data, dict):
            anomaly_stats = anomaly_result.validated_data

    logic_validation_time = pd.Timestamp.now().timestamp() - logic_start_time
    log_metadata = {
        'validation_time_seconds': round(logic_validation_time, 4),
        'row_count': row_count,
        'column_count': len(actual_cols),
        'columns': actual_cols,
        'required_cols_checked': current_required_cols,
        'check_ohlcv': check_ohlcv,
        'min_rows': min_rows,
        'max_rows': max_rows,
        'detect_anomalies_level': detect_anomalies_level,
        'ohlc_validation_stats': ohlc_stats,
        'anomaly_detection_stats': anomaly_stats
    }
    if warnings_log:
        log_metadata['warnings'] = warnings_log
    if not df.empty:
        log_metadata['data_types'] = {str(col): str(df[col].dtype) for col in df.columns}
        if null_counts_series is not None:
            log_metadata['null_counts'] = {str(k): v for k, v in null_counts_series.to_dict().items()}


    if errors:
        logger.warning(f"DataFrame validation failed. Metadata: {log_metadata}. Errors: {errors}. Error Details: {error_details if error_details else 'N/A'}")
        return DataFrameValidationResult(
            is_valid=False, 
            errors=errors, 
            validated_data=df.copy(), 
            error_details=error_details if error_details else None
        )
    else:
        logger.info(f"DataFrame validation successful. Metadata: {log_metadata}")
        return DataFrameValidationResult(
            is_valid=True, 
            errors=None, 
            validated_data=df.copy(),
            error_details=None
        )
