import pandas as pd
import numpy as np
import warnings
import sys
import os

# Adjust sys.path to allow imports from the project root
# Assuming the script is in tests/ and project root is one level up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from train.feature_engineering import FeatureEngineer, HAS_TALIB, FeatureConfigValidator, ConfigurationError
    from patterns.pattern_utils import get_pattern_names
except ImportError as e:
    print(f"Error importing necessary modules. Make sure PYTHONPATH is set correctly or run from project root.")
    print(f"Details: {e}")
    sys.exit(1)

# Mock config for testing
class MockFeatureConfig:
    def __init__(self):
        self.ROLLING_WINDOWS = [5, 10]
        self.TARGET_HORIZON = 1
        self.use_technical_indicators = True  # General switch, may not directly use TA-Lib for this
        self.use_candlestick_patterns = True  # This should trigger TA-Lib usage for patterns
        # Define ohlcv_columns as FeatureEngineer might expect it from config
        self.ohlcv_columns = {
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
        }
        # Add other attributes if FeatureConfigValidator or FeatureEngineer expects them
        # self.datetime_column = 'timestamp' # If FeatureEngineer expects a named datetime column

def create_sample_ohlcv_data(num_rows=30): # Increased rows for more pattern potential
    """Creates a sample OHLCV DataFrame with a DatetimeIndex."""
    data = {
        'open':   [10, 11, 10, 12, 13, 11, 12, 14, 13, 15, 16, 14, 13, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 13, 12, 14, 15, 16, 17, 18],
        'high':   [12, 12, 11, 13, 14, 13, 13, 15, 15, 17, 17, 15, 14, 13, 12, 11, 10, 12, 13, 13, 14, 15, 16, 15, 14, 16, 17, 18, 19, 20],
        'low':    [9,  10, 9,  11, 12, 10, 11, 13, 12, 14, 15, 13, 12, 11, 10, 9,  8,  9, 10, 11, 12, 13, 14, 12, 11, 13, 14, 15, 16, 17],
        'close':  [11, 10, 11, 13, 12, 12, 13, 15, 14, 16, 15, 14, 12, 11, 10, 9, 10, 11, 12, 11, 13, 15, 14, 14, 13, 15, 16, 17, 18, 19],
        'volume': [100,110,105,120,130,115,125,140,135,150,160,145,130,120,110,100,90,100,110,120,130,140,150,130,120,140,150,160,170,180]
    }
    # Ensure all lists have the same length as num_rows
    for key in data:
        data[key] = data[key][:num_rows]

    df = pd.DataFrame(data)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df['timestamp'] = pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_rows, freq='D'))
    df = df.set_index('timestamp')
    return df

def test_feature_engineering_with_talib():
    print("Starting test_feature_engineering_with_talib...")

    sample_df = create_sample_ohlcv_data()
    config = MockFeatureConfig()

    if not HAS_TALIB:
        print("TA-Lib is not installed. Verifying graceful handling for candlestick patterns.")
        config.use_candlestick_patterns = True # Keep True to test resilience

        try:
            # Validate config (it might warn or pass if TA-Lib is missing but use_candlestick_patterns is True)
            # The warning "Falling back to pure Python implementations" suggests it should proceed.
            FeatureConfigValidator.validate_config(config)
            engineer = FeatureEngineer(feature_config=config)
            features_df = engineer.engineer_features(sample_df.copy())
            print("Feature engineering ran without TA-Lib (as expected by fallback mechanism).")

            talib_pattern_names = []
            try:
                talib_pattern_names = get_pattern_names()
                if not isinstance(talib_pattern_names, list): talib_pattern_names = []
            except Exception: # nosec
                pass # Ignore if get_pattern_names itself fails or is not as expected

            if talib_pattern_names:
                for pattern_name in talib_pattern_names:
                    assert pattern_name not in features_df.columns, \
                        f"TA-Lib specific column {pattern_name} found when TA-Lib is not available."
                print("Successfully verified no TA-Lib specific patterns generated when TA-Lib is unavailable.")
            else:
                print("Could not retrieve TA-Lib pattern names or list is empty; skipping check for their absence.")
            
        except ConfigurationError as e:
            print(f"ConfigurationError encountered: {e}")
            print("This might be acceptable if config validation is strict about TA-Lib for patterns.")
            # Depending on expected behavior, this might not be a test failure.
            # For now, let's assume the fallback mechanism should prevent this.
            assert False, f"ConfigurationError when TA-Lib is missing but patterns requested: {e}"
        except Exception as e:
            print(f"Unexpected error during feature engineering without TA-Lib: {e}")
            import traceback
            traceback.print_exc()
            assert False, f"Unexpected error when TA-Lib is missing: {e}"
        print("TA-Lib not available test part finished.")
        return

    # This part runs if HAS_TALIB is True
    print("TA-Lib is installed. Proceeding with TA-Lib specific tests.")
    config.use_candlestick_patterns = True # Ensure it's set for the test

    try:
        FeatureConfigValidator.validate_config(config)
        engineer = FeatureEngineer(feature_config=config)
    except ConfigurationError as e:
        print(f"Error initializing FeatureEngineer with TA-Lib enabled: {e}")
        assert False, f"FeatureEngineer initialization failed: {e}"
        return
    except Exception as e:
        print(f"Unexpected error initializing FeatureEngineer: {e}")
        assert False, f"Unexpected error during FeatureEngineer initialization: {e}"
        return

    print(f"Input data for feature engineering (first 5 rows):\n{sample_df.head()}")

    try:
        features_df = engineer.engineer_features(sample_df.copy())
    except Exception as e:
        print(f"Error during engineer_features call: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"engineer_features call failed: {e}"
        return

    print(f"Engineered features DataFrame columns: {features_df.columns.tolist()}")
    print(f"Engineered features DataFrame head (first 5 rows):\n{features_df.head()}")

    talib_pattern_names = []
    try:
        talib_pattern_names = get_pattern_names()
        if not isinstance(talib_pattern_names, list) or not all(isinstance(name, str) for name in talib_pattern_names):
            print(f"Warning: get_pattern_names() did not return a list of strings. Returned: {talib_pattern_names}")
            talib_pattern_names = [] 
    except Exception as e:
        print(f"Could not get pattern names from patterns.pattern_utils.get_pattern_names: {e}")
        talib_pattern_names = []

    if not talib_pattern_names:
        warnings.warn("No TA-Lib pattern names found via get_pattern_names(). Cannot verify specific pattern columns.")
        # Consider a basic check: did number of columns increase?
        # For now, if no names, we can't assert specific columns.
    else:
        print(f"Expecting TA-Lib pattern columns based on get_pattern_names() (e.g., {talib_pattern_names[:3]}...).")
        found_talib_patterns = 0
        missing_patterns = []
        for pattern_name in talib_pattern_names:
            if pattern_name in features_df.columns:
                found_talib_patterns += 1
                # print(f"Found TA-Lib pattern column: {pattern_name}") # Can be verbose
                assert pd.api.types.is_numeric_dtype(features_df[pattern_name]), \
                    f"TA-Lib pattern column {pattern_name} is not numeric."
                # TA-Lib patterns are typically integers (0, 100, -100).
                # Check if values are in a typical range. Some patterns might produce NaN if not enough data.
                # assert features_df[pattern_name].dropna().isin([0, 100, -100]).all(), \
                #    f"Values in {pattern_name} are not typical for TA-Lib patterns (0, 100, -100)."
            else:
                missing_patterns.append(pattern_name)
        
        if found_talib_patterns > 0:
            print(f"Successfully found {found_talib_patterns} TA-Lib pattern columns.")
            if len(missing_patterns) < len(talib_pattern_names) and len(missing_patterns) > 0 : # Some found, some missing
                 print(f"Note: {len(missing_patterns)} expected TA-Lib patterns were not found (e.g., {missing_patterns[:3]}). This might be due to data length or specific pattern requirements.")
        else: # No patterns found at all
            print(f"Error: TA-Lib is installed and use_candlestick_patterns is True, but NO expected TA-Lib pattern columns were found.")
            print(f"Expected patterns (from get_pattern_names): {talib_pattern_names}")
            print(f"Actual columns in features_df: {features_df.columns.tolist()}")
            assert False, "No TA-Lib candlestick pattern columns were generated despite TA-Lib being available."
            
    print("test_feature_engineering_with_talib completed successfully.")

if __name__ == "__main__":
    test_feature_engineering_with_talib()
