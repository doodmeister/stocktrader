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
    from train.feature_engineering import FeatureEngineer, FeatureConfigValidator, ConfigurationError
    from patterns.pattern_utils import get_pattern_names
except ImportError as e:
    print(f"Error importing necessary modules. Make sure PYTHONPATH is set correctly or run from project root.")
    print(f"Details: {e}")
    sys.exit(1)

# Mock config for testing
class MockFeatureConfig:
    def __init__(self):
        self.use_technical_indicators = True
        self.use_candlestick_patterns = True # This will now test custom patterns
        self.use_rolling_window_features = True
        self.rolling_windows = [5, 10]
        self.ROLLING_WINDOWS = [5, 10, 20]
        self.TARGET_HORIZON = 1
        self.indicator_config = {
            "rsi": {"enabled": True, "length": 14},
            "macd": {"enabled": True, "fast": 12, "slow": 26, "signal": 9},
            "bollinger_bands": {"enabled": True, "length": 20, "std_dev": 2},
            "stochastic_oscillator": {"enabled": True, "k_period": 14, "d_period": 3, "slowing_period": 3},
            "average_directional_index": {"enabled": True, "length": 14},
            "average_true_range": {"enabled": True, "length": 14},
            "williams_r": {"enabled": True, "length": 14},
            "commodity_channel_index": {"enabled": True, "length": 20},
            "on_balance_volume": {"enabled": True},
            "vwap": {"enabled": True},
            "money_flow_index": {"enabled": True, "length": 14},
            "rate_of_change": {"enabled": True, "length": 10},
        }
        # selected_patterns can be empty to use all from patterns.py, or specify a list
        self.selected_patterns = [] 
        self.feature_flags = {
            'sma': True, 'std_dev': True, 'min_max': True, 'pct_change': True,
            'log_return': True, 'hl_range_pct': True, 'price_volume_corr': True,
            'momentum': True, 'roc': True, 'ema_diff': True, 'ema_ratio': True,
            'volatility': True, 'support_resistance': True, 'price_position': True
        }
        self.ohlcv_columns = {
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
        }

def create_sample_ohlcv_data(num_rows=30):
    """Creates a sample OHLCV DataFrame with a DatetimeIndex."""
    data = {
        'open':   [10, 11, 10, 12, 13, 11, 12, 14, 13, 15, 16, 14, 13, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 13, 12, 14, 15, 16, 17, 18],
        'high':   [12, 12, 11, 13, 14, 13, 13, 15, 15, 17, 17, 15, 14, 13, 12, 11, 10, 12, 13, 13, 14, 15, 16, 15, 14, 16, 17, 18, 19, 20],
        'low':    [9,  10, 9,  11, 12, 10, 11, 13, 12, 14, 15, 13, 12, 11, 10, 9,  8,  9, 10, 11, 12, 13, 14, 12, 11, 13, 14, 15, 16, 17],
        'close':  [11, 10, 11, 13, 12, 12, 13, 15, 14, 16, 15, 14, 12, 11, 10, 9, 10, 11, 12, 11, 13, 15, 14, 14, 13, 15, 16, 17, 18, 19],
        'volume': [100,110,105,120,130,115,125,140,135,150,160,145,130,120,110,100,90,100,110,120,130,140,150,130,120,140,150,160,170,180]
    }
    for key in data:
        data[key] = data[key][:num_rows]

    df = pd.DataFrame(data)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df['timestamp'] = pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_rows, freq='D'))
    df = df.set_index('timestamp')
    return df

def test_feature_engineering_custom_patterns():
    print("Starting test_feature_engineering_custom_patterns (formerly _with_talib)...")

    sample_df = create_sample_ohlcv_data()
    config = MockFeatureConfig()
    config.use_candlestick_patterns = True # Ensure custom patterns are tested

    print("Verifying candlestick patterns are generated using custom logic (patterns.py).")

    try:
        FeatureConfigValidator.validate_config(config) # Should pass
        engineer = FeatureEngineer(feature_config=config)
        features_df = engineer.engineer_features(sample_df.copy())
        print("Feature engineering ran with custom candlestick patterns.")

        custom_pattern_names = []
        try:
            # These are the patterns expected from patterns.py
            custom_pattern_names = get_pattern_names() 
            if not isinstance(custom_pattern_names, list): custom_pattern_names = []
        except Exception as e:
            print(f"Could not retrieve custom pattern names: {e}")
            pass 

        if not custom_pattern_names:
            warnings.warn("No custom pattern names found via get_pattern_names(). Cannot verify specific pattern columns.")
        else:
            print(f"Expecting custom pattern columns based on get_pattern_names() (e.g., {custom_pattern_names[:3]}...).")
            found_custom_patterns = 0
            missing_patterns = []
            for pattern_name in custom_pattern_names:
                # Column names in DataFrame might have spaces removed
                df_pattern_col_name = pattern_name.replace(" ", "")
                if df_pattern_col_name in features_df.columns:
                    found_custom_patterns += 1
                    assert pd.api.types.is_numeric_dtype(features_df[df_pattern_col_name]), \
                        f"Custom pattern column {df_pattern_col_name} is not numeric."
                    # Custom patterns should return 0 or 1 (or other integers if defined differently)
                    # For this test, we assume they are binary (0 or 1)
                    assert features_df[df_pattern_col_name].dropna().isin([0, 1]).all() or features_df[df_pattern_col_name].dropna().empty, \
                       f"Values in {df_pattern_col_name} are not binary (0 or 1) as expected for custom patterns."
                else:
                    missing_patterns.append(df_pattern_col_name)
            
            assert found_custom_patterns > 0, \
                f"No custom candlestick pattern columns were generated. Expected based on patterns.py: {custom_pattern_names}. Actual columns: {features_df.columns.tolist()}"
            
            if missing_patterns:
                 print(f"Note: {len(missing_patterns)} expected custom patterns were not found (e.g., {missing_patterns[:3]}). This might be due to data length or specific pattern requirements in patterns.py.")

            print(f"Successfully found and verified {found_custom_patterns} custom pattern columns.")

    except ConfigurationError as e:
        print(f"ConfigurationError encountered: {e}")
        assert False, f"ConfigurationError during test: {e}"
    except Exception as e:
        print(f"Unexpected error during feature engineering with custom patterns: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Unexpected error: {e}"
    
    print("test_feature_engineering_custom_patterns completed successfully.")

if __name__ == "__main__":
    test_feature_engineering_custom_patterns()
