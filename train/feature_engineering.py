"""
Production-Grade Feature Engineering Module

Provides comprehensive feature engineering capabilities for stock market data,
including technical indicators, candlestick patterns, and statistical features.
Optimized for performance with vectorized operations, caching, and robust error handling.

Features:
    - Rolling statistical features with configurable windows
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Candlestick pattern detection with TA-Lib optimization
    - Price momentum and volatility features
    - Memory-efficient processing for large datasets
    - Comprehensive validation and error handling
    - Performance monitoring and caching

Security Notes:
    - Input validation prevents injection attacks
    - Memory limits prevent resource exhaustion
    - Error handling prevents information leakage

Example:
    ```python
    from train.feature_engineering import FeatureEngineer
    from train.ml_config import MLConfig
    
    config = MLConfig()
    engineer = FeatureEngineer(config)
    
    # Engineer features from OHLCV data
    features_df = engineer.engineer_features(stock_data)
    ```
"""

import functools
import gc
import time
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd

# Optional technical analysis libraries
try:
    import talib
    HAS_TALIB = True
except ImportError:
    talib = None
    HAS_TALIB = False
    warnings.warn("TA-Lib not available. Falling back to pure Python implementations.")

try:
    import ta
    HAS_TA = True
except ImportError:
    ta = None
    HAS_TA = False
    warnings.warn("TA library not available. Technical indicators will be limited.")

from patterns.pattern_utils import get_pattern_names, get_pattern_method
from utils.logger import setup_logger

# Centralized validation imports
from core.data_validator import (
    validate_dataframe
)
# Import new indicator functions
from core.indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands

logger = setup_logger(__name__)


# =============================================================================
# EXCEPTION CLASSES
# =============================================================================

class FeatureEngineeringError(Exception):
    """Base exception for feature engineering operations."""
    pass


class DataValidationError(FeatureEngineeringError):
    """Exception raised for invalid input data."""
    pass


class FeatureGenerationError(FeatureEngineeringError):
    """Exception raised during feature generation."""
    pass


class ConfigurationError(FeatureEngineeringError):
    """Exception raised for invalid configuration."""
    pass


class MemoryError(FeatureEngineeringError):
    """Exception raised when memory limits are exceeded."""
    pass


# =============================================================================
# CONFIGURATION VALIDATOR
# =============================================================================

class FeatureConfigValidator:
    """Validates feature engineering configuration parameters."""
    
    @staticmethod
    def validate_config(config) -> None:
        """
        Validate feature engineering configuration.
        
        Args:
            config: Configuration object to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Validate rolling windows
            if not hasattr(config, 'ROLLING_WINDOWS') or not config.ROLLING_WINDOWS:
                raise ConfigurationError("ROLLING_WINDOWS must be defined and non-empty")
            
            if not all(isinstance(w, int) and w > 0 for w in config.ROLLING_WINDOWS):
                raise ConfigurationError("All rolling windows must be positive integers")
            
            if max(config.ROLLING_WINDOWS) > 1000:
                raise ConfigurationError("Rolling windows should not exceed 1000 periods")
            
            # Validate target horizon
            if not hasattr(config, 'TARGET_HORIZON') or config.TARGET_HORIZON < 1:
                raise ConfigurationError("TARGET_HORIZON must be >= 1")
            
            # Validate optional features
            if hasattr(config, 'use_technical_indicators') and config.use_technical_indicators:
                if not HAS_TA:
                    logger.warning("Technical indicators requested but TA library not available")
            
            if hasattr(config, 'use_candlestick_patterns') and config.use_candlestick_patterns:
                if not HAS_TALIB:
                    logger.warning("Candlestick patterns requested but TA-Lib not available")
            
            logger.debug("Feature engineering configuration validated successfully")
            
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")


# =============================================================================
# MAIN FEATURE ENGINEER CLASS
# =============================================================================


class FeatureEngineer:
    """
    Production-grade feature engineering for stock market data.
    
    This class provides comprehensive feature engineering capabilities with
    robust error handling, performance monitoring, and memory management.
    
    Features Generated:
        - Rolling statistical features (SMA, STD, momentum, etc.)
        - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
        - Candlestick pattern detection
        - Price-volume relationships
        - Volatility and momentum indicators
    
    Attributes:
        config: Feature engineering configuration
        max_memory_mb: Maximum memory usage limit (default: 1GB)
        enable_progress: Whether to show progress for long operations
        cache_size_limit: Maximum number of cached pattern results
    
    Example:
        ```python
        engineer = FeatureEngineer(config)
        features = engineer.engineer_features(df)
        feature_names = engineer.get_feature_columns(['close', 'volume'])
        ```
    """
    
    def __init__(
        self, 
        feature_config,
        max_memory_mb: int = 1024,
        enable_progress: bool = True,
        cache_size_limit: int = 1000
    ):
        """
        Initialize feature engineer with configuration and limits.
        
        Args:
            feature_config: Configuration object with feature parameters
            max_memory_mb: Maximum memory usage in MB
            enable_progress: Enable progress tracking for long operations
            cache_size_limit: Maximum number of cached pattern results
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate configuration
        FeatureConfigValidator.validate_config(feature_config)
        
        self.config = feature_config
        self.max_memory_mb = max_memory_mb
        self.enable_progress = enable_progress
        self.cache_size_limit = cache_size_limit
        
        # Initialize caches and tracking
        self._pattern_cache: Dict[Tuple[str, int, float], pd.Series] = {}
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._performance_stats: Dict[str, float] = {}
        
        # Validate dependencies
        self._validate_dependencies()
        
        logger.info(f"FeatureEngineer initialized with {len(self.config.ROLLING_WINDOWS)} rolling windows")
        logger.debug(f"Memory limit: {max_memory_mb}MB, Cache limit: {cache_size_limit}")

    def _validate_dependencies(self) -> None:
        """Validate optional dependencies and log availability."""
        dependencies = {
            'TA-Lib': HAS_TALIB,
            'TA': HAS_TA
        }
        
        for name, available in dependencies.items():
            if available:
                logger.debug(f"{name} library available")
            else:
                logger.warning(f"{name} library not available - some features may be limited")

    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame for feature engineering using centralized validation
        with feature engineering specific checks.
        
        Args:
            df: Input DataFrame to validate
            
        Raises:
            DataValidationError: If data is invalid
        """
        logger.debug("Starting feature engineering validation")
        
        # First run centralized validation
        try:
            validation_result = validate_dataframe(df)
            if not validation_result.is_valid:
                error_summary = "; ".join(validation_result.errors) if validation_result.errors else "Core validation failed without specific error messages."
                logger.error(f"Core validation failed: {error_summary}")
                raise DataValidationError(f"Core validation failed: {error_summary}")
            else:
                logger.debug("Core validation passed")
        except Exception as e:
            logger.error(f"Error during core validation: {e}")
            raise DataValidationError(f"Validation error: {e}")
        
        # Feature engineering specific validation
        logger.debug("Running feature engineering specific validation")
        
        # Check required columns for feature engineering
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise DataValidationError(f"Missing required columns for feature engineering: {missing_columns}")
        
        # Check for valid OHLC relationships (feature engineering specific)
        invalid_ohlc = df[
            (df['high'] < df[['open', 'low', 'close']].max(axis=1)) |
            (df['low'] > df[['open', 'high', 'close']].min(axis=1))
        ]
        
        if not invalid_ohlc.empty:
            error_count = len(invalid_ohlc)
            if error_count > len(df) * 0.05:  # More than 5% invalid
                raise DataValidationError(
                    f"Too many rows ({error_count}) with invalid OHLC relationships"
                )
            else:
                logger.warning(f"Found {error_count} rows with invalid OHLC relationships")
        
        # Check for reasonable value ranges (feature engineering specific)
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if (df[col] <= 0).any():
                zero_count = (df[col] <= 0).sum()
                logger.warning(f"Found {zero_count} non-positive values in '{col}'")
                
        # Check minimum data requirements with adaptive filtering
        available_rows = len(df)
        min_absolute = min(self.config.ROLLING_WINDOWS) + self.config.TARGET_HORIZON
        
        if available_rows < min_absolute:
            raise DataValidationError(
                f"Insufficient data: {available_rows} rows available, "
                f"minimum {min_absolute} required for smallest rolling window "
                f"({min(self.config.ROLLING_WINDOWS)}) + target horizon ({self.config.TARGET_HORIZON})"
            )
        
        # Filter rolling windows to those that can be computed with available data
        max_usable_window = available_rows - self.config.TARGET_HORIZON
        usable_windows = [w for w in self.config.ROLLING_WINDOWS if w <= max_usable_window]
        
        if len(usable_windows) < len(self.config.ROLLING_WINDOWS):
            excluded_windows = [w for w in self.config.ROLLING_WINDOWS if w > max_usable_window]
            logger.warning(
                f"Limited data ({available_rows} rows): excluding rolling windows {excluded_windows}. "
                f"Using windows: {usable_windows}. For full features, provide at least "
                f"{max(self.config.ROLLING_WINDOWS) + self.config.TARGET_HORIZON} rows."
            )
            
            # Temporarily update config for this run
            self._original_windows = self.config.ROLLING_WINDOWS
            self.config.ROLLING_WINDOWS = usable_windows
        else:
            self._original_windows = None
            
        # Memory usage check
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > self.max_memory_mb * 0.5:  # Use 50% threshold for input
            logger.warning(f"Large input data: {memory_mb:.1f}MB (limit: {self.max_memory_mb}MB)")
            
        logger.debug("Feature engineering validation completed successfully")

    def _check_memory_usage(self, df: pd.DataFrame, operation: str) -> None:
        """Check memory usage and clean up if needed."""
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        if memory_mb > self.max_memory_mb:
            logger.warning(f"High memory usage in {operation}: {memory_mb:.1f}MB")
            # Clear caches if memory is high
            self._clear_caches()
            gc.collect()
            
        self._performance_stats[f'{operation}_memory_mb'] = memory_mb
    
    def _clear_caches(self) -> None:
        """Clear all caches to free memory."""
        cache_count = len(self._pattern_cache) + len(self._feature_cache)
        self._pattern_cache.clear()
        self._feature_cache.clear()
        logger.debug(f"Cleared {cache_count} cached entries")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive feature engineering to OHLCV data.
        
        This method applies all configured feature engineering operations
        with robust error handling and performance monitoring.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            
        Returns:
            DataFrame with original data plus engineered features
            
        Raises:
            DataValidationError: If input data is invalid
            FeatureGenerationError: If feature generation fails
            MemoryError: If memory limits are exceeded
            
        Example:
            ```python
            # Input DataFrame with OHLCV data
            ohlcv_data = pd.DataFrame({...})
            
            # Generate features
            features = engineer.engineer_features(ohlcv_data)
            
            # Result includes original data + new features
            assert len(features.columns) > len(ohlcv_data.columns)
            ```
        """
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_input_data(df)
            logger.info(f"Starting feature engineering on {len(df)} rows, {len(df.columns)} columns")
            
            # Create working copy
            result_df = df.copy()
            original_columns = len(result_df.columns)
            
            # Check initial memory usage
            self._check_memory_usage(result_df, "initial")
            
            # Add rolling statistical features
            logger.debug("Adding rolling statistical features...")
            result_df = self._add_rolling_features(result_df)
            self._check_memory_usage(result_df, "rolling_features")
            
            # Add technical indicators if enabled and available
            if getattr(self.config, 'use_technical_indicators', False):
                # The HAS_TA check is now inside _add_technical_indicators
                logger.debug("Adding technical indicators...")
                result_df = self._add_technical_indicators(result_df)
                self._check_memory_usage(result_df, "technical_indicators")
            
            # Add candlestick patterns if enabled
            if getattr(self.config, 'use_candlestick_patterns', False):
                logger.debug("Adding candlestick patterns...")
                result_df = self._add_candlestick_patterns(result_df)
                self._check_memory_usage(result_df, "candlestick_patterns")
            
            # Final validation and cleanup
            final_columns = len(result_df.columns)
            features_added = final_columns - original_columns
            
            # Remove rows with insufficient data for rolling features
            min_valid_rows = max(self.config.ROLLING_WINDOWS)
            if len(result_df) > min_valid_rows:
                result_df = result_df.iloc[min_valid_rows:].copy()
              # Log performance statistics
            execution_time = time.time() - start_time
            self._performance_stats['total_execution_time'] = execution_time
            self._performance_stats['features_added'] = features_added
            self._performance_stats['final_rows'] = len(result_df)
            
            logger.info(
                f"Feature engineering completed: {features_added} features added, "
                f"{len(result_df)} final rows, {execution_time:.2f}s"
            )
            
            # Restore original configuration if it was modified
            if hasattr(self, '_original_windows') and self._original_windows is not None:
                self.config.ROLLING_WINDOWS = self._original_windows
                delattr(self, '_original_windows')
            
            return result_df
            
        except Exception as e:
            # Restore original configuration even on error
            if hasattr(self, '_original_windows') and self._original_windows is not None:
                self.config.ROLLING_WINDOWS = self._original_windows
                delattr(self, '_original_windows')
                
            logger.error(f"Feature engineering failed: {e}")
            if isinstance(e, (DataValidationError, FeatureGenerationError, MemoryError)):
                raise
            else:
                raise FeatureGenerationError(f"Unexpected error during feature engineering: {e}")
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling statistical features with optimized computation.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with added rolling features
            
        Raises:
            FeatureGenerationError: If rolling feature generation fails
        """
        try:
            result_df = df.copy()
            
            # Filter valid windows based on data length
            valid_windows = [
                w for w in self.config.ROLLING_WINDOWS
                if w < len(df) - self.config.TARGET_HORIZON
            ]
            
            if not valid_windows:
                logger.warning("No valid rolling windows for current data length")
                return result_df
            
            logger.debug(f"Computing rolling features for windows: {valid_windows}")
            
            for window in valid_windows:
                try:
                    # Pre-compute rolling objects for efficiency
                    close_rolling = df['close'].rolling(window, min_periods=max(1, window//2))
                    volume_rolling = df['volume'].rolling(window, min_periods=max(1, window//2))
                    high_rolling = df['high'].rolling(window, min_periods=max(1, window//2))
                    low_rolling = df['low'].rolling(window, min_periods=max(1, window//2))
                    
                    # Basic price features
                    result_df[f'close_sma_{window}'] = close_rolling.mean()
                    result_df[f'close_std_{window}'] = close_rolling.std()
                    result_df[f'close_min_{window}'] = close_rolling.min()
                    result_df[f'close_max_{window}'] = close_rolling.max()
                    
                    # Price change features
                    result_df[f'close_pct_change_{window}'] = df['close'].pct_change(window)
                    result_df[f'close_log_return_{window}'] = np.log(df['close'] / df['close'].shift(window))
                    
                    # Volume features
                    result_df[f'volume_sma_{window}'] = volume_rolling.mean()
                    result_df[f'volume_std_{window}'] = volume_rolling.std()
                    result_df[f'volume_pct_change_{window}'] = df['volume'].pct_change(window)
                    
                    # High-Low range features
                    result_df[f'hl_range_{window}'] = (df['high'] - df['low']).rolling(window).mean()
                    result_df[f'hl_pct_{window}'] = ((df['high'] - df['low']) / df['close']).rolling(window).mean()
                    
                    # Price-volume correlation
                    result_df[f'price_volume_corr_{window}'] = (
                        df['close'].rolling(window).corr(df['volume'])
                    )
                    
                    # Momentum and trend features
                    if window > 1:
                        # Simple momentum
                        result_df[f'momentum_{window}'] = (
                            df['close'] / df['close'].shift(window) - 1
                        )
                        
                        # Rate of change
                        result_df[f'roc_{window}'] = (
                            (df['close'] - df['close'].shift(window)) / df['close'].shift(window) * 100
                        )
                        
                        # EMA-based features
                        ema_fast = df['close'].ewm(span=max(2, window//2), adjust=False).mean()
                        ema_slow = df['close'].ewm(span=window, adjust=False).mean()
                        result_df[f'ema_diff_{window}'] = ema_fast - ema_slow
                        result_df[f'ema_ratio_{window}'] = ema_fast / ema_slow
                        
                        # Volatility features
                        result_df[f'volatility_{window}'] = (
                            df['close'].pct_change().rolling(window).std() * np.sqrt(252)
                        )
                        
                        # Support/Resistance levels
                        result_df[f'support_{window}'] = low_rolling.min()
                        result_df[f'resistance_{window}'] = high_rolling.max()
                        result_df[f'price_position_{window}'] = (
                            (df['close'] - low_rolling.min()) / 
                            (high_rolling.max() - low_rolling.min())
                        )
                    
                except Exception as e:
                    logger.warning(f"Failed to compute rolling features for window {window}: {e}")
                    continue
            
            logger.debug(f"Successfully added rolling features for {len(valid_windows)} windows")
            return result_df
            
        except Exception as e:
            raise FeatureGenerationError(f"Rolling features generation failed: {e}")
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators with comprehensive error handling.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
            
        Raises:
            FeatureGenerationError: If technical indicator generation fails
        """
        # REMOVED: TA library check, as we are using our own implementations for core indicators
        
        try:
            result_df = df.copy()
            
            # Extract price and volume data
            # Ensure 'close', 'high', 'low', 'volume' are present, validated by _validate_input_data
            
            # Validate minimum data requirements
            # This might need adjustment based on the requirements of our custom indicators
            min_periods_rsi = 14 + 1  # RSI typically needs period + 1 for initial calculation
            min_periods_macd = 26 + 9 # MACD (26,12,9) needs longest period + signal period
            min_periods_bb = 20       # Bollinger Bands typically 20 periods

            min_periods = max(min_periods_rsi, min_periods_macd, min_periods_bb)

            if len(df) < min_periods:
                logger.warning(f"Insufficient data for core technical indicators: {len(df)} < {min_periods}")
                # Still attempt to calculate other TA-lib indicators if HAS_TA
            
            indicators_added = 0
            
            try:
                # RSI (Relative Strength Index) using core.indicators
                # Corrected parameter name from 'period' to 'length'
                rsi_14_series = calculate_rsi(result_df, length=14)
                if rsi_14_series is not None:
                    result_df['rsi_14'] = rsi_14_series
                    indicators_added += 1
                
                # Corrected parameter name from 'period' to 'length'
                rsi_21_series = calculate_rsi(result_df, length=21)
                if rsi_21_series is not None:
                    result_df['rsi_21'] = rsi_21_series
                    indicators_added += 1

            except Exception as e:
                logger.warning(f"Failed to compute RSI using core.indicators: {e}")
            
            try:
                # MACD (Moving Average Convergence Divergence) using core.indicators
                # Corrected parameter names
                macd_output = calculate_macd(result_df, fast=12, slow=26, signal=9)
                # calculate_macd now returns a tuple: (macd_line, signal_line, histogram)
                if macd_output is not None:
                    macd_line, signal_line, hist_line = macd_output
                    result_df['macd'] = macd_line
                    result_df['macd_signal'] = signal_line
                    result_df['macd_histogram'] = hist_line
                    indicators_added += 3
            except Exception as e:
                logger.warning(f"Failed to compute MACD using core.indicators: {e}")
            
            try:
                # Bollinger Bands using core.indicators
                # Corrected parameter names from 'period' to 'length' and 'std_dev' to 'std'
                bb_output = calculate_bollinger_bands(result_df, length=20, std=2)
                # calculate_bollinger_bands now returns a tuple: (upper_band, middle_band, lower_band)
                if bb_output is not None:
                    upper_band, middle_band, lower_band = bb_output
                    result_df['bb_upper'] = upper_band
                    result_df['bb_middle'] = middle_band
                    result_df['bb_lower'] = lower_band
                    indicators_added += 3 
                    
                    # Calculate BB width and position if needed
                    if middle_band is not None and upper_band is not None and lower_band is not None:
                        # Ensure no division by zero or NaN in middle band for width calculation
                        safe_middle_band = middle_band.replace(0, np.nan) # Avoid division by zero
                        result_df['bb_width'] = (upper_band - lower_band) / safe_middle_band
                        
                        # Ensure no division by zero or NaN in (upper_band - lower_band) for position
                        band_range = upper_band - lower_band
                        safe_band_range = band_range.replace(0, np.nan)
                        result_df['bb_position'] = (result_df['close'] - lower_band) / safe_band_range
                        indicators_added += 2 # for width and position
            except Exception as e:
                logger.warning(f"Failed to compute Bollinger Bands using core.indicators: {e}")

            # Keep other TA-Lib indicators if HAS_TA (pandas_ta)
            if HAS_TA and ta is not None: # 'ta' here refers to 'pandas_ta'
                logger.debug("Adding other technical indicators using pandas_ta...")
                # Use df.ta strategy for pandas_ta
                # Ensure result_df has a DatetimeIndex if required by pandas_ta, or necessary columns
                # It's safer to apply ta calculations to a copy or ensure columns are not overwritten if originals are needed.
                
                # Example for Stochastic using pandas_ta
                try:
                    if len(df) >= 14: # Common period for Stochastic
                        stoch = result_df.ta.stoch(k=14, d=3, smooth_k=3)
                        if stoch is not None and not stoch.empty:
                            result_df['stoch_k'] = stoch.iloc[:,0] # STOCHk_14_3_3
                            result_df['stoch_d'] = stoch.iloc[:,1] # STOCHd_14_3_3
                            indicators_added += 2
                    else:
                        logger.warning("Skipping Stochastic Oscillator (pandas_ta) due to insufficient data.")
                except Exception as e:
                    logger.warning(f"Failed to compute Stochastic with pandas_ta: {e}")

                # Example for ADX using pandas_ta
                try:
                    if len(df) >= 28: # ADX typically needs 2x period
                        adx_df = result_df.ta.adx(length=14)
                        if adx_df is not None and not adx_df.empty:
                            result_df['adx'] = adx_df.iloc[:,0]    # ADX_14
                            result_df['adx_pos'] = adx_df.iloc[:,1] # DMP_14
                            result_df['adx_neg'] = adx_df.iloc[:,2] # DMN_14
                            indicators_added += 3
                    else:
                        logger.warning("Skipping ADX (pandas_ta) due to insufficient data.")
                except Exception as e:
                    logger.warning(f"Failed to compute ADX with pandas_ta: {e}")

                # Example for ATR using pandas_ta
                try:
                    if len(df) >= 14:
                        atr_series = result_df.ta.atr(length=14)
                        if atr_series is not None:
                            result_df['atr'] = atr_series # ATR_14
                            indicators_added += 1
                    else:
                        logger.warning("Skipping ATR (pandas_ta) due to insufficient data.")
                except Exception as e:
                    logger.warning(f"Failed to compute ATR with pandas_ta: {e}")

                # Example for Williams %R using pandas_ta
                try:
                    if len(df) >= 14:
                        willr_series = result_df.ta.willr(length=14)
                        if willr_series is not None:
                            result_df['williams_r'] = willr_series # WILLR_14
                            indicators_added += 1
                    else:
                        logger.warning("Skipping Williams %R (pandas_ta) due to insufficient data.")
                except Exception as e:
                    logger.warning(f"Failed to compute Williams %R with pandas_ta: {e}")

                # Example for CCI using pandas_ta
                try:
                    if len(df) >= 20:
                        cci_series = result_df.ta.cci(length=20)
                        if cci_series is not None:
                            result_df['cci'] = cci_series # CCI_20_0.015
                            indicators_added += 1
                    else:
                        logger.warning("Skipping CCI (pandas_ta) due to insufficient data.")
                except Exception as e:
                    logger.warning(f"Failed to compute CCI with pandas_ta: {e}")
                
                # Example for OBV using pandas_ta
                try:
                    if len(df) >= 1:
                        obv_series = result_df.ta.obv()
                        if obv_series is not None:
                            result_df['obv'] = obv_series
                            indicators_added +=1
                except Exception as e:
                    logger.warning(f"Failed to compute OBV with pandas_ta: {e}")

                # Example for VWAP using pandas_ta (VWAP is typically daily, or needs a reset condition)
                # pandas_ta.vwap might not take a window in the same way.
                # It usually calculates VWAP for the entire series or resets based on a 'anchor' (e.g., 'D' for daily)
                try:
                    if 'high' in result_df and 'low' in result_df and 'close' in result_df and 'volume' in result_df:
                        # For a rolling VWAP-like feature, you might need a custom rolling apply or use a different approach.
                        # pandas_ta.vwap is typically for a fixed period (e.g. daily VWAP)
                        # If a windowed VWAP is needed, it's often calculated as (typical_price * volume).sum() / volume.sum() over the window.
                        # For now, let's assume we want the standard pandas_ta VWAP (which might be for the whole period or anchored)
                        vwap_series = result_df.ta.vwap() # This will compute VWAP over the available data, might need anchoring for meaningful results in a rolling context.
                        if vwap_series is not None:
                             result_df['vwap_pandas_ta'] = vwap_series 
                             indicators_added += 1
                    else:
                        logger.warning("Skipping VWAP (pandas_ta) as HLCV columns are not available.")
                except Exception as e:
                    logger.warning(f"Failed to compute VWAP with pandas_ta: {e}")

                # Example for MFI using pandas_ta
                try:
                    if len(df) >= 14:
                        mfi_series = result_df.ta.mfi(length=14)
                        if mfi_series is not None:
                            result_df['mfi'] = mfi_series # MFI_14
                            indicators_added += 1
                    else:
                        logger.warning("Skipping MFI (pandas_ta) due to insufficient data.")
                except Exception as e:
                    logger.warning(f"Failed to compute MFI with pandas_ta: {e}")

                # Example for ROC using pandas_ta
                try:
                    if len(df) >= 12: # common ROC period
                        roc_series = result_df.ta.roc(length=12)
                        if roc_series is not None:
                            result_df['roc'] = roc_series # ROC_12
                            indicators_added += 1
                    else:
                        logger.warning("Skipping ROC (pandas_ta) due to insufficient data.")
                except Exception as e:
                    logger.warning(f"Failed to compute ROC with pandas_ta: {e}")

                # Trend indicators (SMAs, EMAs)
                # SMAs are often covered by rolling features. EMAs are part of MACD.
                # Add specific ones if not covered.
                try:
                    if len(df) >= 50 and 'close_sma_50' not in result_df.columns:
                        sma50 = result_df.ta.sma(length=50)
                        if sma50 is not None: result_df['sma_50_pta'] = sma50; indicators_added += 1
                    if len(df) >= 200 and 'close_sma_200' not in result_df.columns:
                        sma200 = result_df.ta.sma(length=200)
                        if sma200 is not None: result_df['sma_200_pta'] = sma200; indicators_added += 1
                except Exception as e:
                    logger.warning(f"Failed to compute additional SMAs with pandas_ta: {e}")
            
            logger.debug(f"Successfully added {indicators_added} technical indicators")
            return result_df
            
        except Exception as e:
            raise FeatureGenerationError(f"Technical indicators generation failed: {e}")
    
    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features with vectorized detection."""
        result_df = df.copy()
        
        try:
            # Get pattern names to use
            pattern_names = (
                self.config.selected_patterns or 
                get_pattern_names()
            )
            
            # Use TA-Lib for faster pattern detection where available
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
              # Initialize talib patterns dictionary only if TA-Lib is available
            talib_patterns = {}
            if HAS_TALIB and talib is not None:
                talib_patterns = {
                    'hammer': talib.CDLHAMMER,
                    'hanging_man': talib.CDLHANGINGMAN,
                    'doji': talib.CDLDOJI,
                    'shooting_star': talib.CDLSHOOTINGSTAR,
                    'engulfing_bullish': talib.CDLENGULFING,
                    'morning_star': talib.CDLMORNINGSTAR,
                    'evening_star': talib.CDLEVENINGSTAR,
                    'harami': talib.CDLHARAMI,
                    'piercing': talib.CDLPIERCING,
                    'dark_cloud': talib.CDLDARKCLOUDCOVER
                }
            
            for pattern_name in pattern_names:
                pattern_key = pattern_name.lower().replace(' ', '_')
                
                if HAS_TALIB and pattern_key in talib_patterns:
                    # Use TA-Lib for supported patterns
                    pattern_result = talib_patterns[pattern_key](
                        open_prices, high_prices, low_prices, close_prices
                    )
                    # Convert to binary (TA-Lib returns -100, 0, 100)
                    result_df[pattern_name.replace(" ", "")] = (pattern_result != 0).astype(int)
                else:
                    # Fall back to custom pattern detection
                    result_df = self._add_custom_pattern(result_df, pattern_name)
            
        except Exception as e:
            logger.warning(f"Error adding candlestick patterns: {e}")
        
        return result_df
    
    def _add_custom_pattern(self, df: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
        """Add custom pattern using pattern_utils."""
        try:
            method = get_pattern_method(pattern_name)
            if method is None:
                return df
            
            # Use caching for repeated pattern detection
            cache_key = (pattern_name, len(df), df['close'].iloc[-1] if len(df) > 0 else 0)
            
            if cache_key in self._pattern_cache:
                pattern_series = self._pattern_cache[cache_key]
            else:
                # Vectorized pattern detection
                pattern_series = self._vectorized_pattern_detection(df, method)
                self._pattern_cache[cache_key] = pattern_series
            
            df[pattern_name.replace(" ", "")] = pattern_series
            
        except Exception as e:
            logger.debug(f"Failed to add pattern {pattern_name}: {e}")
        
        return df
    
    def _vectorized_pattern_detection(self, df: pd.DataFrame, method) -> pd.Series:
        """Vectorized pattern detection for better performance."""
        results = np.zeros(len(df), dtype=int)
        
        # Determine minimum window size
        min_rows = 3  # Default minimum
        
        # Process in chunks for memory efficiency
        chunk_size = 1000
        for start_idx in range(min_rows - 1, len(df), chunk_size):
            end_idx = min(start_idx + chunk_size, len(df))
            
            for i in range(start_idx, end_idx):
                window_start = max(0, i + 1 - min_rows)
                window = df.iloc[window_start:i + 1]
                
                try:
                    detected = int(method(window)) if method else 0
                    results[i] = detected
                except Exception:
                    results[i] = 0
        
        return pd.Series(results, index=df.index)
    
    @functools.lru_cache(maxsize=128)
    def get_feature_columns(self, base_columns: List[str]) -> List[str]:
        """Get list of all feature columns that would be generated."""
        feature_cols = []
        
        # Rolling features
        for window in self.config.ROLLING_WINDOWS:
            features = [
                f'close_sma_{window}',
                f'close_std_{window}',
                f'close_pct_change_{window}',
                f'volume_sma_{window}',
                f'volume_std_{window}',
                f'volume_pct_change_{window}',
                f'price_volume_corr_{window}',
                f'hl_range_{window}',
                f'momentum_{window}',
                f'ema_diff_{window}'
            ]
            feature_cols.extend(features)
        
        # Technical indicators
        if self.config.use_technical_indicators:
            technical_features = [
                'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
                'stoch_k', 'stoch_d', 'adx', 'atr', 'williams_r', 'cci', 'obv'
            ]
            feature_cols.extend(technical_features)
        
        # Candlestick patterns
        if self.config.use_candlestick_patterns:
            pattern_names = self.config.selected_patterns or get_pattern_names()
            pattern_features = [pattern.replace(" ", "") for pattern in pattern_names]
            feature_cols.extend(pattern_features)
        
        return feature_cols