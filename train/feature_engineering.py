"""
Feature Engineering Module

Provides comprehensive feature engineering capabilities for stock market data,
including technical indicators, candlestick patterns, and statistical features.
Optimized for performance with vectorized operations and caching.
"""

import functools
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import talib
import ta

from patterns.pattern_utils import get_pattern_names, get_pattern_method
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """
    Handles all feature engineering operations for stock market data.
    
    Features include:
    - Rolling statistical features
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Candlestick pattern detection
    - Price momentum and volatility features
    """
    
    def __init__(self, feature_config):
        """Initialize with feature configuration."""
        self.config = feature_config
        self._pattern_cache = {}
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive feature engineering to OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        result_df = df.copy()
        
        # Add rolling statistical features
        result_df = self._add_rolling_features(result_df)
        
        # Add technical indicators if enabled
        if self.config.use_technical_indicators:
            result_df = self._add_technical_indicators(result_df)
        
        # Add candlestick patterns if enabled
        if self.config.use_candlestick_patterns:
            result_df = self._add_candlestick_patterns(result_df)
        
        return result_df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features efficiently."""
        result_df = df.copy()
        
        # Filter valid windows based on data length
        valid_windows = [
            w for w in self.config.ROLLING_WINDOWS
            if w < len(df) - self.config.TARGET_HORIZON
        ]
        
        for window in valid_windows:
            # Price-based features
            close_rolling = df['close'].rolling(window, min_periods=window//2)
            volume_rolling = df['volume'].rolling(window, min_periods=window//2)
            
            # Basic rolling features
            result_df[f'close_sma_{window}'] = close_rolling.mean()
            result_df[f'close_std_{window}'] = close_rolling.std()
            result_df[f'close_pct_change_{window}'] = df['close'].pct_change(window)
            
            # Volume features
            result_df[f'volume_sma_{window}'] = volume_rolling.mean()
            result_df[f'volume_std_{window}'] = volume_rolling.std()
            result_df[f'volume_pct_change_{window}'] = df['volume'].pct_change(window)
            
            # Price-volume correlation
            result_df[f'price_volume_corr_{window}'] = (
                df['close'].rolling(window).corr(df['volume'])
            )
            
            # High-low range features
            result_df[f'hl_range_{window}'] = (
                (df['high'] - df['low']).rolling(window).mean()
            )
            
            # Momentum features
            if window > 1:
                result_df[f'momentum_{window}'] = (
                    df['close'] / df['close'].shift(window) - 1
                )
                
                # EMA-based features
                ema_fast = df['close'].ewm(span=window//2, adjust=False).mean()
                ema_slow = df['close'].ewm(span=window, adjust=False).mean()
                result_df[f'ema_diff_{window}'] = ema_fast - ema_slow
        
        return result_df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using pure Python ta package."""
        result_df = df.copy()
        
        try:
            close_prices = df['close']
            high_prices = df['high']
            low_prices = df['low']
            open_prices = df['open']
            volume = df['volume']
            
            # RSI
            result_df['rsi_14'] = ta.momentum.rsi(close_prices, window=14)
            
            # MACD
            macd_line = ta.trend.macd(close_prices)
            macd_signal = ta.trend.macd_signal(close_prices)
            macd_histogram = ta.trend.macd_diff(close_prices)
            result_df['macd'] = macd_line
            result_df['macd_signal'] = macd_signal
            result_df['macd_histogram'] = macd_histogram
            
            # Bollinger Bands
            result_df['bb_upper'] = ta.volatility.bollinger_hband(close_prices)
            result_df['bb_middle'] = ta.volatility.bollinger_mavg(close_prices)
            result_df['bb_lower'] = ta.volatility.bollinger_lband(close_prices)
            result_df['bb_width'] = ta.volatility.bollinger_wband(close_prices)
            result_df['bb_position'] = ta.volatility.bollinger_pband(close_prices)
            
            # Stochastic
            result_df['stoch_k'] = ta.momentum.stoch(high_prices, low_prices, close_prices)
            result_df['stoch_d'] = ta.momentum.stoch_signal(high_prices, low_prices, close_prices)
            
            # ADX
            result_df['adx'] = ta.trend.adx(high_prices, low_prices, close_prices)
            
            # ATR
            result_df['atr'] = ta.volatility.average_true_range(high_prices, low_prices, close_prices)
            
            # Williams %R
            result_df['williams_r'] = ta.momentum.williams_r(high_prices, low_prices, close_prices)
            
            # CCI
            result_df['cci'] = ta.trend.cci(high_prices, low_prices, close_prices)
            
            # OBV
            result_df['obv'] = ta.volume.on_balance_volume(close_prices, volume)
            
        except Exception as e:
            logger.warning(f"Error adding technical indicators: {e}")
        
        return result_df
    
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
            
            # Common TA-Lib patterns
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
                
                if pattern_key in talib_patterns:
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