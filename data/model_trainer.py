"""ModelTrainer Module

Handles feature engineering, model training, evaluation and persistence of ML models
for stock market prediction using a standardized pipeline approach with robust
validation, error handling, and performance optimizations.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Configure logger
logger = logging.getLogger(__name__)

# Type aliases
ModelMetrics = Dict[str, float]
ModelArtifacts = Dict[str, Any]

class ModelError(Exception):
    """Base exception for model-related errors"""
    pass

class ValidationError(ModelError):
    """Raised when data validation fails"""
    pass

class ModelType(Enum):
    """Supported model types"""
    RANDOM_FOREST = "random_forest"
    
    @classmethod
    def from_str(cls, value: str) -> 'ModelType':
        try:
            return cls(value.lower())
        except ValueError:
            valid_values = [t.value for t in cls]
            raise ValueError(f"Invalid model type '{value}'. Valid options: {valid_values}")

@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline"""
    PRICE_FEATURES: List[str] = field(
        default_factory=lambda: ['open', 'high', 'low', 'close', 'volume']
    )
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [5, 10, 20])
    TARGET_HORIZON: int = 1


@dataclass
class TrainingParams:
    """Parameters for model training"""
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5


class ModelTrainer:
    """Handles end-to-end model training pipeline including feature engineering,
    validation, training, evaluation and persistence."""

    def __init__(
        self, 
        config: Any,
        feature_config: Optional[FeatureConfig] = None,
        training_params: Optional[TrainingParams] = None
    ):
        """Initialize the model trainer with configuration."""
        self.config = config
        self.feature_config = feature_config or FeatureConfig()
        self.default_params = training_params or TrainingParams()
        self._setup_logger()
        
        # Create model directory if it doesn't exist
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        
        logger.info(f"ModelTrainer initialized with model directory: {self.config.MODEL_DIR}")

    def _setup_logger(self) -> None:
        """Set up logger for the model trainer."""
        log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = Path('logs/model_trainer.log')
        log_file.parent.mkdir(exist_ok=True)
        
        # Set up handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=3
        )
        file_handler.setFormatter(log_formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

    def validate_input_data(self, df: pd.DataFrame) -> None:
        """
        Validate input data format and content.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValidationError: If validation fails
        """
        # Check if DataFrame is empty
        if df.empty:
            raise ValidationError("Input DataFrame is empty")
            
        # Check required columns
        required_columns = [col.lower() for col in self.feature_config.PRICE_FEATURES]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
            
        # Check index type
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValidationError("DataFrame index must be DatetimeIndex")
            
        # Check for NaN values
        if df.isna().any().any():
            logger.warning(f"DataFrame contains NaN values. Rows with NaN: {df.isna().any(axis=1).sum()}")
            
        # Check row count
        min_rows = max(self.feature_config.ROLLING_WINDOWS) + self.feature_config.TARGET_HORIZON + 10
        if len(df) < min_rows:
            raise ValidationError(
                f"Not enough data points. Got {len(df)}, need at least {min_rows} for feature engineering"
            )

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for ML model training.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        # Start with a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure column names are lowercase
        result_df.columns = [c.lower() for c in result_df.columns]
        
        # For each window, calculate rolling features
        for window in self.feature_config.ROLLING_WINDOWS:
            # Add window-specific features
            window_features = self._calc_rolling_features(result_df, window)
            
            # Add to result DataFrame with window suffix
            for name, series in window_features.items():
                result_df[f"{name}_{window}"] = series
                
        # Add target variable - price movement direction
        result_df['target'] = (
            result_df['close'].shift(-self.feature_config.TARGET_HORIZON) > result_df['close']
        ).astype(int)
        
        # Drop NaN values created by rolling windows and shifts
        result_df = result_df.dropna()
        
        return result_df

    def _calc_rolling_features(
        self, 
        df: pd.DataFrame, 
        window: int
    ) -> Dict[str, pd.Series]:
        """
        Calculate rolling window features.
        
        Args:
            df: DataFrame with price data
            window: Rolling window size
            
        Returns:
            Dictionary of feature name to Series
        """
        features = {}
        
        # Price momentum and volatility
        features['close_pct_change'] = df['close'].pct_change(window)
        features['close_std'] = df['close'].rolling(window).std()
        
        # Volume features
        features['volume_pct_change'] = df['volume'].pct_change(window)
        features['volume_std'] = df['volume'].rolling(window).std()
        
        # Price-volume relationship
        features['price_volume_corr'] = df['close'].rolling(window).corr(df['volume'])
        
        # Technical indicators
        # Simple Moving Average
        features['sma'] = df['close'].rolling(window).mean()
        
        # Moving Average Convergence Divergence (basic)
        if window > 1:  # Only calculate for windows > 1
            ema_fast = df['close'].ewm(span=window//2, adjust=False).mean()
            ema_slow = df['close'].ewm(span=window, adjust=False).mean()
            features['macd'] = ema_fast - ema_slow
        
        return features

    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature column names.
        
        Returns:
            List of feature column names
        """
        feature_cols = []
        
        for window in self.feature_config.ROLLING_WINDOWS:
            for feat in [
                'close_pct_change', 'close_std', 'volume_pct_change',
                'volume_std', 'price_volume_corr', 'sma', 'macd'
            ]:
                # Only add MACD for windows > 1
                if feat == 'macd' and window <= 1:
                    continue
                feature_cols.append(f"{feat}_{window}")
                
        return feature_cols

    def train_model(
        self,
        df: pd.DataFrame,
        params: Optional[TrainingParams] = None
    ) -> Tuple[RandomForestClassifier, Dict[str, Dict[str, float]], np.ndarray, str]:
        """
        Train a model on the input data.
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Training parameters (or use default if None)
            
        Returns:
            Tuple of (trained_model, metrics_dict, confusion_matrix_array, classification_report_str)
        """
        logger.info(f"Starting model training with data shape: {df.shape}")
        params = params or self.default_params
        
        try:
            # 1. Validate input data
            self.validate_input_data(df)
            
            # 2. Feature engineering
            features_df = self.feature_engineering(df)
            logger.debug(f"Feature-engineered data shape: {features_df.shape}")
            
            # 3. Split into features and target
            feature_cols = self.get_feature_columns()
            X = features_df[feature_cols]
            y = features_df['target']
            
            # 4. Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 5. Initialize TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=params.cv_folds)
            
            # 6. Initialize model
            model = RandomForestClassifier(
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
                min_samples_split=params.min_samples_split,
                random_state=params.random_state,
                n_jobs=params.n_jobs
            )
            
            # 7. Cross-validation
            cv_metrics = []
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                fold_metrics = self._calculate_metrics(y_test, y_pred)
                cv_metrics.append(fold_metrics)
                
            # 8. Train final model on all data
            model.fit(X_scaled, y)
            y_pred_final = model.predict(X_scaled)
            
            # 9. Calculate final metrics
            final_metrics = self._calculate_metrics(y, y_pred_final)
            cm = confusion_matrix(y, y_pred_final)
            report = classification_report(y, y_pred_final)
            
            # 10. Aggregate and return results
            metrics_dict = self._aggregate_metrics(cv_metrics)
            metrics_dict['final_metrics'] = final_metrics
            
            logger.info(f"Model training completed. Final accuracy: {final_metrics['accuracy']:.4f}")
            
            return model, metrics_dict, cm, report
            
        except Exception as e:
            logger.exception(f"Error during model training: {e}")
            raise ModelError(f"Model training failed: {str(e)}") from e

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _aggregate_metrics(
        self, 
        metric_list: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics from cross-validation.
        
        Args:
            metric_list: List of metric dictionaries from each fold
            
        Returns:
            Dictionary with mean and std of each metric
        """
        result = {'mean': {}, 'std': {}}
        
        # Get all metric keys
        all_keys = set()
        for metrics in metric_list:
            all_keys.update(metrics.keys())
            
        # Calculate mean and std for each metric
        for key in all_keys:
            values = [m.get(key, 0) for m in metric_list]
            result['mean'][key] = np.mean(values)
            result['std'][key] = np.std(values)
            
        return result

    def save_model(
        self, 
        model: RandomForestClassifier,
        symbol: str, 
        interval: str
    ) -> Path:
        """
        Save the trained model to disk.
        
        Args:
            model: Trained model to save
            symbol: Stock symbol
            interval: Time interval (e.g., '1d', '1h')
            
        Returns:
            Path to the saved model file
        """
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{interval}_{timestamp}.pkl"
            filepath = self.config.MODEL_DIR / filename
            
            # Save model using pickle
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
                
            logger.info(f"Model saved successfully to {filepath}")
            return filepath
            
        except Exception as e:
            logger.exception(f"Error saving model: {e}")
            raise ModelError(f"Failed to save model: {str(e)}") from e
