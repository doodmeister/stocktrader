"""ModelTrainer Module

Handles feature engineering, model training, evaluation and persistence of ML models
for stock market prediction using a standardized pipeline approach with robust
validation, error handling, and performance optimizations.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import joblib
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
from functools import partial

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
            raise ValueError(f"Unsupported model type: {value}")

@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline"""
    PRICE_FEATURES: List[str] = field(
        default_factory=lambda: ['Open', 'High', 'Low', 'Close', 'Volume']
    )
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [5, 10, 20])
    TARGET_HORIZON: int = 1
    MIN_SAMPLES: int = 100
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if not self.PRICE_FEATURES:
            raise ValidationError("Must specify at least one price feature")
        if not self.ROLLING_WINDOWS:
            raise ValidationError("Must specify at least one rolling window")
        if self.TARGET_HORIZON < 1:
            raise ValidationError("Target horizon must be positive")
        if self.MIN_SAMPLES < 50:
            raise ValidationError("Minimum samples must be at least 50")

@dataclass
class TrainingParams:
    """Model training hyperparameters"""
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 10
    model_type: ModelType = ModelType.RANDOM_FOREST
    cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = -1
    
    def validate(self) -> None:
        """Validate training parameters"""
        if self.n_estimators < 1:
            raise ValidationError("n_estimators must be positive")
        if self.max_depth < 1:
            raise ValidationError("max_depth must be positive")
        if self.min_samples_split < 2:
            raise ValidationError("min_samples_split must be at least 2")
        if self.cv_folds < 2:
            raise ValidationError("cv_folds must be at least 2")

class ModelTrainer:
    """Handles end-to-end model training pipeline including feature engineering,
    validation, training, evaluation and persistence."""

    def __init__(
        self, 
        config: Any,
        feature_config: Optional[FeatureConfig] = None,
        training_params: Optional[TrainingParams] = None
    ):
        self.model_dir = Path(config.MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_config = feature_config or FeatureConfig()
        self.feature_config.validate()
        
        self.training_params = training_params or TrainingParams()
        self.training_params.validate()
        
        self.scaler = StandardScaler()
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Configure rotating file logger with proper formatting"""
        self.logger = logging.getLogger(__name__)
        
        if not self.logger.handlers:
            handler = RotatingFileHandler(
                self.model_dir / 'model_training.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate input data meets requirements for training
        
        Args:
            df: Input DataFrame containing price/volume data
            
        Raises:
            ValidationError: If data validation fails
        """
        try:
            required_columns = set(self.feature_config.PRICE_FEATURES)
            missing = required_columns - set(df.columns)
            if missing:
                raise ValidationError(f"Missing required columns: {missing}")

            if len(df) < self.feature_config.MIN_SAMPLES:
                raise ValidationError(
                    f"Need at least {self.feature_config.MIN_SAMPLES} samples, "
                    f"got {len(df)}"
                )

            null_cols = df[list(required_columns)].columns[
                df[list(required_columns)].isnull().any()
            ]
            if not null_cols.empty:
                raise ValidationError(f"Null values in columns: {list(null_cols)}")

        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for model training using parallel processing
        
        Args:
            df: Input DataFrame with validated price/volume data
            
        Returns:
            DataFrame with engineered features
        """
        self.validate_input_data(df)
        df = df.copy()
        
        try:
            # Calculate returns
            df['returns'] = df['Close'].pct_change()
            
            # Parallel feature calculation
            with ThreadPoolExecutor() as executor:
                # Calculate rolling features in parallel
                future_to_window = {
                    executor.submit(
                        self._calc_rolling_features, 
                        df, 
                        window
                    ): window 
                    for window in self.feature_config.ROLLING_WINDOWS
                }
                
                # Collect results
                for future in future_to_window:
                    window = future_to_window[future]
                    features = future.result()
                    for col, values in features.items():
                        df[f'{col}_{window}'] = values
                        
            # Calculate target
            df['target'] = df['returns'].shift(
                -self.feature_config.TARGET_HORIZON
            ).apply(lambda x: 1 if x > 0 else 0)
            
            return df.dropna()
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise ModelError(f"Feature engineering failed: {str(e)}")

    def _calc_rolling_features(
        self, 
        df: pd.DataFrame, 
        window: int
    ) -> Dict[str, pd.Series]:
        """Calculate rolling window features
        
        Args:
            df: Input DataFrame
            window: Rolling window size
            
        Returns:
            Dict mapping feature names to Series
        """
        return {
            'volatility': df['returns'].rolling(window=window).std(),
            'sma': df['Close'].rolling(window=window).mean(),
            'volume_sma': df['Volume'].rolling(window=window).mean()
        }

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names"""
        base = list(self.feature_config.PRICE_FEATURES)
        derived = ['returns'] + [
            f'{feat}_{w}'
            for w in self.feature_config.ROLLING_WINDOWS
            for feat in ('volatility', 'sma', 'volume_sma')
        ]
        return base + derived

    def train_model(
        self,
        df: pd.DataFrame,
        params: Optional[TrainingParams] = None
    ) -> Tuple[RandomForestClassifier, ModelMetrics, np.ndarray, str]:
        """Train model with cross-validation and final fit
        
        Args:
            df: Input DataFrame
            params: Optional training parameters override
            
        Returns:
            Tuple containing:
            - Trained model
            - Performance metrics
            - Confusion matrix
            - Classification report
        """
        params = params or self.training_params
        params.validate()
        
        try:
            # Feature engineering
            df = self.feature_engineering(df)
            features = self.get_feature_columns()
            
            # Prepare data
            X = self.scaler.fit_transform(df[features])
            y = df['target'].values

            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=params.cv_folds)
            train_metrics = []
            test_metrics = []
            
            model = RandomForestClassifier(
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
                min_samples_split=params.min_samples_split,
                random_state=params.random_state,
                n_jobs=params.n_jobs
            )

            # Perform cross-validation
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model.fit(X_train, y_train)
                
                # Calculate metrics
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_metrics.append(self._calculate_metrics(y_train, train_pred))
                test_metrics.append(self._calculate_metrics(y_test, test_pred))

            # Final training on full dataset
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Calculate final metrics
            metrics = {
                'train_metrics': self._aggregate_metrics(train_metrics),
                'test_metrics': self._aggregate_metrics(test_metrics),
                'final_metrics': self._calculate_metrics(y, y_pred)
            }
            
            cm = confusion_matrix(y, y_pred)
            report = classification_report(y, y_pred)

            self.logger.info(f"Training complete. Metrics: {metrics}")
            return model, metrics, cm, report
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise ModelError(f"Training failed: {str(e)}")

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
        """Aggregate metrics across CV folds
        
        Args:
            metric_list: List of metric dictionaries
            
        Returns:
            Dict containing mean and std of each metric
        """
        metrics = {}
        for metric in metric_list[0].keys():
            values = [m[metric] for m in metric_list]
            metrics[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        return metrics

    def save_model(
        self, 
        model: RandomForestClassifier,
        symbol: str, 
        interval: str
    ) -> Path:
        """Save trained model and artifacts
        
        Args:
            model: Trained model instance
            symbol: Stock symbol
            interval: Time interval
            
        Returns:
            Path to saved model file
        """
        try:
            # Validate inputs
            if not isinstance(model, RandomForestClassifier):
                raise ValidationError("Invalid model type")
            if not symbol or not interval:
                raise ValidationError("Symbol and interval required")
                
            # Create model artifacts
            artifacts: ModelArtifacts = {
                'model': model,
                'scaler': self.scaler,
                'features': self.get_feature_columns(),
                'metadata': {
                    'symbol': symbol,
                    'interval': interval,
                    'timestamp': datetime.now().isoformat(),
                    'feature_config': self.feature_config.__dict__,
                    'training_params': self.training_params.__dict__
                }
            }
            
            # Save to disk
            model_path = self.model_dir / f"{symbol}_{interval}_model.pkl"
            joblib.dump(artifacts, model_path)
            
            # Save readable metadata
            metadata_path = model_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(artifacts['metadata'], f, indent=2)
                
            self.logger.info(f"Saved model artifacts to {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise ModelError(f"Failed to save model: {str(e)}")
