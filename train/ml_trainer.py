"""
Enhanced ModelTrainer Module

A production-grade ML training pipeline for stock market prediction that combines
technical analysis with candlestick pattern recognition. Features robust validation,
error handling, performance optimizations, and comprehensive logging.

Key Features:
- Vectorized feature engineering for optimal performance
- Time-series aware cross-validation
- Comprehensive input validation and error handling
- Model versioning and metadata tracking
- Memory-efficient processing for large datasets
- Configurable feature selection and hyperparameters
"""

import functools
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train.feature_engineering import FeatureEngineer
from train.model_manager import ModelManager, ModelMetadata
from utils.logger import setup_logger

# Import centralized data validation system
from core.data_validator import (
    validate_dataframe
)

# Suppress sklearn warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = setup_logger(__name__)


class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class ValidationError(ModelError):
    """Exception raised for data validation errors."""
    pass


class TrainingError(ModelError):
    """Exception raised during model training."""
    pass


class ModelType(Enum):
    """Supported model types for training."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline.
    
    Attributes:
        PRICE_FEATURES: Base OHLCV columns required for analysis
        ROLLING_WINDOWS: Window sizes for rolling statistics
        TARGET_HORIZON: Periods ahead for target prediction
        use_candlestick_patterns: Whether to include pattern features
        selected_patterns: Specific patterns to use (None = all available)
        use_technical_indicators: Whether to include technical indicators
        max_features: Maximum number of features to select
        feature_selection_method: Method for feature selection
    """
    PRICE_FEATURES: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume'
    ])
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    TARGET_HORIZON: int = 1
    use_candlestick_patterns: bool = True
    selected_patterns: Optional[List[str]] = None
    use_technical_indicators: bool = True
    max_features: Optional[int] = None
    feature_selection_method: str = "importance"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.TARGET_HORIZON < 1:
            raise ValueError("TARGET_HORIZON must be >= 1")
        if not self.ROLLING_WINDOWS or any(w < 2 for w in self.ROLLING_WINDOWS):
            raise ValueError("ROLLING_WINDOWS must contain values >= 2")
        if self.max_features is not None and self.max_features < 1:
            raise ValueError("max_features must be >= 1")


@dataclass
class TrainingParams:
    """Hyperparameters for model training.
    
    Attributes:
        model_type: Type of model to train
        n_estimators: Number of trees for ensemble methods
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples required to split
        min_samples_leaf: Minimum samples required at leaf node
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores)
        cv_folds: Number of cross-validation folds
        test_size: Fraction of data for final validation
        early_stopping: Whether to use early stopping
        validation_split: Fraction for validation during training
    """
    model_type: ModelType = ModelType.RANDOM_FOREST
    n_estimators: int = 100
    max_depth: Optional[int] = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5
    test_size: float = 0.2
    early_stopping: bool = False
    validation_split: float = 0.1
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
        if not 0.1 <= self.test_size <= 0.5:
            raise ValueError("test_size must be between 0.1 and 0.5")
        if not 0.05 <= self.validation_split <= 0.3:
            raise ValueError("validation_split must be between 0.05 and 0.3")


class ModelTrainer:
    """
    Production-grade ML trainer for stock market prediction.
    
    Combines technical analysis, candlestick patterns, and machine learning
    for robust signal generation. Features comprehensive validation,
    performance optimization, and model management capabilities.
    
    Example:
        ```python
        config = load_config()
        trainer = ModelTrainer(config)
        
        # Train model with custom parameters
        pipeline, metrics, cm, report = trainer.train_model(
            df=stock_data,
            params=TrainingParams(n_estimators=200, max_depth=15)
        )
        
        # Save with version tracking
        model_path = trainer.save_model_with_manager(
            pipeline, "AAPL", "1d", metrics
        )
        ```
    """
    
    def __init__(
        self,
        config: Any,
        feature_config: Optional[FeatureConfig] = None,
        training_params: Optional[TrainingParams] = None,
        max_workers: Optional[int] = None
    ):
        """
        Initialize ModelTrainer with configuration.
        
        Args:
            config: Application configuration object
            feature_config: Feature engineering configuration
            training_params: Default training parameters
            max_workers: Maximum worker threads for parallel processing
        """
        self.config = config
        self.feature_config = feature_config or FeatureConfig()
        self.default_params = training_params or TrainingParams()
        self.max_workers = max_workers or min(4, (os.cpu_count() or 1))
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(self.feature_config)
        
        # Model directory setup
        self.model_dir = Path(getattr(config, 'MODEL_DIR', 'models'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self._training_start_time = None
        self._memory_usage = {}
        
        logger.info(f"ModelTrainer initialized with {self.max_workers} workers")
    
    def validate_input_data(self, df: pd.DataFrame) -> None:
        """
        Comprehensive validation of input DataFrame using the centralized validation system.
        
        Args:
            df: Input DataFrame to validate
            
        Raises:
            ValidationError: If data doesn't meet requirements
        """
        if df.empty:
            raise ValidationError("Input DataFrame is empty")
        
        # Use centralized data validation system
        required_columns = [col.lower() for col in self.feature_config.PRICE_FEATURES]
        
        try:
            logger.info("Starting data validation using core validator")
            
            # Use the comprehensive DataValidator from core module
            validation_result = validate_dataframe(
                df, 
                required_cols=required_columns,
                validate_ohlc=True,
                check_statistical_anomalies=True
            )
            
            # Check validation results
            if not validation_result.is_valid:
                error_message = "; ".join(validation_result.errors)
                logger.error(f"Core validation failed: {error_message}")
                raise ValidationError(f"Data validation failed: {error_message}")
            
            # Display warnings if any
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.warning(f"Data warning: {warning}")
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            logger.exception("Data validation error")
            raise ValidationError(f"Validation error: {str(e)}")
        
        # ML-specific validation checks not covered by core validator
        
        # Check for sufficient data based on ML requirements
        min_required_rows = (
            max(self.feature_config.ROLLING_WINDOWS) + 
            self.feature_config.TARGET_HORIZON + 
            self.default_params.cv_folds * 10  # Minimum samples per fold
        )
        
        if len(df) < min_required_rows:
            raise ValidationError(
                f"Insufficient data for ML training: {len(df)} rows provided, "
                f"minimum {min_required_rows} required (based on rolling windows, target horizon, and CV folds)"
            )
        
        # Validate index for time series requirements
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
                logger.info("Converted index to DatetimeIndex for time series analysis")
            except Exception as e:
                raise ValidationError(f"Cannot convert index to DatetimeIndex for time series analysis: {e}")
        
        # ML-specific data quality checks
        close_col = next((c for c in df.columns if c.lower() == 'close'), 'close')
        if close_col in df.columns:
            if df[close_col].isna().all():
                raise ValidationError("Close price column contains only NaN values")
            
            if (df[close_col] <= 0).any():
                logger.warning("Found non-positive close prices - will be filtered during feature engineering")
        
        # Memory usage check for ML processing
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        if memory_mb > 1000:  # 1GB threshold
            logger.warning(f"Large dataset detected: {memory_mb:.1f}MB - consider chunked processing")
            
        # Data quality metrics for ML
        nan_pct = (df.isna().sum() / len(df) * 100).max()
        if nan_pct > 10:
            logger.warning(f"High missing data percentage: {nan_pct:.1f}% - may impact model performance")
        
        logger.info(
            f"ML data validation passed: {len(df)} rows, {len(df.columns)} columns, "
            f"{memory_mb:.1f}MB memory usage, ready for feature engineering"
        )

    @functools.lru_cache(maxsize=128)
    def _get_cached_feature_columns(self, column_tuple: tuple) -> List[str]:
        """Cache feature column computation for performance."""
        return self.feature_engineer.get_feature_columns(list(column_tuple))

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive feature engineering with performance optimization.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features and target variable
            
        Raises:
            ValidationError: If feature engineering fails
        """
        try:
            logger.info("Starting feature engineering...")
            start_time = datetime.now()
            
            # Normalize column names
            result_df = df.copy()
            result_df.columns = [c.lower() for c in result_df.columns]
            
            # Remove invalid data
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            result_df = result_df[result_df[numeric_cols] > 0].copy()
            
            if len(result_df) < max(self.feature_config.ROLLING_WINDOWS) * 2:
                raise ValidationError("Insufficient valid data after cleaning")
            
            # Delegate to FeatureEngineer for consistency
            result_df = self.feature_engineer.engineer_features(result_df)
            
            # Create target variable
            close_col = 'close'
            if close_col not in result_df.columns:
                raise ValidationError("Missing 'close' column for target creation")
            
            # Forward-looking target (buy signal)
            future_returns = (
                result_df[close_col].shift(-self.feature_config.TARGET_HORIZON) / 
                result_df[close_col] - 1
            )
            
            # Binary classification: positive return = 1, negative = 0
            result_df['target'] = (future_returns > 0).astype(int)
            result_df['future_return'] = future_returns  # Keep for analysis
            
            # Remove rows with missing target
            result_df = result_df.dropna(subset=['target'])
            
            # Feature selection if configured
            if self.feature_config.max_features:
                result_df = self._select_top_features(result_df)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Feature engineering completed in {processing_time:.2f}s. "
                f"Final shape: {result_df.shape}"
            )
            
            return result_df
            
        except Exception as e:
            logger.exception("Feature engineering failed")
            raise ValidationError(f"Feature engineering failed: {str(e)}") from e

    def _select_top_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select top features based on importance or correlation."""
        try:
            feature_cols = [col for col in df.columns if col not in ['target', 'future_return']]
            
            if len(feature_cols) <= self.feature_config.max_features:
                return df
            
            if self.feature_config.feature_selection_method == "correlation":
                # Select features with highest correlation to target
                correlations = df[feature_cols].corrwith(df['target']).abs()
                top_features = correlations.nlargest(self.feature_config.max_features).index.tolist()
            else:
                # Use random forest feature importance
                rf = RandomForestClassifier(
                    n_estimators=50, 
                    random_state=self.default_params.random_state,
                    n_jobs=1  # Limit for feature selection
                )
                
                X_sample = df[feature_cols].fillna(0)
                y_sample = df['target']
                
                rf.fit(X_sample, y_sample)
                importances = pd.Series(rf.feature_importances_, index=feature_cols)
                top_features = importances.nlargest(self.feature_config.max_features).index.tolist()
            
            selected_cols = top_features + ['target', 'future_return']
            logger.info(f"Selected {len(top_features)} top features from {len(feature_cols)}")
            
            return df[selected_cols]
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")
            return df

    def _create_model_pipeline(self, params: TrainingParams) -> Pipeline:
        """Create ML pipeline based on model type."""
        if params.model_type == ModelType.RANDOM_FOREST:
            model = RandomForestClassifier(
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
                min_samples_split=params.min_samples_split,
                min_samples_leaf=params.min_samples_leaf,
                random_state=params.random_state,
                n_jobs=params.n_jobs,
                class_weight='balanced'  # Handle class imbalance
            )
        else:
            raise ValueError(f"Unsupported model type: {params.model_type}")
        
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model)
        ])

    def train_model(
        self,
        df: pd.DataFrame,
        params: Optional[TrainingParams] = None
    ) -> Tuple[Pipeline, Dict[str, Any], np.ndarray, str]:
        """
        Train ML model with comprehensive validation and monitoring.
        
        Args:
            df: Training data DataFrame
            params: Training parameters (uses defaults if None)
            
        Returns:
            Tuple of (trained_pipeline, metrics_dict, confusion_matrix, classification_report)
            
        Raises:
            TrainingError: If training fails
        """
        self._training_start_time = datetime.now()
        params = params or self.default_params
        
        try:
            logger.info(f"Starting model training with {len(df)} samples")
            
            # Validate and prepare data
            self.validate_input_data(df)
            features_df = self.feature_engineering(df)
            
            # Get feature columns
            feature_cols = [col for col in features_df.columns 
                          if col not in ['target', 'future_return']]
            
            if not feature_cols:
                raise ValidationError("No feature columns available after engineering")
            
            X = features_df[feature_cols].fillna(0)  # Handle any remaining NaNs
            y = features_df['target']
            
            # Check class balance
            class_counts = y.value_counts()
            minority_class_pct = class_counts.min() / len(y) * 100
            
            if minority_class_pct < 5:
                logger.warning(
                    f"Severe class imbalance: minority class {minority_class_pct:.1f}%"
                )
            
            logger.info(f"Class distribution: {dict(class_counts)}")
            
            # Create and train pipeline
            pipeline = self._create_model_pipeline(params)
            
            # Time series cross-validation
            cv_metrics = self._cross_validate_model(X, y, pipeline, params)
            
            # Final training on full dataset
            logger.info("Training final model on full dataset...")
            pipeline.fit(X, y)
            
            # Final predictions and metrics
            y_pred = pipeline.predict(X)
            y_proba = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, 'predict_proba') else None
            
            final_metrics = self._calculate_comprehensive_metrics(y, y_pred, y_proba)
            cm = confusion_matrix(y, y_pred)
            report = classification_report(y, y_pred, zero_division=0)
            
            # Aggregate all metrics
            metrics_dict = {
                'cv_metrics': cv_metrics,
                'final_metrics': final_metrics,
                'training_info': {
                    'training_samples': len(X),
                    'n_features': len(feature_cols),
                    'feature_names': feature_cols,
                    'class_distribution': dict(class_counts),
                    'training_time_seconds': (datetime.now() - self._training_start_time).total_seconds()
                }
            }
            
            logger.info(
                f"Training completed successfully. "
                f"Final accuracy: {final_metrics['accuracy']:.4f}, "
                f"AUC: {final_metrics.get('auc', 'N/A')}"
            )
            
            return pipeline, metrics_dict, cm, report
            
        except Exception as e:
            logger.exception("Model training failed")
            raise TrainingError(f"Training failed: {str(e)}") from e

    def _cross_validate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline: Pipeline,
        params: TrainingParams
    ) -> Dict[str, Dict[str, float]]:
        """Perform time series cross-validation with comprehensive metrics."""
        logger.info(f"Starting {params.cv_folds}-fold time series cross-validation...")
        
        # Adjust CV folds if necessary
        max_folds = min(params.cv_folds, len(X) // 50)  # At least 50 samples per fold
        if max_folds < params.cv_folds:
            logger.warning(
                f"Reducing CV folds from {params.cv_folds} to {max_folds} "
                f"due to insufficient data"
            )
        
        tscv = TimeSeriesSplit(n_splits=max(2, max_folds))
        cv_metrics = []
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max_folds)) as executor:
            future_to_fold = {}
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                future = executor.submit(
                    self._train_and_evaluate_fold,
                    X.iloc[train_idx].copy(),
                    y.iloc[train_idx].copy(),
                    X.iloc[test_idx].copy(),
                    y.iloc[test_idx].copy(),
                    pipeline,
                    fold
                )
                future_to_fold[future] = fold
            
            for future in as_completed(future_to_fold):
                fold = future_to_fold[future]
                try:
                    fold_metrics = future.result()
                    cv_metrics.append(fold_metrics)
                    logger.debug(f"Fold {fold} completed with accuracy: {fold_metrics['accuracy']:.4f}")
                except Exception as e:
                    logger.warning(f"Fold {fold} failed: {e}")
        
        if not cv_metrics:
            raise TrainingError("All CV folds failed")
        
        return self._aggregate_cv_metrics(cv_metrics)

    def _train_and_evaluate_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        pipeline: Pipeline,
        fold: int
    ) -> Dict[str, float]:
        """Train and evaluate a single CV fold."""
        try:
            # Create fresh pipeline for this fold
            fold_pipeline = self._create_model_pipeline(self.default_params)
            fold_pipeline.fit(X_train, y_train)
            
            y_pred = fold_pipeline.predict(X_test)
            y_proba = (fold_pipeline.predict_proba(X_test)[:, 1] 
                      if hasattr(fold_pipeline, 'predict_proba') else None)
            
            return self._calculate_comprehensive_metrics(y_test, y_pred, y_proba)
            
        except Exception as e:
            logger.warning(f"Error in fold {fold}: {e}")
            raise

    def _calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Add AUC if probabilities available
        if y_proba is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except Exception as e:
                logger.debug(f"Could not calculate AUC: {e}")
        
        return metrics

    def _aggregate_cv_metrics(self, cv_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate cross-validation metrics with statistics."""
        result = {'mean': {}, 'std': {}, 'min': {}, 'max': {}}
        
        all_keys = set()
        for metrics in cv_metrics:
            all_keys.update(metrics.keys())
        
        for key in all_keys:
            values = [m.get(key, 0) for m in cv_metrics]
            result['mean'][key] = np.mean(values)
            result['std'][key] = np.std(values)
            result['min'][key] = np.min(values)
            result['max'][key] = np.max(values)
        
        return result

    def save_model_with_manager(
        self,
        pipeline: Pipeline,
        symbol: str,
        interval: str,
        metrics: Optional[Dict] = None,
        backend: str = "Classic ML (RandomForest)"
    ) -> str:
        """
        Save model using ModelManager with comprehensive metadata.
        
        Args:
            pipeline: Trained pipeline to save
            symbol: Trading symbol
            interval: Time interval
            metrics: Training metrics
            backend: Model backend identifier
            
        Returns:
            Path to saved model
        """
        try:
            model_manager = ModelManager(base_directory=str(self.model_dir))
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare feature information
            dummy_df = pd.DataFrame(columns=self.feature_config.PRICE_FEATURES)
            expected_features = self.feature_engineer.get_feature_columns(
                dummy_df.columns.tolist()
            )
            
            # Extract accuracy from metrics
            accuracy = None
            if metrics:
                final_metrics = metrics.get('final_metrics', {})
                cv_metrics = metrics.get('cv_metrics', {})
                accuracy = (final_metrics.get('accuracy') or 
                          cv_metrics.get('mean', {}).get('accuracy'))
            
            metadata = ModelMetadata(
                version=version,
                saved_at=datetime.now().isoformat(),
                accuracy=accuracy,
                parameters={
                    "symbol": symbol,
                    "interval": interval,
                    "model_type": self.default_params.model_type.value,
                    "features": expected_features,
                    "feature_config": {
                        "rolling_windows": self.feature_config.ROLLING_WINDOWS,
                        "target_horizon": self.feature_config.TARGET_HORIZON,
                        "use_patterns": self.feature_config.use_candlestick_patterns,
                        "use_indicators": self.feature_config.use_technical_indicators,
                        "selected_patterns": self.feature_config.selected_patterns
                    },
                    "training_params": {
                        "n_estimators": self.default_params.n_estimators,
                        "max_depth": self.default_params.max_depth,
                        "cv_folds": self.default_params.cv_folds
                    },
                    "metrics": metrics
                },
                framework_version="sklearn"
            )
            
            save_path = model_manager.save_model(
                model=pipeline,
                metadata=metadata,
                backend=backend
            )
            
            logger.info(f"Model saved successfully to: {save_path}")
            return save_path
            
        except Exception as e:
            logger.exception("Failed to save model")
            raise ModelError(f"Model save failed: {str(e)}") from e

    def load_model(self, model_path: Union[str, Path]) -> Pipeline:
        """
        Load model with validation.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded pipeline
            
        Raises:
            ModelError: If loading fails
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise ModelError(f"Model file not found: {model_path}")
            
            pipeline = joblib.load(model_path)
            
            # Validate loaded pipeline
            if not hasattr(pipeline, "predict"):
                raise ModelError("Loaded object is not a valid pipeline")
            
            logger.info(f"Model loaded successfully from: {model_path}")
            return pipeline
            
        except Exception as e:
            logger.exception("Failed to load model")
            raise ModelError(f"Model loading failed: {str(e)}") from e

    def predict(self, df: pd.DataFrame, model_path: Union[str, Path]) -> pd.Series:
        """
        Generate predictions using trained model.
        
        Args:
            df: Input DataFrame
            model_path: Path to trained model
            
        Returns:
            Series with predictions
        """
        try:
            pipeline = self.load_model(model_path)
            features_df = self.feature_engineering(df)
            
            feature_cols = [col for col in features_df.columns 
                          if col not in ['target', 'future_return']]
            
            if not feature_cols:
                raise ValidationError("No feature columns found for prediction")
            
            X = features_df[feature_cols].fillna(0)
            predictions = pipeline.predict(X)
            
            return pd.Series(
                predictions,
                index=features_df.index,
                name="model_signal"
            )
            
        except Exception as e:
            logger.exception("Prediction failed")
            raise ModelError(f"Prediction failed: {str(e)}") from e

    def predict_proba(self, df: pd.DataFrame, model_path: Union[str, Path]) -> pd.Series:
        """
        Generate prediction probabilities.
        
        Args:
            df: Input DataFrame
            model_path: Path to trained model
            
        Returns:
            Series with buy probabilities
        """
        try:
            pipeline = self.load_model(model_path)
            
            if not hasattr(pipeline, "predict_proba"):
                raise ModelError("Model does not support probability prediction")
            
            features_df = self.feature_engineering(df)
            feature_cols = [col for col in features_df.columns 
                          if col not in ['target', 'future_return']]
            
            X = features_df[feature_cols].fillna(0)
            probabilities = pipeline.predict_proba(X)[:, 1]
            
            return pd.Series(
                probabilities,
                index=features_df.index,
                name="model_buy_proba"
            )
            
        except Exception as e:
            logger.exception("Probability prediction failed")
            raise ModelError(f"Probability prediction failed: {str(e)}") from e

    def get_model_info(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about a saved model."""
        try:
            model_manager = ModelManager(base_directory=str(self.model_dir))
            return model_manager.get_model_info(str(model_path))
        except Exception as e:
            logger.warning(f"Could not retrieve model info: {e}")
            return {}

    def cleanup_old_models(self, keep_latest: int = 5) -> None:
        """Clean up old model files, keeping only the latest N."""
        try:
            model_files = list(self.model_dir.glob("*.joblib"))
            if len(model_files) <= keep_latest:
                return
            
            # Sort by modification time and remove oldest
            model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for old_model in model_files[keep_latest:]:
                old_model.unlink()
                logger.info(f"Removed old model: {old_model}")
                
        except Exception as e:
            logger.warning(f"Model cleanup failed: {e}")