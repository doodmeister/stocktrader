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
from functools import lru_cache # ENSURED
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import os
import shutil # ENSURED
import warnings
import joblib
import sklearn # ENSURED

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
from train.model_manager import ModelManager, ModelMetadata, ModelManagerConfig
from train.model_manager import ModelType as ManagerModelType # MODIFIED: Aliased import
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
        # Assuming config is loaded and stock_data is a pandas DataFrame
        # from utils.config import load_config # Example: replace with actual config loading
        # config = load_config() 
        # import pandas as pd # Example: replace with actual data loading
        # stock_data = pd.read_csv("your_stock_data.csv", index_col="Date", parse_dates=True)

        feature_cfg = FeatureConfig()
        training_params_cfg = TrainingParams()
        
        # Placeholder for actual config object if needed by ModelTrainer's __init__
        # For example, if config.MODEL_DIR is used.
        class MockConfig:
            MODEL_DIR = "models"
            # Add other attributes expected by ModelTrainer from the config object

        mock_config = MockConfig() # Replace with your actual config loading

        trainer = ModelTrainer(
            config=mock_config, # Pass the mock_config or your actual config
            feature_config=feature_cfg,
            training_params=training_params_cfg
        )
        
        # Train model with custom parameters
        trained_pipeline, model_metrics, conf_matrix, class_report = trainer.train_model(
            df=stock_data, # Make sure stock_data is prepared
            params=TrainingParams(n_estimators=200, max_depth=15)
        )
        
        # Save with version tracking
        if trained_pipeline and model_metrics:
            model_path_saved = trainer.save_model_with_manager(
                model_pipeline=trained_pipeline, # Pass the trained pipeline
                symbol="AAPL", 
                interval="1d", 
                metrics=model_metrics,
                # backend="sklearn" # Explicitly pass backend if not defaulted in ModelManager
            )
            if model_path_saved:
                print(f"Model saved to: {model_path_saved}")
        else:
            print("Model training failed, not saving.")
        ```
    """
    
    def __init__(
        self,
        config: Any, # Can be a simple object or dict if MODEL_DIR is the only thing used
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
                required_cols=required_columns
            )
            
            # Check validation results
            if not validation_result.is_valid:
                error_message = "; ".join(validation_result.errors) if validation_result.errors else "Unknown validation error"
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
    def _get_cached_feature_columns(self, dummy_df: pd.DataFrame) -> List[str]:
        """Internal helper to cache feature and target column names."""
        column_tuple = tuple(dummy_df.columns.tolist())
        return self.__cached_feature_columns(column_tuple)

    @lru_cache(maxsize=None)
    def __cached_feature_columns(self, column_tuple: Tuple[str, ...]) -> List[str]:
        """Cache feature column computation for performance."""
        # Pass tuple directly, assuming feature_engineer.get_feature_columns
        # is either not cached or expects a hashable argument if it is.
        return self.feature_engineer.get_feature_columns(column_tuple)

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
        '''Select top features based on importance or correlation.'''
        try:
            feature_cols = [col for col in df.columns if col not in ['target', 'future_return']]

            if self.feature_config.max_features is None: # ADDED CHECK
                logger.info("max_features is None, skipping feature selection.")
                return df

            # max_features is now guaranteed to be an int here
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
            
            class_counts = y.value_counts()
            minority_class_pct = class_counts.min() / len(y) * 100
            if minority_class_pct < 5:
                logger.warning(
                    f"Severe class imbalance: minority class {minority_class_pct:.1f}%"
                )
            logger.info(f"Class distribution: {dict(class_counts)}")
            
            pipeline = self._create_model_pipeline(params)
            cv_metrics = self._cross_validate_model(X, y, pipeline, params)
            
            logger.info("Training final model on full dataset...")
            pipeline.fit(X, y)
            
            y_pred_final: np.ndarray
            _prediction_output_final = pipeline.predict(X)
            if isinstance(_prediction_output_final, tuple):
                logger.warning(
                    "pipeline.predict(X) in train_model returned a tuple. "
                    "Expected np.ndarray. Using the first element as predictions."
                )
                y_pred_final = _prediction_output_final[0]
                # Consider logging _prediction_output_final[1] if it might contain useful info or indicate an issue
            else:
                y_pred_final = _prediction_output_final
            
            y_proba_final: Optional[np.ndarray] = None
            if hasattr(pipeline, 'predict_proba'):
                y_proba_final = pipeline.predict_proba(X)[:, 1]
            
            final_metrics_tuple = self._calculate_comprehensive_metrics(y.to_numpy(), y_pred_final, y_proba_final)
            
            final_metrics_dict = final_metrics_tuple[0]
            cm_from_calc = final_metrics_tuple[1]
            report_from_calc = final_metrics_tuple[2]

            metrics_dict = {
                'cv_metrics': cv_metrics,
                'final_metrics': final_metrics_dict,
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
                f"Final accuracy: {final_metrics_dict.get('accuracy', 0.0):.4f}, "
                f"AUC: {final_metrics_dict.get('auc', 'N/A')}"

            )
            
            return pipeline, metrics_dict, cm_from_calc, report_from_calc
            
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
        '''Train and evaluate a single CV fold.'''
        try:
            # Create fresh pipeline for this fold
            fold_pipeline = self._create_model_pipeline(self.default_params) # Use self.default_params or specific fold params
            fold_pipeline.fit(X_train, y_train)
            
            y_pred_fold: np.ndarray
            _prediction_output_fold = fold_pipeline.predict(X_test)
            if isinstance(_prediction_output_fold, tuple):
                logger.warning(
                    f"fold_pipeline.predict(X_test) in fold {fold} returned a tuple. "
                    f"Expected np.ndarray. Using the first element as predictions."
                )
                y_pred_fold = _prediction_output_fold[0]
            else:
                y_pred_fold = _prediction_output_fold
            
            y_proba_fold: Optional[np.ndarray] = None
            if hasattr(fold_pipeline, 'predict_proba'):
                 y_proba_fold = fold_pipeline.predict_proba(X_test)[:, 1]
            
            metrics_tuple = self._calculate_comprehensive_metrics(y_test.to_numpy(), y_pred_fold, y_proba_fold)
            return metrics_tuple[0]
            
        except Exception as e:
            logger.warning(f"Error in fold {fold}: {e}")
            raise

    def _calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray, # MODIFIED: Directly np.ndarray
        y_pred: np.ndarray, # MODIFIED: Directly np.ndarray
        y_proba: Optional[np.ndarray] = None # MODIFIED: Directly Optional[np.ndarray]
    ) -> Tuple[Dict[str, Any], np.ndarray, str]:
        """
        Calculates and logs comprehensive classification metrics.
        Handles binary and multiclass classification.
        """
        # Inputs are now expected to be np.ndarray directly
        y_true_np = y_true
        y_pred_np = y_pred
        y_proba_np = y_proba

        num_classes = len(np.unique(y_true_np))
        is_multiclass = num_classes > 2

        metrics = {}
        try:
            metrics["accuracy"] = accuracy_score(y_true_np, y_pred_np)
            
            if is_multiclass:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_np, y_pred_np, average="weighted", zero_division=0
                )
            else: # FIXED: Added colon
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_np, y_pred_np, average="binary", zero_division=0
                )
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1_score"] = f1
            
            # ROC AUC Score
            if y_proba_np is not None:
                if is_multiclass:
                    # For multiclass, y_proba_np should be shape (n_samples, n_classes)
                    if y_proba_np.ndim == 1: # If it's 1D, it might be for the positive class in binary
                        if num_classes == 2: # Check if it's actually binary case passed as multiclass
                             metrics["roc_auc"] = roc_auc_score(y_true_np, y_proba_np)
                        else: # True multiclass but y_proba is 1D - cannot calculate easily
                            logger.warning("ROC AUC for multiclass requires y_proba with shape (n_samples, n_classes). Skipping.")
                            metrics["roc_auc"] = None
                    else:
                        metrics["roc_auc"] = roc_auc_score(y_true_np, y_proba_np, multi_class="ovr", average="weighted")
                else: # Binary classification
                    # y_proba_np should be probabilities of the positive class
                    metrics["roc_auc"] = roc_auc_score(y_true_np, y_proba_np)
            else:
                metrics["roc_auc"] = None # ROC AUC cannot be computed without probabilities

            # Classification Report
            report_str = classification_report(y_true_np, y_pred_np, zero_division=0, output_dict=False)
            
            # Confusion Matrix
            cm = confusion_matrix(y_true_np, y_pred_np)

            logger.info(f"Model Metrics: {metrics}")
            logger.debug(f"Classification Report:\n{report_str}")
            logger.debug(f"Confusion Matrix:\n{cm}")
            
            return metrics, cm, str(report_str) # Ensure report is string

        except ValueError as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return default/error values
            error_metrics = {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, 
                "f1_score": 0.0, "roc_auc": None
            }
            dummy_cm = np.array([[0, 0], [0, 0]]) if not is_multiclass else np.zeros((num_classes, num_classes))
            return error_metrics, dummy_cm, "Error in metrics calculation."

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
        model_pipeline: Pipeline, 
        symbol: str, 
        interval: str, 
        metrics: Dict[str, Any],
        backend: str = "sklearn", # Added backend parameter, default to sklearn
        model_type_enum: Optional[ManagerModelType] = None, # Use aliased ModelType
        **kwargs 
    ) -> Optional[Path]:
        """
        Saves the trained model and its metadata using ModelManager.

        Args:
            model_pipeline: The trained scikit-learn pipeline.
            symbol: Stock symbol (e.g., AAPL).
            interval: Data interval (e.g., 1d, 1h).
            metrics: Dictionary of performance metrics.
            backend: The ML backend used (e.g., "sklearn", "pytorch").
            model_type_enum: The type of model (e.g., ManagerModelType.RANDOM_FOREST).
                           This should come from train.model_manager.ModelType.
            **kwargs: Additional metadata to save.

        Returns:
            Path to the saved model file, or None if saving failed.
        """
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            main_accuracy = None
            if isinstance(metrics, dict):
                if 'final_metrics' in metrics and isinstance(metrics['final_metrics'], dict):
                    main_accuracy = metrics['final_metrics'].get('accuracy')
                elif 'accuracy' in metrics:
                    main_accuracy = metrics.get('accuracy')

            if not isinstance(main_accuracy, (float, int, type(None))):
                main_accuracy = None
            
            metadata_obj = ModelMetadata(
                version=timestamp_str, 
                saved_at=datetime.now().isoformat(),
                accuracy=main_accuracy,
                parameters={
                    "ml_trainer_full_metrics": metrics, 
                    "feature_config": self.feature_config.__dict__,
                    "training_params": self.default_params.__dict__,
                    "custom_kwargs": kwargs,
                    "symbol": symbol.upper(),
                    "interval": interval
                },
                framework_version=sklearn.__version__,
                backend=backend, 
                tags=["stocktrader_v2", backend, symbol.upper(), interval]
            )
            
            model_manager_config = ModelManagerConfig(base_directory=self.model_dir)
            model_manager = ModelManager(config=model_manager_config)

            logger.info(f"Attempting to save model for symbol {symbol} with version {timestamp_str}, backend: {backend}")
            
            saved_path_str = model_manager.save_model(
                model=model_pipeline,
                metadata=metadata_obj, 
                backend=backend
            )

            if saved_path_str:
                logger.info(f"Model and metadata saved successfully. Path: {saved_path_str}")
                return Path(saved_path_str)
            else:
                logger.error(f"Failed to save model for symbol {symbol} using ModelManager (save_model returned None/empty).")
                return None

        except Exception as e:
            _model_name_ref = f"{symbol}_{interval}" if 'symbol' in locals() and 'interval' in locals() else 'UNKNOWN_MODEL'
            logger.error(f"Error saving model {_model_name_ref}: {e}", exc_info=True)
            return None

    def get_model_info(self, model_path: Union[str, Path]) -> Optional[ModelMetadata]:
        """
        Retrieves metadata for a given model path using ModelManager.
        """
        try:
            model_p = Path(model_path)
            if not model_p.is_file():
                logger.error(f"Model file path does not exist or is not a file: {model_p}")
                return None 

            model_manager_config = ModelManagerConfig(base_directory=self.model_dir)
            model_manager = ModelManager(config=model_manager_config)
            
            try:
                _loaded_model, loaded_metadata = model_manager.load_model(path=str(model_p))

                if loaded_metadata:
                    logger.info(f"Successfully loaded metadata for: {model_p} via ModelManager")
                    return loaded_metadata 
                else:
                    logger.warning(f"Failed to load metadata via ModelManager for: {model_p} (load_model returned None for metadata)")
                    return None
            except FileNotFoundError: # Specific exception for model not found by manager
                logger.error(f"Model file not found by ModelManager at {model_p}", exc_info=False)
                return None
            except Exception as mm_load_error: # Catch other model_manager load errors
                logger.error(f"ModelManager failed to load metadata for {model_p}: {mm_load_error}", exc_info=True)
                return None
                
        except Exception as e: # Catch broader errors like Path issues
            logger.error(f"Error retrieving model info for {model_path}: {e}", exc_info=True)
            return None

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

    def _cleanup_temp_files(self):
        """Remove temporary files and directories created during processing."""
        try:
            # Example: remove a temp directory if it exists
            temp_dir = Path(self.config.TEMP_DIR)
            if temp_dir.exists():
                for child in temp_dir.iterdir():
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        shutil.rmtree(child)
                temp_dir.rmdir()  # Remove the empty directory itself
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            
        except Exception as e:
            logger.warning(f"Temporary file cleanup failed: {e}")