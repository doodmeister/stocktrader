"""ModelTrainer Module

Handles feature engineering, model training, evaluation and persistence of ML models
for stock market prediction using a standardized pipeline approach with robust
validation, error handling, and performance optimizations.
"""

from utils.logger import setup_logger
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from train.model_manager import ModelManager, ModelMetadata
from patterns.pattern_utils import get_pattern_names, get_pattern_method
from train.feature_engineering import add_candlestick_pattern_features

logger = setup_logger(__name__)

class ModelError(Exception):
    pass

class ValidationError(ModelError):
    pass

class ModelType(Enum):
    RANDOM_FOREST = "random_forest"

@dataclass
class FeatureConfig:
    PRICE_FEATURES: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'volume'])
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [5, 10, 20])
    TARGET_HORIZON: int = 1
    use_candlestick_patterns: bool = True
    selected_patterns: Optional[List[str]] = None

@dataclass
class TrainingParams:
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 3

def add_candlestick_pattern_features(df: pd.DataFrame, selected_patterns: Optional[List[str]] = None) -> pd.DataFrame:
    pattern_names = selected_patterns or get_pattern_names()
    for pattern in pattern_names:
        method = get_pattern_method(pattern)
        if method is None:
            logger.warning(f"No detection method found for pattern: {pattern}")
            continue
        min_rows = 3
        try:
            from patterns.patterns import CandlestickPatterns
            for name, _, mr in CandlestickPatterns._PATTERNS:
                if name == pattern:
                    min_rows = mr
                    break
        except Exception:
            pass
        results = []
        for i in range(len(df)):
            if i + 1 < min_rows:
                results.append(0)
                continue
            window = df.iloc[i + 1 - min_rows : i + 1]
            try:
                detected = int(method(window)) if method else 0
            except Exception:
                detected = 0
            results.append(detected)
        df[pattern.replace(" ", "")] = results
    return df

class ModelTrainer:
    def __init__(self, config: Any, feature_config: Optional[FeatureConfig] = None, training_params: Optional[TrainingParams] = None):
        self.config = config
        self.feature_config = feature_config or FeatureConfig()
        self.default_params = training_params or TrainingParams()

    def validate_input_data(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValidationError("Input DataFrame is empty")
        required_columns = [col.lower() for col in self.feature_config.PRICE_FEATURES]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValidationError("DataFrame index must be DatetimeIndex")
        if df['close'].isna().all():
            raise ValidationError("Column 'close' contains only NaNs")
        if not all(np.issubdtype(df[col].dtype, np.number) for col in required_columns):
            raise ValidationError(f"One or more required columns are not numeric: {required_columns}")
        if df.isna().any().any():
            logger.warning(f"DataFrame contains NaN values. Rows with NaN: {df.isna().any(axis=1).sum()}")
        if not np.isfinite(df.values).all():
            raise ValueError("Input data contains NaN, infinity, or values too large for dtype('float64').")

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        result_df.columns = [c.lower() for c in result_df.columns]
        available_windows = [
            w for w in self.feature_config.ROLLING_WINDOWS
            if w < len(result_df) - self.feature_config.TARGET_HORIZON
        ]
        if not available_windows:
            raise ValidationError(
                f"Not enough data points. Got {len(result_df)}, need at least "
                f"{min(self.feature_config.ROLLING_WINDOWS) + self.feature_config.TARGET_HORIZON + 1} "
                f"for feature engineering."
            )
        for window in available_windows:
            window_features = self._calc_rolling_features(result_df, window)
            for name, series in window_features.items():
                result_df[f"{name}_{window}"] = series.replace([np.inf, -np.inf], np.nan)
        if self.feature_config.use_candlestick_patterns:
            result_df = add_candlestick_pattern_features(result_df, self.feature_config.selected_patterns)
        result_df['target'] = (
            result_df['close'].shift(-self.feature_config.TARGET_HORIZON) > result_df['close']
        ).astype(int)
        result_df = result_df.dropna()
        logger.info(f"Feature-engineered data shape: {result_df.shape}")
        logger.debug(f"Feature-engineered data statistics:\n{result_df.describe()}")
        return result_df

    def _calc_rolling_features(self, df: pd.DataFrame, window: int) -> Dict[str, pd.Series]:
        features = {}
        features['close_pct_change'] = df['close'].pct_change(window)
        features['close_std'] = df['close'].rolling(window).std()
        features['volume_pct_change'] = df['volume'].pct_change(window)
        features['volume_std'] = df['volume'].rolling(window).std()
        features['price_volume_corr'] = df['close'].rolling(window).corr(df['volume'])
        features['sma'] = df['close'].rolling(window).mean()
        if window > 1:
            ema_fast = df['close'].ewm(span=window//2, adjust=False).mean()
            ema_slow = df['close'].ewm(span=window, adjust=False).mean()
            features['macd'] = ema_fast - ema_slow
        return features

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        feature_cols = []
        for window in self.feature_config.ROLLING_WINDOWS:
            for feat in [
                'close_pct_change', 'close_std', 'volume_pct_change',
                'volume_std', 'price_volume_corr', 'sma', 'macd'
            ]:
                col_name = f"{feat}_{window}"
                if col_name in df.columns:
                    feature_cols.append(col_name)
        if self.feature_config.use_candlestick_patterns:
            all_patterns = self.feature_config.selected_patterns or get_pattern_names()
            for pattern in all_patterns:
                col_name = pattern.replace(" ", "")
                if col_name in df.columns:
                    feature_cols.append(col_name)
        return feature_cols

    def train_model(self, df: pd.DataFrame, params: Optional[TrainingParams] = None) -> Tuple[Pipeline, Dict[str, Dict[str, float]], np.ndarray, str]:
        logger.info(f"Starting training for {len(df)} rows with params: {params}")
        try:
            self.validate_input_data(df)
            features_df = self.feature_engineering(df)
            logger.debug(f"Feature-engineered data shape: {features_df.shape}")
            feature_cols = self.get_feature_columns(features_df)
            if not feature_cols:
                raise ValidationError("No feature columns available after feature engineering.")
            X = features_df[feature_cols]
            y = features_df['target']
            if len(X) < 5:
                raise ValidationError(f"Not enough samples after feature engineering: {len(X)} rows.")
            if y.nunique() < 2:
                raise ValidationError("Target column has only one class — cannot train a classifier.")
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=params.n_estimators,
                    max_depth=params.max_depth,
                    min_samples_split=params.min_samples_split,
                    random_state=params.random_state,
                    n_jobs=params.n_jobs
                ))
            ])
            tscv = TimeSeriesSplit(n_splits=min(params.cv_folds, max(2, len(X) // 2)))
            cv_metrics = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                fold_metrics = self._calculate_metrics(y_test, y_pred)
                cv_metrics.append(fold_metrics)
            pipeline.fit(X, y)
            y_pred_final = pipeline.predict(X)
            final_metrics = self._calculate_metrics(y, y_pred_final)
            cm = confusion_matrix(y, y_pred_final)
            report = classification_report(y, y_pred_final)
            metrics_dict = self._aggregate_metrics(cv_metrics)
            metrics_dict['final_metrics'] = final_metrics
            logger.info(f"Model training completed. Final accuracy: {final_metrics['accuracy']:.4f}")
            return pipeline, metrics_dict, cm, report
        except Exception as e:
            logger.exception(f"Error during model training: {e}")
            raise ModelError(f"Model training failed: {str(e)}") from e

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _aggregate_metrics(self, metric_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        result = {'mean': {}, 'std': {}}
        all_keys = set()
        for metrics in metric_list:
            all_keys.update(metrics.keys())
        for key in all_keys:
            values = [m.get(key, 0) for m in metric_list]
            result['mean'][key] = np.mean(values)
            result['std'][key] = np.std(values)
        return result

    def save_model(self, pipeline: Pipeline, symbol: str, interval: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{interval}_{timestamp}.joblib"
        filepath = self.config.MODEL_DIR / filename
        try:
            joblib.dump(pipeline, filepath)
            if not filepath.exists():
                raise ModelError(f"Model file not found after save: {filepath}")
            logger.info(f"Pipeline saved successfully to {filepath}")
            return filepath
        except Exception as e:
            logger.exception(f"Error saving pipeline: {e}")
            raise ModelError(f"Failed to save pipeline: {e}") from e

    def save_model_with_manager(self, pipeline: Pipeline, symbol: str, interval: str, metrics: dict = None, backend: str = "Classic ML (RandomForest)") -> str:
        model_manager = ModelManager(base_directory=str(self.config.MODEL_DIR))
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        expected_features = self.get_feature_columns(pd.DataFrame(columns=self.feature_config.PRICE_FEATURES))
        metadata = ModelMetadata(
            version=version,
            saved_at=datetime.now().isoformat(),
            accuracy=metrics.get("accuracy") if metrics else None,
            parameters={
                "symbol": symbol,
                "interval": interval,
                "features": expected_features,
                "rolling_windows": self.feature_config.ROLLING_WINDOWS,
                "patterns": self.feature_config.selected_patterns if self.feature_config.use_candlestick_patterns else []
            },
            framework_version="sklearn"
        )
        logger.info(f"Saving model with backend: {backend}")
        save_path = model_manager.save_model(
            model=pipeline,
            metadata=metadata,
            backend=backend
        )
        logger.info(f"Model saved to: {save_path}")
        return save_path

    def load_model(self, model_path: Path) -> Pipeline:
        try:
            pipeline = joblib.load(model_path)
            if not hasattr(pipeline, "predict"):
                raise ModelError("Loaded pipeline does not implement predict().")
            logger.info(f"Pipeline loaded from {model_path}")
            return pipeline
        except Exception as e:
            logger.exception(f"Error loading pipeline: {e}")
            raise ModelError(f"Failed to load pipeline: {e}") from e

    def predict(self, df: pd.DataFrame, model_path: Path) -> pd.Series:
        pipeline = self.load_model(model_path)
        fe_df = self.feature_engineering(df)
        feature_cols = self.get_feature_columns(fe_df)
        if not feature_cols:
            raise ValidationError("No feature columns found during predict()")
        X = fe_df[feature_cols]
        preds = pipeline.predict(X)
        return pd.Series(preds, index=fe_df.index, name="model_signal")

    def predict_proba(self, df: pd.DataFrame, model_path: Path) -> pd.Series:
        pipeline = self.load_model(model_path)
        if not hasattr(pipeline, "predict_proba"):
            raise ModelError("Loaded pipeline does not implement predict_proba().")
        fe_df = self.feature_engineering(df)
        feature_cols = self.get_feature_columns(fe_df)
        if not feature_cols:
            raise ValidationError("No feature columns found during predict_proba()")
        X = fe_df[feature_cols]
        proba = pipeline.predict_proba(X)[:, 1]
        return pd.Series(proba, index=fe_df.index, name="model_buy_proba")
