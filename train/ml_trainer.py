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
import joblib  # Added joblib for model persistence
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler

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

@dataclass
class TrainingParams:
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 3  # Lowered for small datasets

class ModelTrainer:
    def __init__(
        self, 
        config: Any,
        feature_config: Optional[FeatureConfig] = None,
        training_params: Optional[TrainingParams] = None
    ):
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
        if df.isna().any().any():
            logger.warning(f"DataFrame contains NaN values. Rows with NaN: {df.isna().any(axis=1).sum()}")

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        result_df.columns = [c.lower() for c in result_df.columns]
        # Dynamically adjust rolling windows based on data length
        available_windows = [w for w in self.feature_config.ROLLING_WINDOWS if w < len(result_df) - self.feature_config.TARGET_HORIZON]
        if not available_windows:
            raise ValidationError(
                f"Not enough data points. Got {len(result_df)}, need at least {min(self.feature_config.ROLLING_WINDOWS) + self.feature_config.TARGET_HORIZON + 1} for feature engineering."
            )
        for window in available_windows:
            window_features = self._calc_rolling_features(result_df, window)
            for name, series in window_features.items():
                result_df[f"{name}_{window}"] = series
        result_df['target'] = (
            result_df['close'].shift(-self.feature_config.TARGET_HORIZON) > result_df['close']
        ).astype(int)
        result_df = result_df.dropna()
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
        # Only include columns that actually exist in the DataFrame
        feature_cols = []
        for window in self.feature_config.ROLLING_WINDOWS:
            for feat in [
                'close_pct_change', 'close_std', 'volume_pct_change',
                'volume_std', 'price_volume_corr', 'sma', 'macd'
            ]:
                col_name = f"{feat}_{window}"
                if col_name in df.columns:
                    feature_cols.append(col_name)
        return feature_cols

    def train_model(
        self,
        df: pd.DataFrame,
        params: Optional[TrainingParams] = None
    ) -> Tuple[Pipeline, Dict[str, Dict[str, float]], np.ndarray, str]:
        logger.info(f"Starting training for {len(df)} rows with params: {params}")
        logger.info(f"Starting model training with data shape: {df.shape}")
        params = params or self.default_params
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
            
            # Create a single sklearn Pipeline that scales then classifies
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

    def _aggregate_metrics(
        self, 
        metric_list: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        result = {'mean': {}, 'std': {}}
        all_keys = set()
        for metrics in metric_list:
            all_keys.update(metrics.keys())
        for key in all_keys:
            values = [m.get(key, 0) for m in metric_list]
            result['mean'][key] = np.mean(values)
            result['std'][key] = np.std(values)
        return result

    def save_model(
        self,
        pipeline: Pipeline,
        symbol: str,
        interval: str
    ) -> Path:
        """
        Persist the full sklearn Pipeline (scaler + model) to disk.
        Returns the filepath of the saved .joblib file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{interval}_{timestamp}.joblib"
        filepath = self.config.MODEL_DIR / filename
        try:
            joblib.dump(pipeline, filepath)
            logger.info(f"Pipeline saved successfully to {filepath}")
            return filepath
        except Exception as e:
            logger.exception(f"Error saving pipeline: {e}")
            raise ModelError(f"Failed to save pipeline: {e}") from e

    def load_model(self, model_path: Path) -> Pipeline:
        """
        Load and return a previously saved Pipeline from disk.
        """
        try:
            pipeline = joblib.load(model_path)
            logger.info(f"Pipeline loaded from {model_path}")
            return pipeline
        except Exception as e:
            logger.exception(f"Error loading pipeline: {e}")
            raise ModelError(f"Failed to load pipeline: {e}") from e

    def predict(
        self,
        df: pd.DataFrame,
        model_path: Path
    ) -> pd.Series:
        """
        Given a new OHLCV DataFrame and a saved Pipeline path,
        runs the same feature engineering + scaling + model.predict
        and returns a Series of 0/1 predictions indexed by the dates.
        """
        # 1) Load pipeline
        pipeline = self.load_model(model_path)

        # 2) Run feature engineering (must match training)
        fe_df = self.feature_engineering(df)
        feature_cols = self.get_feature_columns(fe_df)
        if not feature_cols:
            raise ValidationError("No feature columns found during predict()")

        # 3) Predict
        X = fe_df[feature_cols]
        preds = pipeline.predict(X)

        # 4) Return as a Pandas Series aligned with fe_df index
        return pd.Series(preds, index=fe_df.index, name="model_signal")

    def predict_proba(
        self,
        df: pd.DataFrame,
        model_path: Path
    ) -> pd.Series:
        """
        Given a new OHLCV DataFrame and a saved Pipeline path,
        returns the probability of a positive signal (class 1) as a Series.
        
        This allows for more nuanced trading strategies than binary predictions.
        """
        # Load pipeline
        pipeline = self.load_model(model_path)
        
        # Run feature engineering (must match training)
        fe_df = self.feature_engineering(df)
        feature_cols = self.get_feature_columns(fe_df)
        if not feature_cols:
            raise ValidationError("No feature columns found during predict_proba()")
        
        # Get probabilities (second column is probability of class 1)
        X = fe_df[feature_cols]
        proba = pipeline.predict_proba(X)[:, 1]
        
        # Return as a Pandas Series aligned with fe_df index
        return pd.Series(proba, index=fe_df.index, name="model_buy_proba")
