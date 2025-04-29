
"""ModelTrainer Module

Handles feature engineering, model training, evaluation and persistence of ML models
for stock market prediction.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from logging.handlers import RotatingFileHandler

ModelMetrics = Dict[str, float]

class ModelType(Enum):
    RANDOM_FOREST = "random_forest"

@dataclass
class FeatureConfig:
    PRICE_FEATURES: List[str] = ('Open', 'High', 'Low', 'Close', 'Volume')
    ROLLING_WINDOWS: List[int] = (5, 10, 20)
    TARGET_HORIZON: int = 1
    MIN_SAMPLES: int = 100

@dataclass
class TrainingParams:
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 10
    model_type: ModelType = ModelType.RANDOM_FOREST

class ModelTrainer:
    def __init__(self, config, feature_config: Optional[FeatureConfig] = None):
        self.model_dir = Path(config.MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_config = feature_config or FeatureConfig()
        self.scaler = StandardScaler()
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)
        handler = RotatingFileHandler(
            self.model_dir / 'model_training.log',
            maxBytes=1024*1024,
            backupCount=5
        )
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def validate_input_data(self, df: pd.DataFrame) -> None:
        required_columns = set(self.feature_config.PRICE_FEATURES)
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        if len(df) < self.feature_config.MIN_SAMPLES:
            raise ValueError(f"Need at least {self.feature_config.MIN_SAMPLES} rows")
        if df.isnull().any().any():
            raise ValueError("Input data contains null values")

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input_data(df)
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        for w in self.feature_config.ROLLING_WINDOWS:
            df[f'volatility_{w}'] = df['returns'].rolling(window=w).std()
            df[f'sma_{w}'] = df['Close'].rolling(window=w).mean()
            df[f'volume_sma_{w}'] = df['Volume'].rolling(window=w).mean()
        df['target'] = df['returns'].shift(-self.feature_config.TARGET_HORIZON).apply(lambda x: 1 if x > 0 else 0)
        return df.dropna()

    def get_feature_columns(self) -> List[str]:
        base = list(self.feature_config.PRICE_FEATURES)
        derived = ['returns'] + [
            f for w in self.feature_config.ROLLING_WINDOWS
            for f in (f'volatility_{w}', f'sma_{w}', f'volume_sma_{w}')
        ]
        return base + derived

    def train_model(
        self,
        df: pd.DataFrame,
        params: TrainingParams = TrainingParams()
    ) -> Tuple[object, ModelMetrics, np.ndarray, str]:
        df = self.feature_engineering(df)
        features = self.get_feature_columns()
        X = self.scaler.fit_transform(df[features])
        y = df['target'].values

        tscv = TimeSeriesSplit(n_splits=5)
        train_accs, test_accs = [], []

        model = RandomForestClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            random_state=42,
            n_jobs=-1
        )

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            train_accs.append(accuracy_score(y_train, model.predict(X_train)))
            test_accs.append(accuracy_score(y_test, model.predict(X_test)))

        # Final training on full data
        model.fit(X, y)
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred)

        metrics = {
            'train_acc_mean': float(np.mean(train_accs)),
            'train_acc_std': float(np.std(train_accs)),
            'test_acc_mean': float(np.mean(test_accs)),
            'test_acc_std': float(np.std(test_accs))
        }

        self.logger.info(f"Training complete. Metrics: {metrics}")
        return model, metrics, cm, report

    def save_model(self, model, symbol: str, interval: str) -> Path:
        model_path = self.model_dir / f"{symbol}_{interval}_model.pkl"
        joblib.dump({
            'model': model,
            'scaler': self.scaler,
            'features': self.get_feature_columns()
        }, model_path)
        self.logger.info(f"Saved model to {model_path}")
        return model_path
