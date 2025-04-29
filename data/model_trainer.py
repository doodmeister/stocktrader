"""ModelTrainer Module

Handles feature engineering, model training, evaluation, and saving models.
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles ML model training and persistence for OHLCV stock data."""

    def __init__(self, config):
        self.model_dir = config.MODEL_DIR
        self.model_dir.mkdir(exist_ok=True)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations on OHLCV data."""
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=5).std()
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)
        df = df.dropna()
        return df

    def train_model(self, df: pd.DataFrame, model_type: str = "random_forest"):
        """Train a machine learning model based on input dataframe features."""
        df_processed = self.feature_engineering(df)

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'returns', 'volatility', 'sma_5', 'sma_10']
        X = df_processed[features]
        y = df_processed['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        cm = confusion_matrix(y_test, model.predict(X_test))

        return model, train_acc, test_acc, cm

    def save_model(self, model, symbol: str, interval: str) -> Path:
        """Save a trained model to disk."""
        model_path = self.model_dir / f"{symbol}_{interval}_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        return model_path
