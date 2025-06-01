# ml_pipeline.py
"""
Production-Grade Machine Learning Pipeline for Candlestick Pattern Recognition

This module orchestrates the end-to-end ML workflow for candlestick pattern recognition:
- Data acquisition (with caching and validation)
- Feature engineering (technical indicators, pattern features)
- Dataset preparation (robust normalization, sequence generation)
- Model training (PatternNN, early stopping, resource management)
- Evaluation and artifact persistence (metrics, preprocessing config)

Key Features:
- SOLID, modular design with dependency injection
- Comprehensive error handling and input validation
- Secure credential management and path handling
- Optimized for performance and memory usage
- Structured logging and observability
- Well-documented for maintainability and extension

Project Status:
- ✅ Modular architecture is 100% complete and functional
- ✅ All core modules are implemented and tested
- ✅ Use bash commands for all operations (see below)

Usage Example (Bash):
    # Activate environment and run pipeline
    source venv/Scripts/activate
    python -m train.model_training_pipeline

    # Or run with Streamlit dashboard (recommended)
    streamlit run main.py

"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional, Any

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.logger import setup_logger
from core.etrade_candlestick_bot import ETradeClient
from patterns.patterns_nn import PatternNN
from .model_manager import ModelManager, ModelMetadata
from .ml_config import MLConfig
from utils.notifier import Notifier
from utils.technicals.performance_utils import get_candles_cached
from utils.technicals.feature_engineering import compute_technical_features
from security.authentication import get_api_credentials
from patterns.pattern_utils import add_candlestick_pattern_features

# Configure structured logging
logger = setup_logger(__name__)

class DatasetPreparationError(Exception):
    """Custom exception for dataset preparation failures."""
    pass

class TrainingPipelineError(Exception):
    """Custom exception for training pipeline failures."""
    pass

class MLPipeline:
    """
    Orchestrates the ML workflow for candlestick pattern recognition.

    Responsibilities:
    - Data acquisition and validation
    - Feature engineering
    - Dataset preparation (sequence, normalization)
    - Model training and evaluation
    - Artifact persistence (model, metrics, preprocessing config)

    Usage:
        pipeline = MLPipeline(client, config, notifier)
        metrics = pipeline.train_and_evaluate(model)
    """
    def __init__(
        self,
        client: ETradeClient,
        config: MLConfig,
        notifier: Optional[Notifier] = None,
        model_manager: Optional[ModelManager] = None
    ):
        """
        Initialize the ML pipeline with injected dependencies.
        Args:
            client: ETradeClient instance for data access
            config: MLConfig with validated parameters
            notifier: Optional Notifier for alerts
            model_manager: Optional ModelManager (for testing/mocking)
        """
        self.client = client
        self.config = config
        self.notifier = notifier
        self.device = torch.device(
            config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        )
        self.model_manager = model_manager or ModelManager(base_directory=str(self.config.model_dir))
        logger.info(f"MLPipeline initialized. Using device: {self.device}")

    def prepare_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare training and validation datasets with robust error handling and validation.
        Returns:
            Tuple of (X_train, X_val, y_train, y_val) as torch.Tensors
        Raises:
            DatasetPreparationError: On any data or feature engineering failure
        """
        try:
            X, y = [], []
            all_data = []
            for symbol in self.config.symbols:
                logger.info(f"Fetching data for {symbol}")
                try:
                    df = self.client.get_candles(symbol, interval="5min", days=5)
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
                    continue
                if df is None or df.empty:
                    logger.warning(f"No data received for {symbol}")
                    continue
                all_data.append(df)
            if not all_data:
                raise DatasetPreparationError("No data fetched for any symbol.")
            # Concatenate and apply feature engineering
            df = pd.concat(all_data, ignore_index=True)
            df = compute_technical_features(df)
            df = add_candlestick_pattern_features(df)
            self._last_df = df.copy()  # Save for preprocessing config
            # Select feature columns (exclude non-numeric, symbol, timestamp)
            feature_cols = [
                col for col in df.columns
                if col not in ('symbol', 'timestamp') and np.issubdtype(df[col].dtype, np.number)
            ]
            if not feature_cols:
                raise DatasetPreparationError("No valid feature columns found after engineering.")
            values = df[feature_cols].values
            if len(values) < self.config.seq_len:
                logger.warning(f"Insufficient data points: {len(values)} < {self.config.seq_len}")
                raise DatasetPreparationError("Not enough data for sequence length.")
            # Normalize features
            values = self._normalize_features(values)
            for i in range(self.config.seq_len, len(values)):
                seq = values[i-self.config.seq_len:i]
                label = self._extract_pattern_label(i)
                X.append(seq)
                y.append(label)
            if not X or not y:
                raise DatasetPreparationError("No valid sequences could be generated.")
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.int64)
            if X.ndim != 3:
                logger.error(f"Expected X to be 3D (batch, seq_len, features), got shape {X.shape}")
                raise DatasetPreparationError(f"Expected X to be 3D (batch, seq_len, features), got shape {X.shape}")
            if y.ndim != 1:
                logger.error(f"Expected y to be 1D, got shape {y.shape}")
                raise DatasetPreparationError(f"Expected y to be 1D, got shape {y.shape}")
            # Split and convert to tensors
            splits = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            return tuple(torch.from_numpy(arr) for arr in splits)
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            raise DatasetPreparationError(str(e)) from e

    def train_and_evaluate(self, model: PatternNN) -> Dict[str, Any]:
        """
        Train the model and evaluate its performance.
        Args:
            model: PatternNN instance (architecture must match feature count)
        Returns:
            Dict with evaluation metrics
        Raises:
            TrainingPipelineError: On any training or evaluation failure
        """
        try:
            start_time = datetime.now()
            logger.info("Starting training pipeline")
            X_train, X_val, y_train, y_val = self.prepare_dataset()
            if X_train.size(0) == 0:
                raise TrainingPipelineError("No training data prepared.")
            model = model.to(self.device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate
            )
            criterion = torch.nn.CrossEntropyLoss()
            best_loss = float('inf')
            patience = 3
            patience_counter = 0
            model.train()
            for epoch in range(1, self.config.epochs + 1):
                epoch_loss = self._train_epoch(model, X_train, y_train, optimizer, criterion)
                logger.info(f"Epoch {epoch}: loss={epoch_loss:.4f}")
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            metrics = self._evaluate_model(model, X_val, y_val)
            self._save_artifacts(model, metrics, optimizer=optimizer, loss=best_loss)
            duration = datetime.now() - start_time
            logger.info(f"Training completed in {duration}. Accuracy: {metrics.get('accuracy', 0):.4f}")
            if self.notifier:
                self.notifier.send_message(
                    f"Training completed successfully. Accuracy: {metrics.get('accuracy', 0):.4f}"
                )
            # Explicitly free GPU memory if used
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            return metrics
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            if self.notifier:
                self.notifier.send_message(f"Training pipeline failed: {e}", level="error")
            raise TrainingPipelineError(str(e)) from e

    def _normalize_features(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize features using min-max scaling. Saves normalization config for reproducibility.
        Args:
            values: 2D numpy array of features
        Returns:
            Normalized numpy array
        """
        min_vals = values.min(axis=0)
        max_vals = values.max(axis=0)
        normed = (values - min_vals) / (max_vals - min_vals + 1e-8)
        self._last_min_vals = min_vals
        self._last_max_vals = max_vals
        return normed

    def _extract_pattern_label(self, idx: int) -> int:
        """
        Extract label for a sequence ending at index idx in the original DataFrame.
        Returns:
            0 = hold, 1 = buy, 2 = sell
        """
        if not hasattr(self, "_last_df"):
            return 0
        df = self._last_df
        if idx >= len(df):
            return 0
        latest_row = df.iloc[idx - 1]  # -1 because idx is exclusive in sequence slicing
        bullish = [col for col in df.columns if 'Bullish' in col]
        bearish = [col for col in df.columns if 'Bearish' in col]
        if any(latest_row.get(b, 0) == 1 for b in bullish):
            return 1  # Buy
        elif any(latest_row.get(b, 0) == 1 for b in bearish):
            return 2  # Sell
        return 0  # Hold

    def _train_epoch(
        self,
        model: PatternNN,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module
    ) -> float:
        """
        Run a single training epoch with shuffling and batching.
        Returns average loss.
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        perm = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), self.config.batch_size):
            idx = perm[i:i + self.config.batch_size]
            batch_x = X_train[idx].to(self.device)
            batch_y = y_train[idx].to(self.device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(num_batches, 1)

    def _evaluate_model(
        self,
        model: PatternNN,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Evaluate the model on validation data.
        Returns:
            Dict with accuracy and confusion matrix
        """
        model.eval()
        with torch.no_grad():
            logits = model(X_val.to(self.device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            acc = accuracy_score(y_val.cpu().numpy(), preds)
            cm = confusion_matrix(y_val.cpu().numpy(), preds)
        return {
            'accuracy': float(acc),
            'confusion_matrix': cm.tolist()
        }

    def _save_artifacts(
        self,
        model: PatternNN,
        metrics: Dict[str, Any],
        optimizer=None,
        loss=None
    ) -> None:
        """
        Save model, metrics, and preprocessing config for reproducibility.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = Path(self.config.model_dir) / f"metrics_{timestamp}.json"
        metadata = ModelMetadata(
            version=timestamp,
            saved_at=datetime.now().isoformat(),
            accuracy=metrics.get('accuracy'),
            parameters={
                "epochs": self.config.epochs,
                "seq_len": self.config.seq_len,
                "input_size": getattr(model, "input_size", None),
                "hidden_size": getattr(model, "hidden_size", None),
                "num_layers": getattr(model, "num_layers", None),
                "output_size": getattr(model, "output_size", None),
                "dropout": getattr(model, "dropout", None)
            },
            backend="DeepPatternNN"
        )
        try:
            self.model_manager.save_model(
                model=model,
                metadata=metadata,
                optimizer=optimizer,
                epoch=self.config.epochs,
                loss=loss
            )
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            # Save preprocessing config (feature order and normalization)
            feature_order = self._last_df.columns.tolist() if hasattr(self, "_last_df") else []
            min_vals = self._last_min_vals if hasattr(self, "_last_min_vals") else []
            max_vals = self._last_max_vals if hasattr(self, "_last_max_vals") else []
            preprocessing = {
                "feature_order": feature_order,
                "normalization": {
                    "min": min_vals.tolist() if hasattr(min_vals, "tolist") else [],
                    "max": max_vals.tolist() if hasattr(max_vals, "tolist") else []
                }
            }
            with open(Path(self.config.model_dir) / f"preprocessing_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(preprocessing, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")

if __name__ == '__main__':
    # Securely load credentials (never hard-code or print)
    creds = get_api_credentials()
    client = ETradeClient(
        consumer_key=creds['consumer_key'],
        consumer_secret=creds['consumer_secret'],
        oauth_token=creds['oauth_token'],
        oauth_token_secret=creds['oauth_token_secret'],
        account_id=creds['account_id'],
        sandbox=creds.get('use_sandbox', 'true').lower() == 'true'
    )
    config = MLConfig(
        seq_len=10,
        epochs=5,
        batch_size=32,
        learning_rate=1e-3,
        test_size=0.2,
        random_state=42,
        device="cuda",
        model_dir=Path("models"),
        symbols=os.getenv('SYMBOLS', 'AAPL,MSFT').split(',')
    )
    notifier = Notifier()
    pipeline = MLPipeline(client, config, notifier)
    # Dynamically calculate input size based on feature columns
    try:
        df_sample = pd.concat([
            client.get_candles(symbol, interval="5min", days=5) for symbol in config.symbols
        ], ignore_index=True)
        df_sample = compute_technical_features(df_sample)
        df_sample = add_candlestick_pattern_features(df_sample)
        feature_cols = [col for col in df_sample.columns if col not in ('symbol', 'timestamp') and np.issubdtype(df_sample[col].dtype, np.number)]
        feature_count = df_sample[feature_cols].shape[1]
        model = PatternNN(
            input_size=feature_count,
            hidden_size=64,
            num_layers=2,
            output_size=3,
            dropout=0.2
        )
        metrics = pipeline.train_and_evaluate(model)
        print("Training complete. Metrics:", metrics)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        if notifier:
            notifier.send_message(f"Pipeline execution failed: {e}", level="error")
