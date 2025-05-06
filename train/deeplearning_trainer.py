import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from train.config import TrainingConfig
from utils.patterns_nn import PatternNN
from patterns import CandlestickPatterns
from utils.model_manager import ModelManager


# Configure module logger
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration parameters for pattern model training."""
    epochs: int = 10
    seq_len: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    min_patterns: int = 20  # or even lower, e.g., 10 for testing
    max_samples_per_symbol: int = 10000  # Per-symbol sample cap

class PatternModelTrainer:
    """Handles the training of neural network models for candlestick pattern recognition."""

    def __init__(
        self,
        model_manager: ModelManager,
        config: TrainingConfig,
        selected_patterns: List[str]
    ):
        self.model_manager = model_manager
        self.config = config
        self.selected_patterns = selected_patterns
        self._validate_training_params()

    def _validate_training_params(self) -> None:
        from utils.validation import validate_training_params
        validate_training_params(self.config)

    def prepare_training_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect and prepare training data from provided candlestick data.
        """
        X, y = [], []

        try:
            features, labels = self._process_data(data)
            X.extend(features)
            y.extend(labels)

        except Exception as e:
            logger.error(f"Error processing data: {e}", exc_info=True)
            raise ValueError("Error in preparing training data")

        if len(X) < self.config.min_patterns:
            raise ValueError(
                f"Insufficient training data: {len(X)} samples. "
                f"Need at least {self.config.min_patterns} patterns."
            )

        return np.array(X), np.array(y)

    def _process_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features, labels = [], []
        for i in range(len(data) - self.config.seq_len):
            window = data.iloc[i : i + self.config.seq_len][["open", "high", "low", "close"]].values
            seq = (window - window[0]) / window[0]
            patterns = CandlestickPatterns.detect_patterns(
                data.iloc[i : i + self.config.seq_len]
            )
            if patterns:
                features.append(seq.flatten())  # Only append if label will be appended
                labels.append(self._encode_patterns(patterns, self.selected_patterns))
        return features, labels

    def _encode_patterns(
        self,
        patterns: List[str],
        selected_patterns: List[str]
    ) -> np.ndarray:
        """One-hot encode detected patterns."""
        label = np.zeros(len(selected_patterns), dtype=float)
        for i, pattern in enumerate(selected_patterns):
            if pattern in patterns:
                label[i] = 1
        return label

    def train_model(
        self,
        data: pd.DataFrame,
        model: Optional[PatternNN] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[PatternNN, Dict[str, float]]:
        """
        Train a pattern recognition model on provided data.
        """
        logger.info("Starting model training")

        # Device selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # Prepare data
            if data.empty:
                raise ValueError("Provided data is empty and cannot be used for training.")
            X, y = self.prepare_training_data(data)
            # Ensure sufficient data for splitting
            if len(X) < self.config.batch_size or len(X) * self.config.validation_split < 1:
                raise ValueError(
                    f"Insufficient data for splitting: {len(X)} samples. "
                    f"Ensure dataset size is greater than batch size ({self.config.batch_size}) "
                    f"and validation split requirements."
                )
            X_train, X_val, y_train, y_val = self._train_test_split(X, y)

            # Initialize or fine-tune model
            model = model or PatternNN(
                input_size=X.shape[1], 
                output_size=len(self.selected_patterns)
            )
            model.to(device)

            # Data loaders
            train_loader = self._create_data_loader(X_train, y_train)
            val_loader = self._create_data_loader(X_val, y_val)

            # Fit the model (ensure PatternNN has a fit method)
            metrics = model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
                early_stopping_patience=self.config.early_stopping_patience,
                device=device
            )

            # Persist the trained model
            self._save_model(model, metadata)
            return model, metrics

        except Exception as e:
            logger.error("Training failed", exc_info=True)
            raise ValueError(f"Model training failed: {e}") from e

    def _train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and validation sets."""
        split_idx = int(len(X) * (1 - self.config.validation_split))
        return (
            X[:split_idx],
            X[split_idx:],
            y[:split_idx],
            y[split_idx:]
        )

    def _create_data_loader(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> DataLoader:
        """Create PyTorch DataLoader for training."""
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32)
        return DataLoader(
            TensorDataset(tensor_x, tensor_y),
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )

    def _save_model(
        self,
        model: PatternNN,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save trained model with metadata."""
        meta = metadata.copy() if metadata else {}
        meta.update({
            "config": self.config.__dict__,
            "timestamp": datetime.now().isoformat()
        })
        self.model_manager.save_model(model=model, metadata=meta)

def train_pattern_model(
    symbols: List[str],
    data: pd.DataFrame,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    selected_patterns: List[str],
    # ...other params...
) -> Tuple[Any, Dict[str, float]]:
    # Prepare data, model, loss function, and optimizer
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    manager = ModelManager()
    trainer = PatternModelTrainer(
        model_manager=manager,
        config=config,
        selected_patterns=selected_patterns
    )

    X, y = trainer.prepare_training_data(data)
    train_loader = trainer._create_data_loader(X, y)

    model = PatternNN(input_size=X.shape[1], output_size=len(selected_patterns))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    metrics = {}
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
        # Optionally: log metrics, loss, etc.

    # Evaluate on training data (or use a validation set if available)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(batch_y, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=False)

    metrics = {
        "metrics": {
            "mean": {"accuracy": acc},  # Add more if you compute them
            "std": {},
            "final_metrics": {"accuracy": acc}
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    return model, metrics

if __name__ == "__main__":
    # Example usage or placeholder for main logic
    print("Please define the main function or provide script usage logic.")