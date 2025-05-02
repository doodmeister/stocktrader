import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from train.config import TrainingConfig
from utils.patterns_nn import PatternNN
from utils.etrade_candlestick_bot import ETradeClient
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
    min_patterns: int = 100            # Minimum total patterns needed
    max_samples_per_symbol: int = 10000  # Per-symbol sample cap

class PatternModelTrainer:
    """Handles the training of neural network models for candlestick pattern recognition."""

    def __init__(
        self,
        client: ETradeClient,
        model_manager: ModelManager,
        config: TrainingConfig
    ):
        self.client = client
        self.model_manager = model_manager
        self.config = config
        self._validate_dependencies()
        # Validate config values
        self._validate_training_params()

    def _validate_dependencies(self) -> None:
        if not isinstance(self.client, ETradeClient):
            raise ValueError("Invalid ETradeClient instance")
        if not isinstance(self.model_manager, ModelManager):
            raise ValueError("Invalid ModelManager instance")

    def _validate_training_params(self) -> None:
        from utils.validation import validate_training_params
        validate_training_params(self.config)

    def prepare_training_data(
        self,
        symbols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect and prepare training data from historical candlesticks.
        """
        X, y = [], []

        for symbol in tqdm(symbols, desc="Collecting training data"):
            try:
                features, labels = self._process_symbol(symbol)
                # Enforce per-symbol cap
                if len(features) > self.config.max_samples_per_symbol:
                    features = features[: self.config.max_samples_per_symbol]
                    labels = labels[: self.config.max_samples_per_symbol]

                X.extend(features)
                y.extend(labels)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                continue

        if len(X) < self.config.min_patterns:
            raise ValueError(
                f"Insufficient training data: {len(X)} samples. "
                f"Need at least {self.config.min_patterns} patterns."
            )

        return np.array(X), np.array(y)

    def _process_symbol(
        self,
        symbol: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process individual symbol data into training sequences.
        """
        df = self.client.get_candles(symbol, interval="5min", days=30)
        if df.empty:
            raise ValueError(f"No data available for {symbol}")

        features, labels = [], []
        for i in range(len(df) - self.config.seq_len):
            # Extract and normalize sequence
            window = df.iloc[i : i + self.config.seq_len][["open", "high", "low", "close"]].values
            seq = (window - window[0]) / window[0]

            # Detect patterns in this window
            patterns = CandlestickPatterns.detect_patterns(
                df.iloc[i : i + self.config.seq_len]
            )

            if patterns:
                features.append(seq)
                labels.append(self._encode_patterns(patterns))

        return features, labels

    def _encode_patterns(
        self,
        patterns: List[str]
    ) -> np.ndarray:
        """One-hot encode detected patterns."""
        label = np.zeros(len(PatternNN.PATTERN_CLASSES), dtype=float)
        for pat in patterns:
            if pat in PatternNN.PATTERN_CLASSES:
                idx = PatternNN.PATTERN_CLASSES.index(pat)
                label[idx] = 1.0
        return label

    def train_model(
        self,
        symbols: List[str],
        model: Optional[PatternNN] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PatternNN:
        """
        Train a pattern recognition model on historical data.
        """
        logger.info(f"Starting model training with {len(symbols)} symbols")

        # Device selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # Prepare data
            X, y = self.prepare_training_data(symbols)
            X_train, X_val, y_train, y_val = self._train_test_split(X, y)

            # Initialize or fine-tune model
            model = model or PatternNN()
            model.to(device)

            # Data loaders
            train_loader = self._create_data_loader(X_train, y_train)
            val_loader = self._create_data_loader(X_val, y_val)

            # Fit the model (ensure PatternNN has a fit method)
            model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
                early_stopping_patience=self.config.early_stopping_patience,
                device=device
            )

            # Persist the trained model
            self._save_model(model, metadata)
            return model

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
    client,
    symbols,
    model=None,
    epochs: int = 10,
    seq_len: int = 10,
    learning_rate: float = 0.001
):
    """
    Convenience wrapper that mirrors the old signature:
    train_pattern_model(client, symbols, model, epochs, seq_len, learning_rate)
    """
    # Build the config and manager
    config = TrainingConfig(
        epochs=epochs,
        seq_len=seq_len,
        learning_rate=learning_rate
    )
    manager = ModelManager()

    # Instantiate and run the trainer
    trainer = PatternModelTrainer(
        client=client,
        model_manager=manager,
        config=config
    )
    return trainer.train_model(symbols=symbols, model=model)