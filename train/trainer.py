import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset

from stocktrader.models.pattern_nn import PatternNN
from stocktrader.etrade_candlestick_bot import ETradeClient
from stocktrader.patterns import CandlestickPatterns
from stocktrader.utils.model_manager import ModelManager
from stocktrader.utils.validation import validate_training_params

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
    min_patterns: int = 100  # Minimum patterns needed for training
    max_samples_per_symbol: int = 10000  # Prevent memory issues

class PatternModelTrainer:
    """Handles the training of neural network models for candlestick pattern recognition."""

    def __init__(
        self,
        client: ETradeClient,
        model_manager: ModelManager,
        config: TrainingConfig
    ):
        """
        Initialize the pattern model trainer.

        Args:
            client: Authenticated ETradeClient instance
            model_manager: ModelManager instance for model persistence
            config: Training configuration parameters
        """
        self.client = client
        self.model_manager = model_manager
        self.config = config
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        """Validate required dependencies and configurations."""
        if not isinstance(self.client, ETradeClient):
            raise ValueError("Invalid ETradeClient instance")
        if not isinstance(self.model_manager, ModelManager):
            raise ValueError("Invalid ModelManager instance")
        validate_training_params(self.config)

    def prepare_training_data(
        self, 
        symbols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect and prepare training data from historical candlesticks.

        Args:
            symbols: List of stock symbols to train on

        Returns:
            Tuple of (features, labels) as numpy arrays
        
        Raises:
            ValueError: If insufficient training data is collected
        """
        X, y = [], []
        
        for symbol in tqdm(symbols, desc="Collecting training data"):
            try:
                features, labels = self._process_symbol(symbol)
                X.extend(features)
                y.extend(labels)
                
                if len(X) >= self.config.max_samples_per_symbol:
                    logger.info(f"Reached max samples for {symbol}")
                    break
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
                continue

        if len(X) < self.config.min_patterns:
            raise ValueError(
                f"Insufficient training data: {len(X)} samples. "
                f"Need at least {self.config.min_patterns}"
            )

        return np.array(X), np.array(y)

    def _process_symbol(
        self, 
        symbol: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process individual symbol data into training sequences.

        Args:
            symbol: Stock symbol to process

        Returns:
            Tuple of (feature_sequences, pattern_labels)
        """
        df = self.client.get_candles(symbol, interval="5min", days=30)
        if df.empty:
            raise ValueError(f"No data available for {symbol}")

        features, labels = [], []
        for i in range(len(df) - self.config.seq_len):
            sequence = self._extract_sequence(df, i)
            patterns = CandlestickPatterns.detect_patterns(
                df.iloc[i:i + self.config.seq_len]
            )
            
            if patterns:
                features.append(sequence)
                labels.append(self._encode_patterns(patterns))

        return features, labels

    def _extract_sequence(
        self, 
        df: pd.DataFrame, 
        start_idx: int
    ) -> np.ndarray:
        """Extract and normalize a training sequence."""
        sequence = df.iloc[
            start_idx:start_idx + self.config.seq_len
        ][['open', 'high', 'low', 'close']].values
        
        # Normalize using percentage changes
        return (sequence - sequence[0]) / sequence[0]

    def _encode_patterns(
        self, 
        patterns: List[str]
    ) -> np.ndarray:
        """One-hot encode detected patterns."""
        label = np.zeros(len(PatternNN.PATTERN_CLASSES))
        for pattern in patterns:
            if pattern in PatternNN.PATTERN_CLASSES:
                label[PatternNN.PATTERN_CLASSES.index(pattern)] = 1
        return label

    def train_model(
        self,
        symbols: List[str],
        model: Optional[PatternNN] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PatternNN:
        """
        Train a pattern recognition model on historical data.

        Args:
            symbols: List of stock symbols to train on
            model: Optional existing model to fine-tune
            metadata: Optional metadata to save with model

        Returns:
            Trained PatternNN model

        Raises:
            ValueError: If training fails or validation errors occur
        """
        logger.info(f"Starting model training with {len(symbols)} symbols")
        
        try:
            # Prepare data
            X, y = self.prepare_training_data(symbols)
            X_train, X_val, y_train, y_val = self._train_test_split(X, y)

            # Initialize or use existing model
            model = model or PatternNN()
            
            # Create data loaders
            train_loader = self._create_data_loader(X_train, y_train)
            val_loader = self._create_data_loader(X_val, y_val)

            # Train model
            model.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
                early_stopping_patience=self.config.early_stopping_patience
            )

            # Save trained model
            self._save_model(model, metadata)
            
            return model

        except Exception as e:
            logger.error("Training failed", exc_info=True)
            raise ValueError(f"Model training failed: {str(e)}") from e

    def _train_test_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and validation sets."""
        split_idx = int(len(X) * (1 - self.config.validation_split))
        return (
            X[:split_idx], X[split_idx:],
            y[:split_idx], y[split_idx:]
        )

    def _create_data_loader(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> DataLoader:
        """Create PyTorch DataLoader for training."""
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y)
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

    def _save_model(
        self,
        model: PatternNN,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save trained model with metadata."""
        metadata = metadata or {}
        metadata.update({
            'config': self.config.__dict__,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        self.model_manager.save_model(
            model=model,
            metadata=metadata
        )