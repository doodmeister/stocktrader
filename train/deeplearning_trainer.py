from utils.logger import setup_logger
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, classification_report

from train.deeplearning_config import TrainingConfig
from patterns.patterns_nn import PatternNN
from patterns.patterns import CandlestickPatterns
from train.model_manager import ModelManager, ModelMetadata
from utils.technicals.feature_engineering import compute_technical_features
from train.feature_engineering import add_candlestick_pattern_features

logger = setup_logger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        from utils.config.validation import validate_training_params
        validate_training_params(self.config)
        if not self.selected_patterns:
            raise ValueError("No candlestick patterns selected for training.")

    def prepare_training_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")

        # --- Feature engineering step ---
        data = compute_technical_features(data)
        data = add_candlestick_pattern_features(data)
        self._last_df = data.copy()

        features, labels = self._process_data(data)
        if not features:
            raise ValueError("No valid sequences generated for training.")

        X = np.array(features)
        y = np.array(labels)

        # Ensure X is 3D: (batch, seq_len, feat_dim)
        if X.ndim == 2 and self.config.seq_len > 1:
            num_features = X.shape[1] // self.config.seq_len
            if X.shape[1] % self.config.seq_len == 0:
                X = X.reshape(-1, self.config.seq_len, num_features)
                logger.warning("Auto-reshaped X to 3D for LSTM input.")
            else:
                raise ValueError(f"Cannot reshape X of shape {X.shape} to (batch, seq_len, features)")

        if X.ndim != 3:
            raise ValueError(f"Expected X to be 3D (batch, seq_len, features), got shape {X.shape}")

        if len(X) == 0 or len(y) == 0:
            raise ValueError("No data available for training after preprocessing.")

        return X, y

    def _process_data(
        self, data: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features, labels = [], []
        feature_cols = [col for col in data.columns if col not in ["timestamp", "symbol"]]

        for i in range(len(data) - self.config.seq_len):
            window = data.iloc[i: i + self.config.seq_len][feature_cols].values
            # Normalize each window relative to its first row
            seq = (window - window[0]) / (np.abs(window[0]) + 1e-8)
            if np.random.rand() < 0.5:
                seq += np.random.normal(0, 0.01, seq.shape)

            patterns = CandlestickPatterns.detect_patterns(data.iloc[i: i + self.config.seq_len])
            label = self._encode_patterns(patterns, self.selected_patterns)
            if np.any(label):
                features.append(seq)
                labels.append(label)
        return features, labels

    def _encode_patterns(
        self,
        patterns: List[str],
        selected_patterns: List[str]
    ) -> np.ndarray:
        """One-hot encode detected patterns."""
        label = np.zeros(len(selected_patterns), dtype=float)
        for idx, pattern in enumerate(selected_patterns):
            if pattern in patterns:
                label[idx] = 1
        return label

    def train_model(
        self,
        data: pd.DataFrame,
        model: Optional[PatternNN] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[PatternNN, Dict[str, Any]]:
        logger.info("Starting model training")

        # Reproducibility
        seed = getattr(self.config, 'seed', None)
        if seed is not None:
            set_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X, y = self.prepare_training_data(data)

        # Time-based train/validation split
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Feature scaling based on training data
        _, seq_len, feat_dim = X_train.shape
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, feat_dim)
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_val_flat = X_val.reshape(-1, feat_dim)
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)

        # Instantiate model with correct feature dimension
        model = model or PatternNN(input_size=feat_dim, output_size=len(self.selected_patterns))
        model.to(device)

        # Create data loaders
        train_loader = self._create_data_loader(X_train_scaled, y_train, shuffle=True)
        val_loader   = self._create_data_loader(X_val_scaled,   y_val,   shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        loss_fn   = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(self.config.epochs):
            # Training step
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                if outputs is None or batch_y is None:
                    logger.error(f"Model output or batch_y is None. batch_X shape: {batch_X.shape}, batch_y: {batch_y}")
                    raise ValueError("Model output or batch_y is None. Check your data and model forward method.")
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_loader.dataset)

            # Validation step
            model.eval()
            val_loss = 0.0
            all_y_true, all_y_pred = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = loss_fn(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int)
                    all_y_pred.extend(preds)
                    all_y_true.extend(batch_y.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)

            logger.info(f"Epoch {epoch+1}/{self.config.epochs} "
                        f"- Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model, self._evaluate_model(model, val_loader, device)

    def _evaluate_model(self, model: PatternNN, val_loader: DataLoader, device) -> Dict[str, Any]:
        """Evaluate multi-label metrics on validation set."""
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int)
                y_pred.extend(preds)
                y_true.extend(batch_y.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Multi-label metrics
        subset_acc = accuracy_score(y_true, y_pred)
        ham_loss   = hamming_loss(y_true, y_pred)
        precision  = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall     = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1         = f1_score(y_true, y_pred, average='macro', zero_division=0)
        report     = classification_report(y_true, y_pred, output_dict=True)

        return {
            "subset_accuracy": subset_acc,
            "hamming_loss": ham_loss,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "classification_report": report
        }

    def _create_data_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool
    ) -> DataLoader:
        """Builds a DataLoader with optional shuffling."""
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32)
        dataset  = TensorDataset(tensor_x, tensor_y)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available()
        )


def train_pattern_model(
    symbols: List[str],
    data: pd.DataFrame,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    selected_patterns: List[str],
) -> Tuple[Any, Dict[str, Any]]:
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
    return trainer.train_model(data)