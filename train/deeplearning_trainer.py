from utils.logger import setup_logger
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

from train.deeplearning_config import TrainingConfig
from patterns.patterns_nn import PatternNN
from patterns.patterns import CandlestickPatterns
from train.model_manager import ModelManager, ModelMetadata

logger = setup_logger(__name__)

@dataclass
class TrainingConfig:
    """Configuration parameters for pattern model training."""
    epochs: int = 10
    seq_len: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    min_patterns: int = 20
    max_samples_per_symbol: int = 10000

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
        data = self._add_candlestick_pattern_features(data)
        self._last_df = data.copy()

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
        self, data: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features, labels = [], []
        feature_cols = [col for col in data.columns if col not in ["timestamp", "symbol"]]

        for i in range(len(data) - self.config.seq_len):
            window = data.iloc[i:i + self.config.seq_len][feature_cols].values
            seq = (window - window[0]) / (np.abs(window[0]) + 1e-8)
            if np.random.rand() < 0.5:
                seq += np.random.normal(0, 0.01, seq.shape)
            patterns = CandlestickPatterns.detect_patterns(data.iloc[i:i + self.config.seq_len])
            label = self._encode_patterns(patterns, self.selected_patterns)
            if np.any(label):
                features.append(seq.flatten())
                labels.append(label)
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
        logger.info("Starting model training")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X, y = self.prepare_training_data(data)

        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("Expected 2D input arrays for X and y.")

        X_train, X_val, y_train, y_val = self._train_test_split(X, y)

        model = model or PatternNN(input_size=X.shape[1], output_size=len(self.selected_patterns))
        model.to(device)

        train_loader = self._create_data_loader(X_train, y_train)
        val_loader = self._create_data_loader(X_val, y_val)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        best_val_loss, epochs_no_improve, best_model_state = float('inf'), 0, None

        for epoch in range(self.config.epochs):
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = loss_fn(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    preds = torch.sigmoid(outputs).cpu().numpy()
                    y_pred.extend((preds > 0.5).astype(int))
                    y_true.extend(batch_y.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)

            logger.info(f"Epoch {epoch+1}/{self.config.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)

        return model, self._evaluate_model(model, val_loader, device)

    def _evaluate_model(self, model, val_loader, device) -> Dict[str, float]:
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                preds = torch.sigmoid(outputs).cpu().numpy()
                y_pred.extend((preds > 0.5).astype(int))
                y_true.extend(batch_y.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = np.mean(np.all(y_true == y_pred, axis=1))
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
        report = classification_report(y_true.argmax(axis=1), y_pred.argmax(axis=1), output_dict=False)

        return {
            "metrics": {
                "mean": {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1},
                "std": {},
                "final_metrics": {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }

            # Compose ModelMetadata
            model_metadata = ModelMetadata(
                version=datetime.now().strftime("%Y%m%d_%H%M%S"),
                saved_at=datetime.now().isoformat(),
                accuracy=metrics["metrics"]["final_metrics"]["accuracy"],
                parameters=self.config.__dict__,
                framework_version=torch.__version__
            )
            logger.info("Model training and saving completed.")
            return model, metrics

        except Exception as e:
            logger.error("Training failed", exc_info=True)
            raise ValueError(f"Model training failed: {e}") from e

    def _train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        split_idx = int(len(X) * (1 - self.config.validation_split))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    def _create_data_loader(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> DataLoader:
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32)
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )

    def _save_model(
        self,
        model: PatternNN,
        metadata: Optional[ModelMetadata] = None
    ) -> None:
        self.model_manager.save_model(model=model, metadata=metadata, backend="Deep Learning (PatternNN)")

def train_pattern_model(
    symbols: List[str],
    data: pd.DataFrame,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    selected_patterns: List[str],
) -> Tuple[Any, Dict[str, float]]:
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
