# ml_pipeline.py
"""
Machine Learning Pipeline for Candlestick Pattern Recognition
"""
from utils.logger import setup_logger
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import pandas as pd
from utils.etrade_candlestick_bot import ETradeClient
from patterns.patterns_nn import PatternNN
from train.model_manager import ModelManager, ModelMetadata
from train.ml_config import MLConfig
from utils.notifier import Notifier
from utils.technicals.performance_utils import get_candles_cached
from patterns.pattern_utils import get_pattern_names, get_pattern_method
from utils.technicals.feature_engineering import compute_technical_features
from utils.security import get_api_credentials
from train.feature_engineering import add_candlestick_pattern_features

# Configure logging
logger = setup_logger(__name__)

class MLPipeline:
    def __init__(
        self,
        client: ETradeClient,
        config: MLConfig,
        notifier: Optional[Notifier] = None
    ):
        self.client = client
        self.config = config
        self.notifier = notifier
        self.device = torch.device(
            config.device if torch.cuda.is_available() and config.device == "cuda"
            else "cpu"
        )
        # Initialize ModelManager once here
        self.model_manager = ModelManager(base_directory=str(self.config.model_dir))
        logger.info(f"Using device: {self.device}")

    def prepare_dataset(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare training and validation datasets with error handling and validation.
        """
        try:
            X, y = [], []
            all_data = []

            for symbol in self.config.symbols:
                logger.info(f"Fetching data for {symbol}")
                df = self.client.get_candles(
                    symbol,
                    interval="5min",
                    days=5
                )
                if df.empty:
                    logger.warning(f"No data received for {symbol}")
                    continue

                all_data.append(df)

            if not all_data:
                raise ValueError("No data fetched for any symbol.")

            # Concatenate and apply feature engineering
            df = pd.concat(all_data, ignore_index=True)
            df = compute_technical_features(df)  # <-- Enrich features here

            # Continue with your pattern feature engineering and dataset creation
            df = add_candlestick_pattern_features(df)
            self._last_df = df.copy()  # Save for preprocessing config

            # Dynamically select all feature columns except symbol/timestamp
            feature_cols = [col for col in df.columns if col not in ['symbol', 'timestamp'] and df[col].dtype in [np.float64, np.float32]]
            values = df[feature_cols].values

            if len(values) < self.config.seq_len:
                logger.warning(
                    f"Insufficient data points: {len(values)} < {self.config.seq_len}"
                )
                raise ValueError("Not enough data for sequence length.")

            # Normalize features
            values = self._normalize_features(values)

            for i in range(self.config.seq_len, len(values)):
                seq = values[i-self.config.seq_len:i]
                label = self._extract_pattern_label(seq)
                X.append(seq)
                y.append(label)

            if not X or not y:
                raise ValueError("No valid sequences could be generated")

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.int64)

            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                logger.error("X or y is not a numpy array after processing.")
                raise ValueError("Feature or label array is not a numpy array.")
            if X.ndim != 3:
                logger.error(f"Expected X to be 3D (batch, seq_len, features), got shape {X.shape}")
                raise ValueError(f"Expected X to be 3D (batch, seq_len, features), got shape {X.shape}")
            if y.ndim != 1 and y.ndim != 2:
                logger.error(f"Expected y to be 1D or 2D, got shape {y.shape}")
                raise ValueError(f"Expected y to be 1D or 2D, got shape {y.shape}")

            # Split and convert to tensors
            splits = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            return tuple(torch.from_numpy(arr) for arr in splits)

        except Exception as e:
            logger.error(f"Dataset preparation failed: {str(e)}")
            raise

    def train_and_evaluate(
        self,
        model: PatternNN
    ) -> Dict[str, float]:
        """
        Train model with proper error handling and performance monitoring.
        """
        try:
            start_time = datetime.now()
            logger.info("Starting training pipeline")

            # Prepare data
            X_train, X_val, y_train, y_val = self.prepare_dataset()

            if len(X_train) == 0:
                raise RuntimeError("No training data prepared.")
            
            # Setup training
            model = model.to(self.device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate
            )
            criterion = torch.nn.CrossEntropyLoss()
            
            # Training loop with early stopping
            best_loss = float('inf')
            patience = 3
            patience_counter = 0
            
            model.train()
            for epoch in range(1, self.config.epochs + 1):
                epoch_loss = self._train_epoch(
                    model, X_train, y_train, optimizer, criterion
                )
                
                # Early stopping check
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            # Evaluation
            metrics = self._evaluate_model(model, X_val, y_val)
            
            # Save artifacts
            self._save_artifacts(model, metrics, optimizer=optimizer, loss=best_loss)
            
            duration = datetime.now() - start_time
            logger.info(
                f"Training completed in {duration}. "
                f"Accuracy: {metrics['accuracy']:.4f}"
            )
            
            if self.notifier:
                self.notifier.send_message(
                    f"Training completed successfully. "
                    f"Accuracy: {metrics['accuracy']:.4f}"
                )
                
            return metrics

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            if self.notifier:
                self.notifier.send_message(f"Training pipeline failed: {e}", level="error")
            raise

    def _normalize_features(self, values: np.ndarray) -> np.ndarray:
        """Normalize OHLCV data using min-max scaling and save normalization config."""
        min_vals = values.min(axis=0)
        max_vals = values.max(axis=0)
        normed = (values - min_vals) / (max_vals - min_vals + 1e-8)
        # Save normalization config for reproducibility
        self._last_min_vals = min_vals
        self._last_max_vals = max_vals
        return normed

    def _extract_pattern_label(self, sequence: np.ndarray) -> int:
        """
        Extract label from sequence based on last bar's pattern columns.
        Returns:
            0 = hold, 1 = buy, 2 = sell
        """
        # Use self._last_df to get the corresponding unnormalized pattern indicators
        if not hasattr(self, "_last_df"):
            return 0

        # Find the index of the last row in the current sequence within the original DataFrame
        seq_len = sequence.shape[0]
        df = self._last_df
        if len(df) < seq_len:
            return 0
        latest_row = df.iloc[len(df) - (len(self._last_df) - seq_len + 1)]

        bullish = [col for col in df.columns if 'Bullish' in col]
        bearish = [col for col in df.columns if 'Bearish' in col]

        if any(latest_row[b] == 1 for b in bullish):
            return 1  # Buy
        elif any(latest_row[b] == 1 for b in bearish):
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
        total_loss = 0
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
            
        return total_loss / (X_train.size(0) // self.config.batch_size)

    def _evaluate_model(
        self,
        model: PatternNN,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            logits = model(X_val.to(self.device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            acc = accuracy_score(y_val.numpy(), preds)
            cm = confusion_matrix(y_val.numpy(), preds)
            
        return {
            'accuracy': float(acc),
            'confusion_matrix': cm.tolist()
        }

    def _save_artifacts(
        self,
        model: PatternNN,
        metrics: Dict[str, float],
        optimizer=None,
        loss=None
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = self.config.model_dir / f"metrics_{timestamp}.json"

        # Build ModelMetadata object
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

        self.model_manager.save_model(
            model=model,
            metadata=metadata,
            optimizer=optimizer,
            epoch=self.config.epochs,
            loss=loss
        )

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # --- Save preprocessing config (feature order and normalization) ---
        try:
            # Assume last used DataFrame is available as self._last_df
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
            with open(self.config.model_dir / f"preprocessing_{timestamp}.json", 'w') as f:
                json.dump(preprocessing, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save preprocessing config: {e}")

def add_candlestick_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each registered candlestick pattern, add a binary column to the DataFrame
    indicating whether the pattern is detected at each row (using a rolling window).
    Uses pandas' rolling().apply() for efficiency.
    """
    pattern_names = get_pattern_names()
    for pattern in pattern_names:
        method = get_pattern_method(pattern)
        min_rows = 3  # Default window size
        try:
            from patterns.patterns import CandlestickPatterns
            for name, _, mr in CandlestickPatterns._PATTERNS:
                if name == pattern:
                    min_rows = mr
                    break
        except Exception:
            pass

        # Use rolling().apply() for batch processing
        def safe_method(window):
            try:
                return int(method(window)) if method else 0
            except Exception:
                return 0

        # rolling().apply() returns NaN for the first (min_rows-1) rows, so fill with 0
        df[pattern.replace(" ", "")] = (
            df.rolling(window=min_rows, min_periods=min_rows)
              .apply(safe_method, raw=False)
              .fillna(0)
              .astype(int)
              .reset_index(drop=True)  # <-- Ensure index matches DataFrame
        )
    return df

if __name__ == '__main__':
    # Environment variables or config
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
    df_sample = pd.concat([client.get_candles(symbol, interval="5min", days=5) for symbol in config.symbols], ignore_index=True)
    df_sample = compute_technical_features(df_sample)
    df_sample = add_candlestick_pattern_features(df_sample)
    feature_cols = [col for col in df_sample.columns if col not in ['symbol', 'timestamp']]
    feature_count = df_sample[feature_cols].shape[1]

    model = PatternNN(
        input_size=feature_count,  # count of all feature columns
        hidden_size=64,
        num_layers=2,
        output_size=3,
        dropout=0.2
    )
    metrics = pipeline.train_and_evaluate(model)
    print("Training complete. Metrics:", metrics)
