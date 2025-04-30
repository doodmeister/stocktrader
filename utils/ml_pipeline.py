# ml_pipeline.py
"""
Machine Learning Pipeline for Candlestick Pattern Recognition
"""
import logging
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from utils.etrade_candlestick_bot import ETradeClient
from models.patterns_nn import PatternNN

from utils.performance_utils import get_candles_cached
from utils.model_manager import save_model
from config import MLConfig
from utils.notifier import Notifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        logger.info(f"Using device: {self.device}")

    def prepare_dataset(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare training and validation datasets with error handling and validation.
        """
        try:
            X, y = [], []
            
            for symbol in self.config.symbols:
                logger.info(f"Fetching data for {symbol}")
                try:
                    df = self.client.get_candles(
                        symbol,
                        interval="5min",
                        days=5
                    )
                    if df.empty:
                        logger.warning(f"No data received for {symbol}")
                        continue
                        
                    values = df[['open','high','low','close','volume']].values
                    if len(values) < self.config.seq_len:
                        logger.warning(
                            f"Insufficient data points for {symbol}: "
                            f"{len(values)} < {self.config.seq_len}"
                        )
                        continue

                    # Normalize features
                    values = self._normalize_features(values)
                    
                    for i in range(self.config.seq_len, len(values)):
                        seq = values[i-self.config.seq_len:i]
                        label = self._extract_pattern_label(seq)
                        X.append(seq)
                        y.append(label)
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    continue

            if not X or not y:
                raise ValueError("No valid sequences could be generated")

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.int64)

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
            self._save_artifacts(model, metrics)
            
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
            error_msg = f"Training pipeline failed: {str(e)}"
            logger.error(error_msg)
            if self.notifier:
                self.notifier.send_message(error_msg, level="error")
            raise

    def _normalize_features(self, values: np.ndarray) -> np.ndarray:
        """Normalize OHLCV data using min-max scaling"""
        min_vals = values.min(axis=0)
        max_vals = values.max(axis=0)
        return (values - min_vals) / (max_vals - min_vals + 1e-8)

    def _extract_pattern_label(self, sequence: np.ndarray) -> int:
        """Extract candlestick pattern label"""
        # TODO: Implement pattern detection logic
        return 0

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
        metrics: Dict[str, float]
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.config.model_dir / f"pattern_nn_{timestamp}.pth"
        metrics_path = self.config.model_dir / f"metrics_{timestamp}.json"
        
        save_model(
            model,
            model_path,
            metadata={
                'accuracy': metrics['accuracy'],
                'epochs': self.config.epochs,
                'seq_len': self.config.seq_len
            }
        )
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    # Environment variables or config
    client = ETradeClient(
        consumer_key=os.getenv('ETRADE_CONSUMER_KEY', ''),
        consumer_secret=os.getenv('ETRADE_CONSUMER_SECRET', ''),
        oauth_token=os.getenv('ETRADE_OAUTH_TOKEN', ''),
        oauth_token_secret=os.getenv('ETRADE_OAUTH_TOKEN_SECRET', ''),
        account_id=os.getenv('ETRADE_ACCOUNT_ID', ''),
        sandbox=True
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
    model = PatternNN()
    metrics = pipeline.train_and_evaluate(model)
    print("Training complete. Metrics:", metrics)
