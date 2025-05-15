"""
Machine Learning Pipeline for Candlestick Pattern Recognition
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from utils.logger import setup_logger
from utils.etrade_candlestick_bot import ETradeClient
from utils.notifier import Notifier
from utils.technicals.performance_utils import get_candles_cached
from patterns.pattern_utils import get_pattern_names, get_pattern_method
from train.model_manager import ModelManager, ModelMetadata
from train.ml_config import MLConfig
from train.deeplearning_trainer import PatternModelTrainer, TrainingConfig
from patterns.patterns_nn import PatternNN

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
        self.model_manager = ModelManager(base_directory=str(self.config.model_dir))

    def train_and_evaluate(self) -> Dict[str, float]:
        try:
            logger.info("Starting training pipeline")
            df = self._collect_data()
            if df.empty:
                raise ValueError("No data collected for training")

            selected_patterns = get_pattern_names()
            trainer = PatternModelTrainer(
                model_manager=self.model_manager,
                config=TrainingConfig(
                    epochs=self.config.epochs,
                    seq_len=self.config.seq_len,
                    learning_rate=self.config.learning_rate,
                    batch_size=self.config.batch_size
                ),
                selected_patterns=selected_patterns
            )

            model, metrics = trainer.train_model(df)
            logger.info(f"Training complete. Accuracy: {metrics['metrics']['final_metrics']['accuracy']:.4f}")

            if self.notifier:
                self.notifier.send_message(
                    f"Training completed. Accuracy: {metrics['metrics']['final_metrics']['accuracy']:.4f}"
                )

            return metrics

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            if self.notifier:
                self.notifier.send_message(f"Training failed: {e}", level="error")
            raise

    def _collect_data(self) -> pd.DataFrame:
        all_data = []
        for symbol in self.config.symbols:
            logger.info(f"Fetching data for {symbol}")
            df = self.client.get_candles(symbol, interval="5min", days=5)
            if df.empty:
                logger.warning(f"No data received for {symbol}")
                continue
            df['symbol'] = symbol
            all_data.append(df)
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def add_candlestick_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    pattern_names = get_pattern_names()
    for pattern in pattern_names:
        method = get_pattern_method(pattern)
        min_rows = 3
        try:
            from patterns.patterns import CandlestickPatterns
            for name, _, mr in CandlestickPatterns._PATTERNS:
                if name == pattern:
                    min_rows = mr
                    break
        except Exception:
            pass

        results = []
        for i in range(len(df)):
            if i + 1 < min_rows:
                results.append(0)
                continue
            window = df.iloc[i + 1 - min_rows : i + 1]
            try:
                detected = int(method(window)) if method else 0
            except Exception:
                detected = 0
            results.append(detected)
        df[pattern.replace(" ", "")] = results
    return df

if __name__ == '__main__':
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
    metrics = pipeline.train_and_evaluate()
    print("Training complete. Metrics:", metrics)
