"""
Training Manager for ML model training operations.
Uses existing train/ infrastructure: ModelManager and MLPipeline.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from train.model_manager import ModelManager
from train.model_training_pipeline import MLPipeline
from train.ml_config import MLConfig
from train.deeplearning_config import TrainingConfig
from train.deeplearning_trainer import PatternModelTrainer
from patterns.patterns_nn import PatternNN
from patterns.patterns import CandlestickPatterns
from utils.logger import setup_logger
from utils.notifier import Notifier

logger = setup_logger(__name__)

@dataclass
class TrainingResult:
    """Result of a training operation."""
    success: bool
    model: Optional[Any] = None
    metrics: Optional[Dict] = None
    error_message: Optional[str] = None
    data_points: int = 0
    backend: str = ""
    model_path: Optional[str] = None

class TrainingManager:
    """
    Manages ML model training operations using existing train/ infrastructure.
    Uses ModelManager for model persistence and MLPipeline for training workflow.
    """
    
    def __init__(self, etrade_client=None, model_dir: str = "models/"):
        self.etrade_client = etrade_client
        self.model_manager = ModelManager(base_directory=model_dir)
        self.model_dir = Path(model_dir)
        self.notifier = None
        
        # Initialize notifier if available
        try:
            self.notifier = Notifier()
        except Exception as e:
            logger.warning(f"Failed to initialize notifier: {e}")
    
    def train_pattern_model_with_pipeline(self, 
                                        symbols: List[str],
                                        config: Optional[MLConfig] = None) -> TrainingResult:
        """
        Train a pattern recognition model using the existing MLPipeline infrastructure.
        This uses the complete pipeline from train/model_training_pipeline.py
        """
        try:
            if not self.etrade_client:
                return TrainingResult(
                    success=False,
                    error_message="E*Trade client not available for data collection",
                    backend="MLPipeline"
                )
            
            # Use default config if not provided
            if config is None:
                config = MLConfig(
                    seq_len=10,
                    epochs=20,
                    batch_size=32,
                    learning_rate=0.001,
                    test_size=0.2,
                    random_state=42,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    model_dir=self.model_dir,
                    symbols=symbols
                )
            else:
                # Update symbols in config
                config.symbols = symbols
            
            # Initialize MLPipeline with existing infrastructure
            pipeline = MLPipeline(
                client=self.etrade_client,
                config=config,
                notifier=self.notifier
            )
            
            # Prepare dataset using pipeline's data collection
            logger.info(f"Preparing dataset for symbols: {symbols}")
            X_train, X_val, y_train, y_val = pipeline.prepare_dataset()
            
            data_points = len(X_train) + len(X_val)
            
            if data_points < 100:  # Minimum data requirement
                return TrainingResult(
                    success=False,
                    error_message=f"Insufficient data: {data_points} samples, need at least 100",
                    data_points=data_points,
                    backend="MLPipeline"
                )
            
            # Calculate input size from prepared data
            input_size = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
            
            # Create model with correct dimensions
            model = PatternNN(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                output_size=3,  # hold, buy, sell
                dropout=0.2
            )
            
            # Train using pipeline
            logger.info(f"Starting MLPipeline training with {data_points} data points")
            metrics = pipeline.train_and_evaluate(model)
            
            # The pipeline automatically saves the model, get the path
            model_files = sorted(
                self.model_dir.glob("pattern_nn_v*.pth"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            model_path = str(model_files[0]) if model_files else None
            
            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics,
                data_points=data_points,
                backend="MLPipeline",
                model_path=model_path
            )
            
        except Exception as e:
            logger.error(f"MLPipeline training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                backend="MLPipeline"
            )
    
    def train_pattern_model_direct(self, 
                                  symbols: List[str],
                                  config: Optional[TrainingConfig] = None) -> TrainingResult:
        """
        Train using the direct PatternModelTrainer approach.
        Uses ModelManager for persistence.
        """
        try:
            # Use default config if not provided
            if config is None:
                config = TrainingConfig(
                    epochs=20,
                    seq_len=10,
                    learning_rate=0.001,
                    batch_size=32,
                    validation_split=0.2,
                    early_stopping_patience=5,
                    min_patterns=100
                )
            
            # Initialize trainer using existing infrastructure
            trainer = PatternModelTrainer(
                model_manager=self.model_manager,
                config=config,
                selected_patterns=CandlestickPatterns.get_pattern_names()
            )
            
            # Collect training data
            training_data = self._collect_training_data(symbols)
            if not training_data:
                return TrainingResult(
                    success=False,
                    error_message="No training data available",
                    backend="PatternModelTrainer"
                )
            
            data = pd.concat(training_data, ignore_index=True)
            data_points = len(data)
            
            if data_points < config.min_patterns:
                return TrainingResult(
                    success=False,
                    error_message=f"Insufficient data: {data_points} rows, need {config.min_patterns}",
                    data_points=data_points,
                    backend="PatternModelTrainer"
                )
            
            # Train model using existing trainer
            logger.info(f"Starting PatternModelTrainer training with {data_points} data points")
            model, metrics = trainer.train_model(data)
            
            # Save model using ModelManager
            from train.model_manager import ModelMetadata
            from datetime import datetime
            
            metadata = ModelMetadata(
                version=datetime.now().strftime("%Y%m%d_%H%M%S"),
                saved_at=datetime.now().isoformat(),
                accuracy=metrics.get("f1_macro"),
                parameters={
                    "epochs": config.epochs,
                    "seq_len": config.seq_len,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "patterns": CandlestickPatterns.get_pattern_names()
                },
                backend="PatternNN"
            )
            
            model_path = self.model_manager.save_model(
                model=model,
                metadata=metadata,
                backend="Deep Learning"
            )
            
            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics,
                data_points=data_points,
                backend="PatternModelTrainer",
                model_path=model_path
            )
            
        except Exception as e:
            logger.error(f"PatternModelTrainer training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                backend="PatternModelTrainer"
            )
    
    def train_model(self, 
                   symbols: List[str],
                   backend: str = "MLPipeline",
                   **kwargs) -> TrainingResult:
        """
        Unified training interface that delegates to appropriate trainer.
        
        Args:
            symbols: List of stock symbols for training
            backend: "MLPipeline" or "PatternModelTrainer"
            **kwargs: Additional configuration parameters
        """
        try:
            if backend == "MLPipeline":
                config = kwargs.get('config')
                return self.train_pattern_model_with_pipeline(symbols, config)
            else:
                config = kwargs.get('config')
                return self.train_pattern_model_direct(symbols, config)
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                backend=backend
            )
    
    def load_latest_model(self) -> Optional[Any]:
        """Load the most recent model using ModelManager."""
        try:
            from train.model_manager import load_latest_model
            model = load_latest_model(PatternNN, str(self.model_dir))
            return model
        except Exception as e:
            logger.error(f"Failed to load latest model: {e}")
            return None
    
    def list_available_models(self) -> List[str]:
        """List all available models using ModelManager."""
        try:
            return self.model_manager.list_models()
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def _collect_training_data(self, symbols: List[str]) -> List[pd.DataFrame]:
        """Collect training data for the given symbols."""
        training_data = []
        
        if not self.etrade_client:
            logger.warning("No E*Trade client available for data collection")
            return training_data
        
        for symbol in symbols:
            try:
                # Fetch historical data for the symbol
                data = self.etrade_client.get_candles(symbol, interval='1d', days=252)  # 1 year
                if data is not None and not data.empty:
                    # Add symbol column for multi-symbol training
                    data['symbol'] = symbol
                    training_data.append(data)
                    logger.info(f"Collected {len(data)} data points for {symbol}")
                else:
                    logger.warning(f"No data available for symbol: {symbol}")
            except Exception as e:
                logger.error(f"Failed to collect data for {symbol}: {e}")
        
        return training_data