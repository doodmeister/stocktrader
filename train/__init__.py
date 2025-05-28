"""
Machine Learning Training Module

This module provides comprehensive ML training capabilities for the stocktrader
application, including both classical ML and deep learning approaches for
candlestick pattern recognition and stock market prediction.

Key Components:
- MLPipeline: Complete training pipeline orchestration
- ModelManager: Model persistence and versioning
- FeatureEngineer: Feature engineering and technical indicators
- TrainingConfig: Configuration management
- MLConfig: General ML configuration
- MLTrainer: Classical ML model training
- PatternModelTrainer: Deep learning pattern recognition

Usage:
    from train import MLPipeline, MLConfig, ModelManager
    
    config = MLConfig()
    pipeline = MLPipeline(client, config)
    metrics = pipeline.train_and_evaluate(model)
"""

# Core pipeline and configuration
from .model_training_pipeline import MLPipeline, DatasetPreparationError, TrainingPipelineError
from .ml_config import MLConfig
from .deeplearning_config import TrainingConfig, ConfigurationError

# Model management
from .model_manager import (
    ModelManager, 
    ModelMetadata, 
    ModelManagerConfig,
    ModelType,
    ModelNotFoundError,
    InvalidModelError,
    ModelValidationError
)

# Training modules
from .ml_trainer import ModelTrainer, ModelError, ValidationError, TrainingError
from .deeplearning_trainer import PatternModelTrainer, TrainingMetrics

# Feature engineering
from .feature_engineering import FeatureEngineer

# Version information
__version__ = "2.0.0"
__author__ = "StockTrader Development Team"

# Public API
__all__ = [
    # Core pipeline
    "MLPipeline",
    "DatasetPreparationError", 
    "TrainingPipelineError",
    
    # Configuration
    "MLConfig",
    "TrainingConfig",
    "ConfigurationError",
    
    # Model management
    "ModelManager",
    "ModelMetadata", 
    "ModelManagerConfig",
    "ModelType",
    "ModelNotFoundError",
    "InvalidModelError",
    "ModelValidationError",
    
    # Training
    "ModelTrainer",
    "PatternModelTrainer",
    "TrainingMetrics",
    "ModelError",
    "ValidationError", 
    "TrainingError",
    
    # Feature engineering
    "FeatureEngineer",
    
    # Metadata
    "__version__",
    "__author__"
]