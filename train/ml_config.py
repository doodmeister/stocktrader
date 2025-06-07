"""
Production-Grade ML Configuration Module

Provides comprehensive configuration management for classical machine learning
training with validation, security best practices, and robust error handling.
"""

from pathlib import Path
from typing import Dict, Any
import os


class MLConfigError(Exception):
    """Exception raised for ML configuration errors."""
    pass


class MLConfig:
    """
    Comprehensive configuration for classical ML training pipeline.
    
    This configuration class manages all parameters needed for classical machine
    learning workflows, including E*TRADE integration, model parameters, and
    risk management settings.
    
    Security Note:
        Sensitive credentials are loaded from environment variables.
        Never hardcode credentials in source code.
    """
    
    def __init__(self, **kwargs):
        """Initialize configuration with default values and environment loading."""
        
        # --- Platform/Integration Settings ---
        self.ETRADE_CONSUMER_KEY = kwargs.get('ETRADE_CONSUMER_KEY', os.environ.get('ETRADE_CONSUMER_KEY', ''))
        self.ETRADE_CONSUMER_SECRET = kwargs.get('ETRADE_CONSUMER_SECRET', os.environ.get('ETRADE_CONSUMER_SECRET', ''))
        self.ETRADE_OAUTH_TOKEN = kwargs.get('ETRADE_OAUTH_TOKEN', os.environ.get('ETRADE_OAUTH_TOKEN', ''))
        self.ETRADE_OAUTH_TOKEN_SECRET = kwargs.get('ETRADE_OAUTH_TOKEN_SECRET', os.environ.get('ETRADE_OAUTH_TOKEN_SECRET', ''))
        self.ETRADE_ACCOUNT_ID = kwargs.get('ETRADE_ACCOUNT_ID', os.environ.get('ETRADE_ACCOUNT_ID', ''))
        self.ETRADE_USE_SANDBOX = kwargs.get('ETRADE_USE_SANDBOX', True)

        # --- Email/Notification Settings ---
        self.smtp_port = kwargs.get('smtp_port', '587')
        
        # --- Risk Management Parameters ---
        self.max_positions = kwargs.get('max_positions', 5)
        self.max_loss_percent = kwargs.get('max_loss_percent', 0.02)
        self.profit_target_percent = kwargs.get('profit_target_percent', 0.03)
        self.max_daily_loss = kwargs.get('max_daily_loss', 0.05)

        # --- Deep Learning Parameters (for compatibility) ---
        self.seq_len = kwargs.get('seq_len', 10)
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.device = kwargs.get('device', 'cpu')

        # --- Shared/Data/General Parameters ---
        self.test_size = kwargs.get('test_size', 0.2)
        self.random_state = kwargs.get('random_state', 42)
        self.model_dir = Path(kwargs.get('model_dir', 'models'))
        self.symbols = kwargs.get('symbols', ['AAPL'])

        # --- Classical ML Parameters ---
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.max_depth = kwargs.get('max_depth', 10)
        self.min_samples_split = kwargs.get('min_samples_split', 2)
        self.cv_folds = kwargs.get('cv_folds', 3)
        
        # --- Advanced ML Parameters ---
        self.feature_selection_method = kwargs.get('feature_selection_method', 'importance')
        self.max_features = kwargs.get('max_features', None)
        self.validation_strategy = kwargs.get('validation_strategy', 'time_series')
        self.scoring_metric = kwargs.get('scoring_metric', 'accuracy')
        
        # --- Data Processing Parameters ---
        self.min_data_points = kwargs.get('min_data_points', 1000)
        self.outlier_detection = kwargs.get('outlier_detection', True)
        self.feature_scaling = kwargs.get('feature_scaling', 'standard')
        
        # Create model directory
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate risk parameters
        if not (0 < self.max_loss_percent <= 0.5):
            raise MLConfigError("max_loss_percent must be between 0 and 0.5")
        if not (0 < self.profit_target_percent <= 1.0):
            raise MLConfigError("profit_target_percent must be between 0 and 1.0")
        if not (0 < self.max_daily_loss <= 1.0):
            raise MLConfigError("max_daily_loss must be between 0 and 1.0")
        
        # Validate symbols
        if not self.symbols or not isinstance(self.symbols, list):
            raise MLConfigError("At least one symbol must be specified")
        
        # Validate device
        if self.device not in {'cpu', 'cuda', 'mps', 'auto'}:
            raise MLConfigError("device must be one of: cpu, cuda, mps, auto")

    def get_risk_params(self) -> Dict[str, float]:
        """Get risk management parameters as float values."""
        return {
            "max_loss_percent": self.max_loss_percent,
            "profit_target_percent": self.profit_target_percent, 
            "max_daily_loss": self.max_daily_loss,
            "max_positions": self.max_positions
        }

    def get_ml_params(self) -> Dict[str, Any]:
        """Get classical ML parameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
            "max_features": self.max_features
        }

    def get_training_params(self) -> Dict[str, Any]:
        """Get training configuration parameters."""
        return {
            "test_size": self.test_size,
            "cv_folds": self.cv_folds,
            "validation_strategy": self.validation_strategy,
            "scoring_metric": self.scoring_metric,
            "feature_scaling": self.feature_scaling,
            "feature_selection_method": self.feature_selection_method
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excluding sensitive data)."""
        config_dict = {}
        
        # Add all attributes except sensitive ones
        for key, value in self.__dict__.items():
            if key.startswith('ETRADE_') and key.endswith(('KEY', 'SECRET', 'TOKEN')):
                config_dict[key] = "***REDACTED***"
            elif isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        
        return config_dict

    def __repr__(self):
        """String representation of the configuration."""
        return f"MLConfig(symbols={self.symbols}, device={self.device}, test_size={self.test_size})"