"""
Production-Grade Deep Learning Configuration Module

Provides comprehensive configuration management for deep learning training
with validation, type safety, and best practices for neural network training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch


class ConfigurationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


@dataclass
class TrainingConfig:
    """
    Comprehensive configuration for deep learning model training.
    
    This configuration class provides validated parameters for neural network
    training with sensible defaults and comprehensive validation.
    
    Attributes:
        epochs: Number of training epochs
        seq_len: Sequence length for time series data
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
        early_stopping_patience: Epochs to wait before early stopping
        min_patterns: Minimum number of patterns required for training
        device: Training device (auto-detected if None)
        model_save_dir: Directory to save trained models
        checkpoint_frequency: Save checkpoint every N epochs
        gradient_clip_value: Gradient clipping threshold
        weight_decay: L2 regularization coefficient
        scheduler_factor: Learning rate reduction factor
        scheduler_patience: Epochs to wait before reducing LR
        dropout_rate: Dropout rate for regularization
        hidden_layers: List of hidden layer sizes
        activation_function: Activation function name
        optimizer_type: Optimizer type (adam, sgd, adamw)
        loss_function: Loss function name
        metrics_to_track: List of metrics to track during training
    """
    
    # Core training parameters
    epochs: int = 10
    seq_len: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    min_patterns: int = 100
    
    # Device and I/O configuration
    device: Optional[str] = None
    model_save_dir: Path = field(default_factory=lambda: Path("models"))
    checkpoint_frequency: int = 5
    
    # Advanced training parameters
    gradient_clip_value: Optional[float] = 1.0
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    dropout_rate: float = 0.2
    
    # Model architecture parameters
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    activation_function: str = "relu"
    optimizer_type: str = "adam"
    loss_function: str = "cross_entropy"
    
    # Monitoring and evaluation
    metrics_to_track: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1_score"
    ])
    
    # Data preprocessing
    normalize_features: bool = True
    feature_selection_threshold: float = 0.01
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_configuration()
        self._setup_device()
        self._create_directories()
    
    def _validate_configuration(self) -> None:
        """Validate all configuration parameters."""
        # Validate epochs
        if self.epochs <= 0:
            raise ConfigurationError("epochs must be positive")
        if self.epochs > 10000:
            raise ConfigurationError("epochs too large (max 10000)")
        
        # Validate sequence length
        if self.seq_len <= 0:
            raise ConfigurationError("seq_len must be positive")
        if self.seq_len > 1000:
            raise ConfigurationError("seq_len too large (max 1000)")
        
        # Validate learning rate
        if not (1e-6 <= self.learning_rate <= 1.0):
            raise ConfigurationError("learning_rate must be between 1e-6 and 1.0")
        
        # Validate batch size
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")
        if self.batch_size > 10000:
            raise ConfigurationError("batch_size too large (max 10000)")
        
        # Validate validation split
        if not (0.1 <= self.validation_split <= 0.5):
            raise ConfigurationError("validation_split must be between 0.1 and 0.5")
        
        # Validate early stopping patience
        if self.early_stopping_patience <= 0:
            raise ConfigurationError("early_stopping_patience must be positive")
        
        # Validate minimum patterns
        if self.min_patterns <= 0:
            raise ConfigurationError("min_patterns must be positive")
        
        # Validate dropout rate
        if not (0.0 <= self.dropout_rate <= 0.9):
            raise ConfigurationError("dropout_rate must be between 0.0 and 0.9")
        
        # Validate optimizer type
        valid_optimizers = {"adam", "sgd", "adamw", "rmsprop"}
        if self.optimizer_type.lower() not in valid_optimizers:
            raise ConfigurationError(f"optimizer_type must be one of {valid_optimizers}")
        
        # Validate activation function
        valid_activations = {"relu", "tanh", "sigmoid", "leaky_relu", "gelu"}
        if self.activation_function.lower() not in valid_activations:
            raise ConfigurationError(f"activation_function must be one of {valid_activations}")
        
        # Validate loss function
        valid_losses = {"cross_entropy", "mse", "mae", "binary_cross_entropy"}
        if self.loss_function.lower() not in valid_losses:
            raise ConfigurationError(f"loss_function must be one of {valid_losses}")
    
    def _setup_device(self) -> None:
        """Auto-detect and setup training device."""
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"
        
        # Validate device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError("CUDA requested but not available")
        
        if self.device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise ConfigurationError("MPS requested but not available")
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        self.model_save_dir = Path(self.model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        # Convert string paths back to Path objects
        if 'model_save_dir' in config_dict:
            config_dict['model_save_dir'] = Path(config_dict['model_save_dir'])
        
        return cls(**config_dict)
    
    def get_optimizer_params(self) -> Dict[str, Any]:
        """Get optimizer parameters."""
        return {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay
        }
    
    def get_scheduler_params(self) -> Dict[str, Any]:
        """Get learning rate scheduler parameters."""
        return {
            "factor": self.scheduler_factor,
            "patience": self.scheduler_patience,
            "verbose": True
        }