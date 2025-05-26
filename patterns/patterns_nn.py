"""
patterns/patterns_nn.py

Production-grade neural network for multi-label candlestick pattern classification.
Integrates with the trading bot's ML pipeline and risk management system.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class PatternNNConfig:
    """Configuration class for PatternNN model parameters."""
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 3,
        dropout: float = 0.2,
        sequence_length: int = 20,
        use_batch_norm: bool = True,
        use_residual: bool = False,
        activation: str = "relu"
    ):
        """Initialize configuration with validation.
        
        Args:
            input_size: Number of input features (OHLCV + technical indicators)
            hidden_size: Units in LSTM hidden layers
            num_layers: Number of LSTM layers (1-4 recommended)
            output_size: Number of output classes (3 for BUY/SELL/HOLD)
            dropout: Dropout rate for regularization (0.0-0.5)
            sequence_length: Length of input sequences
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            activation: Activation function ('relu', 'gelu', 'swish')
        """
        self.validate_config(
            input_size, hidden_size, num_layers, output_size, 
            dropout, sequence_length, activation
        )
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.activation = activation
    
    @staticmethod
    def validate_config(
        input_size: int, hidden_size: int, num_layers: int,
        output_size: int, dropout: float, sequence_length: int,
        activation: str
    ) -> None:
        """Validate configuration parameters."""
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0 or num_layers > 4:
            raise ValueError(f"num_layers must be 1-4, got {num_layers}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if not 0.0 <= dropout <= 0.5:
            raise ValueError(f"dropout must be in [0.0, 0.5], got {dropout}")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if activation not in ['relu', 'gelu', 'swish', 'tanh']:
            raise ValueError(f"activation must be one of ['relu', 'gelu', 'swish', 'tanh'], got {activation}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'sequence_length': self.sequence_length,
            'use_batch_norm': self.use_batch_norm,
            'use_residual': self.use_residual,
            'activation': self.activation
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PatternNNConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class PatternNN(nn.Module):
    """Production-grade neural network for candlestick pattern classification.
    
    This model uses LSTM layers to process sequential candlestick data and
    outputs trading signals (BUY/SELL/HOLD). It includes:
    - Robust input validation and error handling
    - Configurable architecture with batch normalization
    - Optional residual connections for deeper networks
    - Comprehensive logging and monitoring
    - Integration with the trading system's risk management
    
    The model is designed to work with the ModelManager for persistence
    and versioning, and integrates with the broader ML pipeline.
    """
    
    def __init__(self, config: Optional[PatternNNConfig] = None):
        """Initialize the PatternNN model.
        
        Args:
            config: Model configuration object. If None, uses default config.
        """
        super(PatternNN, self).__init__()
        
        if config is None:
            config = PatternNNConfig()
        
        self.config = config
        self._model_metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '2.0.0',
            'framework': 'pytorch'
        }
        
        # Initialize layers
        self._build_model()
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"PatternNN initialized with config: {self.config.to_dict()}")
    
    def _build_model(self) -> None:
        """Build the neural network architecture."""
        config = self.config
        
        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Batch normalization for LSTM output
        if config.use_batch_norm:
            self.lstm_bn = nn.BatchNorm1d(config.hidden_size)
        
        # Activation function
        self.activation = self._get_activation_fn(config.activation)
        
        # Classifier head
        classifier_layers = []
        
        # First layer
        classifier_layers.append(nn.Linear(config.hidden_size, config.hidden_size // 2))
        if config.use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(config.hidden_size // 2))
        classifier_layers.append(self.activation)
        classifier_layers.append(nn.Dropout(config.dropout))
        
        # Second layer (optional for deeper networks)
        if config.hidden_size >= 128:
            classifier_layers.append(nn.Linear(config.hidden_size // 2, config.hidden_size // 4))
            if config.use_batch_norm:
                classifier_layers.append(nn.BatchNorm1d(config.hidden_size // 4))
            classifier_layers.append(self.activation)
            classifier_layers.append(nn.Dropout(config.dropout))
            final_input_size = config.hidden_size // 4
        else:
            final_input_size = config.hidden_size // 2
        
        # Output layer
        classifier_layers.append(nn.Linear(final_input_size, config.output_size))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Residual connection (if enabled and dimensions match)
        if config.use_residual and config.hidden_size == config.output_size:
            self.residual_projection = nn.Linear(config.hidden_size, config.output_size)
        else:
            self.residual_projection = None
    
    def _get_activation_fn(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),  # SiLU is Swish
            'tanh': nn.Tanh()
        }
        return activations[activation]
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using best practices."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Xavier uniform for LSTM weights
                    nn.init.xavier_uniform_(param)
                elif 'classifier' in name and param.dim() >= 2:
                    # He initialization for classifier layers
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        logger.debug("Model weights initialized")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
            
        Raises:
            ValueError: If input tensor has incorrect shape or contains invalid values
        """
        # Input validation
        self._validate_input(x)
        
        try:
            # LSTM forward pass
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Take the last hidden state
            last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)
            
            # Apply batch normalization if enabled
            if self.config.use_batch_norm and hasattr(self, 'lstm_bn'):
                last_hidden = self.lstm_bn(last_hidden)
            
            # Store residual for potential skip connection
            residual = last_hidden if self.residual_projection is not None else None
            
            # Classifier forward pass
            output = self.classifier(last_hidden)
            
            # Add residual connection if configured
            if residual is not None:
                output = output + self.residual_projection(residual)
            
            return output
            
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            logger.error(f"Input shape: {x.shape}, dtype: {x.dtype}")
            raise RuntimeError(f"Model forward pass failed: {str(e)}") from e
    
    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        if x.ndim != 3:
            raise ValueError(
                f"Expected input of shape (batch_size, seq_len, input_size), "
                f"got {x.ndim}D tensor with shape {x.shape}"
            )
        
        batch_size, seq_len, input_size = x.shape
        
        if input_size != self.config.input_size:
            raise ValueError(
                f"Expected input_size={self.config.input_size}, "
                f"got {input_size}"
            )
        
        if seq_len != self.config.sequence_length:
            logger.warning(
                f"Input sequence length {seq_len} differs from configured "
                f"sequence length {self.config.sequence_length}"
            )
        
        # Check for NaN or infinite values
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN values")
        
        if torch.isinf(x).any():
            raise ValueError("Input contains infinite values")
        
        # Check for reasonable value ranges (assuming normalized inputs)
        if x.abs().max() > 100:
            logger.warning(
                f"Input values seem unusually large (max abs value: {x.abs().max():.2f}). "
                "Consider checking data normalization."
            )
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities using softmax.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability distribution over classes
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get class predictions.
        
        Args:
            x: Input tensor
            threshold: Confidence threshold for predictions
            
        Returns:
            Class predictions
        """
        probabilities = self.predict_proba(x)
        predictions = torch.argmax(probabilities, dim=-1)
        
        # Apply confidence threshold
        max_probs = torch.max(probabilities, dim=-1)[0]
        low_confidence_mask = max_probs < threshold
        
        if low_confidence_mask.any():
            # Set low confidence predictions to "HOLD" (assuming class 0)
            predictions[low_confidence_mask] = 0
            logger.debug(f"Set {low_confidence_mask.sum()} predictions to HOLD due to low confidence")
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'config': self.config.to_dict(),
            'metadata': self._model_metadata,
            'parameters': {
                'total': total_params,
                'trainable': trainable_params,
                'non_trainable': total_params - trainable_params
            },
            'architecture': str(self),
            'device': next(self.parameters()).device.type if total_params > 0 else 'cpu'
        }
    
    def save_config(self, filepath: Union[str, Path]) -> None:
        """Save model configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            'config': self.config.to_dict(),
            'metadata': self._model_metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Model configuration saved to {filepath}")
    
    @classmethod
    def load_from_config(cls, filepath: Union[str, Path]) -> 'PatternNN':
        """Load model from configuration file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        config = PatternNNConfig.from_dict(config_data['config'])
        model = cls(config)
        
        # Update metadata if present
        if 'metadata' in config_data:
            model._model_metadata.update(config_data['metadata'])
        
        logger.info(f"Model loaded from configuration: {filepath}")
        return model
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"PatternNN(\n"
            f"  input_size={self.config.input_size},\n"
            f"  hidden_size={self.config.hidden_size},\n"
            f"  num_layers={self.config.num_layers},\n"
            f"  output_size={self.config.output_size},\n"
            f"  dropout={self.config.dropout},\n"
            f"  sequence_length={self.config.sequence_length}\n"
            f")"
        )


# Factory functions for common configurations
def create_lightweight_model() -> PatternNN:
    """Create a lightweight model for fast inference."""
    config = PatternNNConfig(
        input_size=5,
        hidden_size=32,
        num_layers=1,
        output_size=3,
        dropout=0.1,
        use_batch_norm=False,
        use_residual=False
    )
    return PatternNN(config)


def create_robust_model() -> PatternNN:
    """Create a robust model for production trading."""
    config = PatternNNConfig(
        input_size=10,  # More technical indicators
        hidden_size=128,
        num_layers=2,
        output_size=3,
        dropout=0.2,
        use_batch_norm=True,
        use_residual=True,
        activation='gelu'
    )
    return PatternNN(config)


def create_deep_model() -> PatternNN:
    """Create a deep model for complex pattern recognition."""
    config = PatternNNConfig(
        input_size=15,  # Rich feature set
        hidden_size=256,
        num_layers=3,
        output_size=3,
        dropout=0.3,
        use_batch_norm=True,
        use_residual=True,
        activation='swish'
    )
    return PatternNN(config)