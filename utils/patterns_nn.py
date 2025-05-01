"""
models/pattern_nn.py

Neural network for multi-label candlestick pattern classification.
"""

import torch.nn as nn

class PatternNN(nn.Module):
    """Neural network for candlestick pattern classification and trading signals.
    
    Features:
    - Configurable number of hidden layers
    - Batch normalization for training stability
    - Dropout for regularization
    - Xavier weight initialization
    
    Args:
        input_size (int): Number of input features (default: 10)
        hidden_size (int): Units in hidden layers (default: 64)
        num_layers (int): Number of hidden layers (default: 2)
        output_size (int): Number of output classes (default: 3 for HOLD/BUY/SELL)
        dropout (float): Dropout rate (default: 0.2)
    """
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, output_size=3, dropout=0.2):
        """
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Units in hidden layer.
            num_layers (int): Number of hidden layers.
            output_size (int): Number of pattern classes.
            dropout (float): Dropout rate for regularization.
        """
        super(PatternNN, self).__init__()
        
        layers = []
        # Input layer
        layers.extend([
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size)
        Returns:
            Tensor: Raw logits (apply Sigmoid externally if needed)
        """
        return self.model(x)
