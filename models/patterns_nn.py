"""
models/pattern_nn.py

Neural network for multi-label candlestick pattern classification.
"""

import torch.nn as nn

class PatternNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=13, dropout=0.2):
        """
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Units in hidden layer.
            output_size (int): Number of pattern classes.
            dropout (float): Dropout rate for regularization.
        """
        super(PatternNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)  # No Sigmoid here
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size)
        Returns:
            Tensor: Raw logits (apply Sigmoid externally if needed)
        """
        return self.model(x)
