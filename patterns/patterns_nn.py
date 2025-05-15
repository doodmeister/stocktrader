"""
models/pattern_nn.py

Neural network for multi-label candlestick pattern classification.
"""

import torch
import torch.nn as nn

class PatternNN(nn.Module):
    """Neural network for candlestick pattern classification and trading signals.
    
    Features:
    - LSTM for sequential data processing
    - Configurable number of hidden layers
    - Dropout for regularization
    
    Args:
        input_size (int): Number of input features (default: 5)
        hidden_size (int): Units in hidden layers (default: 64)
        num_layers (int): Number of LSTM layers (default: 2)
        output_size (int): Number of output classes (default: 3 for HOLD/BUY/SELL)
        dropout (float): Dropout rate (default: 0.2)
    """
    def __init__(
        self,
        input_size: int = 5,        # number of input features (OHLCV + patterns)
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 3,       # e.g., buy / sell / hold
        dropout: float = 0.2
    ):
        """
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Units in hidden layer.
            num_layers (int): Number of LSTM layers.
            output_size (int): Number of pattern classes.
            dropout (float): Dropout rate for regularization.
        """
        super(PatternNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_size)
        """
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]  # take the last
