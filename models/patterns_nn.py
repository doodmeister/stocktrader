"""
models/pattern_nn.py

Placeholder neural network class for candlestick pattern classification.
Extend this with PyTorch layers and logic as needed.
"""

import torch.nn as nn

class PatternNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, output_size=13):
        super(PatternNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)