# model_manager.py
"""
Model management utilities for saving, loading, and versioning PyTorch models.

Functions:
    - save_model(model: torch.nn.Module, path: str, metadata: dict)
    - load_model(model_class, path: str) -> torch.nn.Module
    - list_models(directory: str) -> List[str]
"""
import os
import torch
from typing import List, Type


def save_model(model: torch.nn.Module, path: str, metadata: dict = None):
    """
    Save the model parameters and metadata to disk.

    path: e.g. 'models/pattern_nn_v1.pth'
    metadata: optional dict to save training info (json alongside)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'metadata': metadata or {}
    }, path)


def load_model(model_class: Type[torch.nn.Module], path: str) -> torch.nn.Module:
    """
    Instantiate model_class and load state_dict from path.

    model_class: class reference, e.g. PatternNN
    path: filesystem path to .pth file
    """
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model = model_class()
    model.load_state_dict(checkpoint['state_dict'])
    return model


def list_models(directory: str) -> List[str]:
    """
    List all model files in a directory (.pth).
    """
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith('.pth')]
