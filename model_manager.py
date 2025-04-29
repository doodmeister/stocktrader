# model_manager.py
"""
Model management utilities for saving, loading, and versioning PyTorch models.

Functions:
    - save_model(model: torch.nn.Module, directory: str, versioning: bool = True) -> str
    - load_model(model_class: Type[torch.nn.Module], path: str) -> torch.nn.Module
    - load_latest_model(model_class: Type[torch.nn.Module], directory: str) -> torch.nn.Module
    - list_models(directory: str) -> List[str]
"""
import os
import torch
from typing import List, Type
from datetime import datetime

def save_model(model: torch.nn.Module, directory: str = "models/", versioning: bool = True) -> str:
    """
    Save the model parameters to disk with optional versioning.

    Args:
        model: PyTorch model to save.
        directory: Directory to save model files.
        versioning: If True, append timestamp to filename.

    Returns:
        Path to the saved model file.
    """
    os.makedirs(directory, exist_ok=True)
    
    if versioning:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pattern_nn_{timestamp}.pth"
    else:
        filename = "pattern_nn_latest.pth"

    path = os.path.join(directory, filename)
    torch.save({
        'state_dict': model.state_dict(),
        'metadata': {
            'saved_at': datetime.now().isoformat()
        }
    }, path)
    return path

def load_model(model_class: Type[torch.nn.Module], path: str) -> torch.nn.Module:
    """
    Instantiate model_class and load state_dict from the given path.

    Args:
        model_class: Class reference, e.g., PatternNN.
        path: Filesystem path to .pth file.

    Returns:
        Loaded model instance.
    """
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model = model_class()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def load_latest_model(model_class: Type[torch.nn.Module], directory: str = "models/") -> torch.nn.Module:
    """
    Load the latest Pattern NN model based on filename timestamp.

    Args:
        model_class: Class reference, e.g., PatternNN.
        directory: Directory where models are saved.

    Returns:
        Loaded latest model instance.
    """
    model_files = [f for f in os.listdir(directory) if f.startswith("pattern_nn_") and f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError("No Pattern NN models found in models/ directory.")

    model_files.sort(reverse=True)
    latest_model_path = os.path.join(directory, model_files[0])

    return load_model(model_class, latest_model_path)

def list_models(directory: str) -> List[str]:
    """
    List all model files in a directory (.pth files).

    Args:
        directory: Directory path.

    Returns:
        List of model filenames.
    """
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith('.pth')]
