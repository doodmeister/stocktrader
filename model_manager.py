# model_manager.py
"""
Model management utilities for saving, loading, and versioning PyTorch models.

Functions:
    - save_model(model: torch.nn.Module, directory: str, versioning: bool = True, metadata: Optional[Dict[str, Any]] = None) -> str
    - load_model(model_class: Type[torch.nn.Module], path: str, device: Optional[torch.device] = None) -> tuple[torch.nn.Module, ModelMetadata]
    - load_latest_model(model_class: Type[torch.nn.Module], directory: str, device: Optional[torch.device] = None) -> tuple[torch.nn.Module, ModelMetadata]
    - get_model_history(directory: str) -> List[Dict[str, Any]]
    - cleanup_old_models(directory: str, keep_versions: int, keep_latest: bool) -> None
"""
import os
import torch
import json
from typing import List, Type, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

class ModelMetadata:
    """Class to handle model metadata."""
    def __init__(self, 
                 version: str,
                 saved_at: str,
                 accuracy: float = None,
                 parameters: Dict[str, Any] = None):
        self.version = version
        self.saved_at = saved_at
        self.accuracy = accuracy
        self.parameters = parameters or {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'saved_at': self.saved_at,
            'accuracy': self.accuracy,
            'parameters': self.parameters
        }

def save_model(model: torch.nn.Module, 
               directory: str = "models/", 
               versioning: bool = True,
               metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Save the model parameters and metadata to disk with versioning.

    Args:
        model: PyTorch model to save
        directory: Directory to save model files
        versioning: If True, append timestamp to filename
        metadata: Additional metadata to save with the model

    Returns:
        Path to the saved model file
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    # Generate version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"v{timestamp}" if versioning else "latest"
    
    # Create metadata
    model_metadata = ModelMetadata(
        version=version,
        saved_at=datetime.now().isoformat(),
        parameters=metadata
    )

    # Save model with metadata
    filename = f"pattern_nn_{version}.pth"
    path = directory / filename
    
    torch.save({
        'state_dict': model.state_dict(),
        'metadata': model_metadata.to_dict()
    }, path)

    # Save separate metadata file for easier access
    metadata_path = directory / f"pattern_nn_{version}_metadata.json"
    with metadata_path.open('w') as f:
        json.dump(model_metadata.to_dict(), f, indent=2)

    return str(path)

def load_model(model_class: Type[torch.nn.Module], 
               path: str, 
               device: Optional[torch.device] = None) -> tuple[torch.nn.Module, ModelMetadata]:
    """
    Load model and metadata from path.

    Args:
        model_class: Class reference for model instantiation
        path: Path to model file
        device: Target device for model

    Returns:
        Tuple of (loaded_model, metadata)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path, map_location=device)
    model = model_class()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    metadata = ModelMetadata.from_dict(checkpoint['metadata'])
    return model, metadata

def load_latest_model(model_class: Type[torch.nn.Module], 
                     directory: str = "models/",
                     device: Optional[torch.device] = None) -> tuple[torch.nn.Module, ModelMetadata]:
    """
    Load the latest model based on version/timestamp.

    Args:
        model_class: Class reference for model
        directory: Models directory
        device: Target device for model

    Returns:
        Tuple of (loaded_model, metadata)
    """
    directory = Path(directory)
    model_files = sorted(
        [f for f in directory.glob("pattern_nn_*.pth") if not f.name.endswith('_metadata.json')],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not model_files:
        raise FileNotFoundError(f"No models found in {directory}")

    return load_model(model_class, str(model_files[0]), device)

def get_model_history(directory: str = "models/") -> List[Dict[str, Any]]:
    """
    Get training history from all model metadata files.

    Args:
        directory: Models directory

    Returns:
        List of metadata dictionaries ordered by date
    """
    directory = Path(directory)
    metadata_files = directory.glob("pattern_nn_*_metadata.json")
    
    history = []
    for metadata_file in metadata_files:
        with metadata_file.open() as f:
            metadata = json.load(f)
            history.append(metadata)
    
    return sorted(history, key=lambda x: x['saved_at'], reverse=True)

def cleanup_old_models(directory: str = "models/", 
                      keep_versions: int = 5,
                      keep_latest: bool = True) -> None:
    """
    Remove old model versions, keeping only the specified number of recent versions.

    Args:
        directory: Models directory
        keep_versions: Number of recent versions to keep
        keep_latest: Whether to always keep the latest version
    """
    directory = Path(directory)
    model_files = sorted(
        [f for f in directory.glob("pattern_nn_*.pth") if not f.name.endswith('_metadata.json')],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if keep_latest:
        latest = next((f for f in model_files if "latest" in f.name), None)
        if latest:
            model_files.remove(latest)

    # Remove old versions
    for model_file in model_files[keep_versions:]:
        metadata_file = directory / f"{model_file.stem}_metadata.json"
        model_file.unlink(missing_ok=True)
        metadata_file.unlink(missing_ok=True)
