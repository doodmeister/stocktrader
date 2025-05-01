"""
Model management utilities for saving, loading, and versioning PyTorch models.
Provides a robust interface for model persistence with versioning and metadata handling.
"""
import logging
import os
import torch
import json
from typing import List, Type, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Base exception for model management errors."""
    pass

class ModelNotFoundError(ModelError):
    """Raised when model file cannot be found."""
    pass

class ModelVersionError(ModelError):
    """Raised when version handling fails."""
    pass

class ModelFormat(Enum):
    """Supported model file formats."""
    PTH = ".pth"
    ONNX = ".onnx"

@dataclass
class ModelMetadata:
    """Immutable metadata for model versioning and tracking."""
    version: str
    saved_at: str
    accuracy: Optional[float] = None
    parameters: Dict[str, Any] = None
    framework_version: str = torch.__version__
    
    def __post_init__(self):
        """Validate metadata fields."""
        if not isinstance(self.version, str):
            raise ValueError("Version must be a string")
        if not self.parameters:
            self.parameters = {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata instance from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

class ModelManager:
    """Handles model persistence operations with versioning support."""
    
    def __init__(self, base_directory: str = "models/"):
        """
        Initialize model manager.
        
        Args:
            base_directory: Root directory for model storage
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)
        
    def save_model(self, 
                  model: torch.nn.Module, 
                  metadata: Optional[Dict[str, Any]] = None,
                  versioning: bool = True) -> str:
        """
        Save model with metadata and versioning.
        
        Args:
            model: PyTorch model to save
            metadata: Additional metadata to store
            versioning: Whether to use timestamped versioning
            
        Returns:
            Path to saved model file
            
        Raises:
            ModelError: If save operation fails
        """
        try:
            # Generate version info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v{timestamp}" if versioning else "latest"
            
            # Create and validate metadata
            model_metadata = ModelMetadata(
                version=version,
                saved_at=datetime.now().isoformat(),
                parameters=metadata
            )

            # Prepare filenames
            model_filename = f"pattern_nn_{version}{ModelFormat.PTH.value}"
            metadata_filename = f"pattern_nn_{version}_metadata.json"
            
            model_path = self.base_directory / model_filename
            metadata_path = self.base_directory / metadata_filename

            # Save model state
            torch.save({
                'state_dict': model.state_dict(),
                'metadata': model_metadata.to_dict()
            }, model_path)

            # Save separate metadata for easier access
            with metadata_path.open('w') as f:
                json.dump(model_metadata.to_dict(), f, indent=2)

            logger.info(f"Model saved successfully: {model_path}")
            return str(model_path)

        except Exception as e:
            raise ModelError(f"Failed to save model: {str(e)}") from e

    def load_model(self, 
                  model_class: Type[torch.nn.Module],
                  path: str,
                  device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, ModelMetadata]:
        """
        Load model and metadata from path.
        
        Args:
            model_class: Model class for instantiation
            path: Path to model file
            device: Target device for model
            
        Returns:
            Tuple of (loaded_model, metadata)
            
        Raises:
            ModelNotFoundError: If model file doesn't exist
            ModelError: If loading fails
        """
        try:
            path = Path(path)
            if not path.exists():
                raise ModelNotFoundError(f"Model file not found: {path}")

            device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location=device)
            
            # Initialize and load model
            model = model_class()
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            model.eval()

            # Load metadata
            metadata = ModelMetadata.from_dict(checkpoint['metadata'])
            
            logger.info(f"Model loaded successfully from {path}")
            return model, metadata

        except Exception as e:
            raise ModelError(f"Failed to load model: {str(e)}") from e

    def get_model_history(self) -> List[Dict[str, Any]]:
        """
        Get training history from metadata files.
        
        Returns:
            List of metadata dictionaries ordered by date
        """
        try:
            metadata_files = list(self.base_directory.glob("pattern_nn_*_metadata.json"))
            
            history = []
            for metadata_file in metadata_files:
                with metadata_file.open() as f:
                    metadata = json.load(f)
                    history.append(metadata)
            
            return sorted(history, key=lambda x: x['saved_at'], reverse=True)

        except Exception as e:
            logger.error(f"Failed to get model history: {str(e)}")
            return []

    def cleanup_old_models(self, 
                         keep_versions: int = 5,
                         keep_latest: bool = True) -> None:
        """
        Remove old model versions.
        
        Args:
            keep_versions: Number of recent versions to keep
            keep_latest: Whether to preserve latest version
            
        Raises:
            ModelError: If cleanup fails
        """
        try:
            model_files = sorted(
                [f for f in self.base_directory.glob(f"pattern_nn_*{ModelFormat.PTH.value}") 
                 if not f.name.endswith('_metadata.json')],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            if keep_latest:
                latest = next((f for f in model_files if "latest" in f.name), None)
                if latest:
                    model_files.remove(latest)

            # Remove old versions
            for model_file in model_files[keep_versions:]:
                metadata_file = self.base_directory / f"{model_file.stem}_metadata.json"
                model_file.unlink(missing_ok=True)
                metadata_file.unlink(missing_ok=True)
                logger.info(f"Removed old model version: {model_file.name}")

        except Exception as e:
            raise ModelError(f"Failed to cleanup old models: {str(e)}") from e

def load_latest_model(model_class, base_directory: str = "models/"):
    """
    Load the most recent model from the model directory.
    """
    base_dir = Path(base_directory)
    model_files = sorted(
        base_dir.glob("pattern_nn_v*.pth"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    if not model_files:
        raise FileNotFoundError("No saved models found in the model directory.")
    latest_model_path = model_files[0]
    manager = ModelManager(base_directory)
    model, metadata = manager.load_model(model_class, str(latest_model_path))
    return model
