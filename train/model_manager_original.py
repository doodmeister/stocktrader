"""
Model management utilities for saving, loading, and versioning PyTorch models.
Provides a robust interface for model persistence with versioning and metadata handling.
"""
from utils.logger import setup_logger
import os
import torch
import json
import joblib
from train.deeplearning_config import TrainingConfig
from typing import List, Type, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from sklearn.base import BaseEstimator
import jsonschema
import re
import pandas as pd

# Configure logging
logger = setup_logger(__name__)

MODEL_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"type": "string"},
        "saved_at": {"type": "string"},
        "accuracy": {"type": ["number", "null"]},
        "parameters": {"type": ["object", "null"]},
        "framework_version": {"type": "string"},
        "backend": {"type": ["string", "null"]}
    },
    "required": ["version", "saved_at", "framework_version"],
    "additionalProperties": True
}

def generate_version_id() -> str:
    """Generate a consistent version ID based on UTC time."""
    return datetime.utcnow().strftime("v%Y%m%d_%H%M%S")

def extract_ticker_interval(filename: str):
    match = re.match(r"([A-Za-z0-9\-\.]+)_([A-Za-z0-9]+)\.csv", filename)
    if match:
        ticker, interval = match.groups()
        return ticker.upper(), interval
    return None, None

def get_timeframe(df: pd.DataFrame):
    if 'timestamp' in df.columns:
        start = pd.to_datetime(df['timestamp']).min()
        end = pd.to_datetime(df['timestamp']).max()
    elif 'date' in df.columns:
        start = pd.to_datetime(df['date']).min()
        end = pd.to_datetime(df['date']).max()
    else:
        start = end = None
    return str(start), str(end)

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
    backend: Optional[str] = None
    
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

    def _load_metadata(self, path: Path) -> Dict[str, Any]:
        """
        Load and validate metadata from a JSON file.
        """
        if not path.exists():
            raise ModelError(f"Metadata file not found: {path}")
        with path.open() as f:
            metadata = json.load(f)
        # Validate metadata
        try:
            jsonschema.validate(instance=metadata, schema=MODEL_METADATA_SCHEMA)
        except jsonschema.ValidationError as ve:
            raise ModelError(f"Metadata schema validation failed: {ve.message}")
        return metadata
        
    def save_model(self, model, metadata, backend=None, optimizer=None, epoch=None, loss=None, csv_filename=None, df=None):
        """
        Save model with metadata and optional backend support.
        """
        if model is None:
            logger.error("Attempted to save a None model.")
            raise ValueError("Model object is None.")
        if not hasattr(metadata, "version"):
            logger.error("Metadata missing required 'version' attribute.")
            raise ValueError("Metadata missing required 'version' attribute.")

        logger.info(f"Saving model to directory: {self.base_directory.resolve()}")
        version = getattr(metadata, "version", None) or generate_version_id()
        if backend is None and hasattr(metadata, "backend"):
            backend = metadata.backend

        if not backend or (not backend.startswith("Classic") and not backend.startswith("Deep")):
            logger.error(f"Invalid or missing backend string: '{backend}'. Must start with 'Classic' or 'Deep'.")
            raise ValueError("Backend string must start with 'Classic' or 'Deep'.")

        try:
            logger.info(f"[DEBUG] backend={backend}, model type={type(model)}")
            logger.info(f"Backend for saving: {backend}")

            if backend.startswith("Classic"):
                # ...existing classic ML save logic...
                model_filename = f"classic_ml_{version}.joblib"
                model_path = self.base_directory / model_filename
                joblib.dump(model, model_path)
            else:
                # --- Extract architecture parameters from model instance ---
                arch_params = {}
                for param in ["input_size", "hidden_size", "num_layers", "output_size", "dropout"]:
                    if hasattr(model, param):
                        arch_params[param] = getattr(model, param)
                # Merge with any training params in metadata
                if hasattr(metadata, "parameters") and isinstance(metadata.parameters, dict):
                    arch_params.update({k: v for k, v in metadata.parameters.items() if k not in arch_params})
                metadata.parameters = arch_params

                if csv_filename and df is not None:
                    ticker, interval = extract_ticker_interval(csv_filename)
                    start_time, end_time = get_timeframe(df)

                    metadata.parameters.update({
                        "ticker": ticker,
                        "interval": interval,
                        "timeframe_start": start_time,
                        "timeframe_end": end_time,
                        "csv_filename": csv_filename
                    })

                model_filename = f"pattern_nn_{version}.pth"
                model_path = self.base_directory / model_filename
                logger.info(f"Saving PyTorch model to: {model_path}")
                torch.save({
                    'state_dict': model.state_dict(),
                    'metadata': metadata.to_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                    'epoch': epoch,
                    'loss': loss,
                }, model_path)

            # Save metadata as JSON
            metadata_filename = model_filename.replace(".pth", ".json").replace(".joblib", ".json")
            metadata_path = self.base_directory / metadata_filename
            logger.info(f"Saving metadata to: {metadata_path}")
            with metadata_path.open('w') as f:
                json.dump(metadata.to_dict(), f, indent=2)

            logger.info(f"Model and metadata saved successfully.")
            logger.info(f"Model file absolute path: {model_path.resolve()}")
            logger.info(f"Metadata file absolute path: {metadata_path.resolve()}")
            logger.info(f"Model file exists after save: {model_path.exists()}")
            logger.info(f"Metadata file exists after save: {metadata_path.exists()}")

            # Optional: Try loading to validate
            try:
                if backend.startswith("Classic"):
                    joblib.load(model_path)
                else:
                    torch.load(model_path)
                logger.info("Model loaded successfully after save.")
            except Exception as e:
                logger.error(f"Failed to load model after save: {e}")

            return str(model_path)
        except Exception as e:
            logger.error(f"Exception during model save: {e}")
            raise

    def load_model(self, model_class: Optional[Type[torch.nn.Module]] = None, path: str = "", device: Optional[torch.device] = None) -> Tuple[Any, Any]:
        """
        Load model and metadata from path.
        """
        if not path or not Path(path).exists():
            logger.error(f"Model file not found: {path}")
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            path = Path(path)
            if not path.exists():
                raise ModelNotFoundError(f"Model file not found: {path}")

            if path.suffix == ".joblib":
                model = joblib.load(path)
                metadata_path = path.with_suffix('.json')
                if metadata_path.exists():
                    metadata = self._load_metadata(metadata_path)
                else:
                    metadata = {}
                logger.info(f"Scikit-learn model loaded successfully from {path}")
                return model, metadata

            elif path.suffix == ".pth":
                device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = torch.load(path, map_location=device)
                if model_class is None:
                    raise ValueError("model_class must be provided for PyTorch models")
                # --- Read parameters from metadata ---
                if 'metadata' in checkpoint:
                    metadata_dict = checkpoint['metadata']
                    # Validate metadata with jsonschema
                    try:
                        jsonschema.validate(instance=metadata_dict, schema=MODEL_METADATA_SCHEMA)
                    except jsonschema.ValidationError as ve:
                        raise ModelError(f"Metadata schema validation failed: {ve.message}")
                else:
                    metadata_path = path.with_suffix('.json')
                    if metadata_path.exists():
                        metadata_dict = self._load_metadata(metadata_path)
                    else:
                        raise ModelError("No metadata found for model.")

                params = metadata_dict.get("parameters", {})
                # --- Validate required architecture params with try/except ---
                required = ["input_size", "hidden_size", "num_layers", "output_size", "dropout"]
                try:
                    missing = [p for p in required if p not in params]
                    if missing:
                        raise ModelError(f"Missing parameters in metadata: {missing}")
                except Exception as e:
                    raise ModelError(f"Error during parameter validation: {e}")
                # Instantiate model with correct parameters
                if callable(model_class):
                    model = model_class(
                        input_size=params["input_size"],
                        hidden_size=params["hidden_size"],
                        num_layers=params["num_layers"],
                        output_size=params["output_size"],
                        dropout=params["dropout"]
                    )
                else:
                    raise ValueError("model_class must be a callable factory function")

                # Add this assertion to prevent misconfiguration
                from patterns.patterns_nn import PatternNN  # Adjust import path if needed
                if model_class is PatternNN:
                    assert isinstance(model, PatternNN), "Loaded model does not match expected architecture"

                model.load_state_dict(checkpoint['state_dict'])
                model.to(device)
                model.eval()
                metadata = ModelMetadata.from_dict(metadata_dict)
                logger.info(f"PyTorch model loaded successfully from {path}")
                return model, metadata

            else:
                raise ModelError(f"Unsupported model file extension: {path.suffix}")

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

    def list_models(self, pattern: str = "*.*") -> list:
        """
        List all saved model files in the model directory.

        Args:
            pattern: Glob pattern for model files (default: "*.*")

        Returns:
            List of model file paths as strings.
        """
        model_files = sorted(self.base_directory.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        return [str(f) for f in model_files]

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

def train_model_deep_learning(
    epochs,
    seq_len,
    learning_rate,
    batch_size,
    validation_split,
    early_stopping_patience,
    min_patterns,
    max_samples_per_symbol,
    *args,
    **kwargs
):
    config = TrainingConfig(
        epochs=epochs,
        seq_len=seq_len,
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_split=validation_split,
        early_stopping_patience=early_stopping_patience,
        min_patterns=min_patterns,
        max_samples_per_symbol=max_samples_per_symbol
    )
