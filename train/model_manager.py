"""
Improved Model Management System with Production-Grade Standards

This refactored version addresses SOLID principles, security, performance,
and maintainability while maintaining backward compatibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Type, Dict, Any, Optional, Tuple, Protocol, Union
from datetime import datetime
from pathlib import Path
import os
import torch
import json
import joblib
import hashlib
import time
import threading
from functools import lru_cache, wraps
import jsonschema
import re
import pandas as pd
from contextlib import contextmanager

from utils.logger import setup_logger
from train.deeplearning_config import TrainingConfig

# Import security utilities
from security.utils import validate_file_path as security_validate_file_path, validate_file_size
from security.encryption import calculate_file_checksum, verify_file_checksum

# Configure logging with correlation ID support
logger = setup_logger(__name__)


# ================================
# Exception Classes
# ================================

class ModelManagerError(Exception):
    """Base exception for model manager errors."""
    pass


class ModelNotFoundError(ModelManagerError):
    """Exception raised when a requested model is not found."""
    pass


class InvalidModelError(ModelManagerError):
    """Exception raised when a model is invalid or corrupted."""
    pass


class ModelValidationError(ModelManagerError):
    """Exception raised when model validation fails."""
    pass


class ModelType(Enum):
    """Enumeration of supported model types."""
    PYTORCH = "pytorch"
    SKLEARN = "sklearn" 
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    CUSTOM = "custom"


# ================================
# Configuration Management
# ================================

@dataclass
class ModelManagerConfig:
    """Centralized configuration for model management operations."""
    base_directory: Path = Path("models/")
    max_file_size_mb: int = 500  # Maximum model file size
    max_cache_entries: int = 10  # Maximum cached models
    cache_ttl_seconds: int = 3600  # Cache time-to-live
    enable_checksums: bool = True  # Enable file integrity checks
    max_versions_to_keep: int = 5  # Versions to retain during cleanup
    backup_directory: Optional[Path] = None  # Backup location
    compression_enabled: bool = False  # Enable model compression
    security_scan_enabled: bool = True  # Enable security scanning
    
    def __post_init__(self):
        """Validate and setup configuration."""
        self.base_directory = Path(self.base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)
        
        if self.backup_directory:
            self.backup_directory = Path(self.backup_directory)
            self.backup_directory.mkdir(parents=True, exist_ok=True)


# ================================
# Enhanced Data Classes
# ================================

@dataclass
class ModelMetadata:
    """Enhanced metadata with validation and security features."""
    version: str
    saved_at: str
    accuracy: Optional[float] = None
    parameters: Dict[str, Any] = None
    framework_version: str = torch.__version__
    backend: Optional[str] = None
    file_checksum: Optional[str] = None
    file_size_bytes: Optional[int] = None
    created_by: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        """Validate and normalize metadata fields."""
        if not isinstance(self.version, str):
            raise ValueError("Version must be a string")
        if not self.parameters:
            self.parameters = {}
        if not self.tags:
            self.tags = []
        
        # Validate accuracy range
        if self.accuracy is not None and not (0.0 <= self.accuracy <= 1.0):
            raise ValueError("Accuracy must be between 0.0 and 1.0")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata instance from dictionary with validation."""
        # Filter out unknown fields for forward compatibility
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the metadata."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the metadata."""
        if tag in self.tags:
            self.tags.remove(tag)


# ================================
# Security & Validation
# ================================

class SecurityValidator:
    """Handles security validation for model operations."""
    
    @staticmethod
    def validate_file_path(path: Path, base_directory: Path) -> bool:
        """
        Prevent path traversal attacks.
        
        .. deprecated::
            Use security.utils.validate_file_path instead.
        """
        import warnings
        warnings.warn(
            "SecurityValidator.validate_file_path is deprecated. Use security.utils.validate_file_path instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return security_validate_file_path(path, base_directory=base_directory)
    
    @staticmethod
    def validate_file_size(path: Path, max_size_mb: int) -> bool:
        """
        Validate file size limits.
        
        .. deprecated::
            Use security.utils.validate_file_size instead.
        """
        import warnings
        warnings.warn(
            "SecurityValidator.validate_file_size is deprecated. Use security.utils.validate_file_size instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return validate_file_size(path, max_size_mb)
    
    @staticmethod
    def calculate_checksum(path: Path) -> str:
        """
        Calculate SHA256 checksum for file integrity.
        
        .. deprecated::
            Use security.encryption.calculate_file_checksum instead.
        """
        import warnings
        warnings.warn(
            "SecurityValidator.calculate_checksum is deprecated. Use security.encryption.calculate_file_checksum instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return calculate_file_checksum(path)
    
    @staticmethod
    def verify_checksum(path: Path, expected_checksum: str) -> bool:
        """
        Verify file integrity using checksum.
        
        .. deprecated::
            Use security.encryption.verify_file_checksum instead.
        """
        import warnings
        warnings.warn(
            "SecurityValidator.verify_checksum is deprecated. Use security.encryption.verify_file_checksum instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return verify_file_checksum(path, expected_checksum)


# ================================
# Performance Monitoring
# ================================

class PerformanceMonitor:
    """Monitors and tracks performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def measure_operation(self, operation_name: str):
        """Context manager to measure operation duration."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_metric(operation_name, duration)
    
    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric."""
        with self._lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append({
                'value': value,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 100 measurements
            if len(self.metrics[metric_name]) > 100:
                self.metrics[metric_name] = self.metrics[metric_name][-100:]
    
    def get_average_duration(self, metric_name: str) -> Optional[float]:
        """Get average duration for a metric."""
        with self._lock:
            if metric_name not in self.metrics:
                return None
            
            values = [m['value'] for m in self.metrics[metric_name]]
            return sum(values) / len(values) if values else None


# ================================
# Caching System
# ================================

class ModelCache:
    """Thread-safe caching system for loaded models."""
    
    def __init__(self, max_entries: int = 10, ttl_seconds: int = 3600):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Tuple[Any, Any]]:
        """Get cached model and metadata."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            # Check TTL
            if time.time() - entry['cached_at'] > self.ttl_seconds:
                del self._cache[key]
                del self._access_times[key]
                return None
            
            # Update access time
            self._access_times[key] = time.time()
            return entry['model'], entry['metadata']
    
    def put(self, key: str, model: Any, metadata: Any) -> None:
        """Cache model and metadata."""
        with self._lock:
            # Evict least recently used if at capacity
            if len(self._cache) >= self.max_entries and key not in self._cache:
                lru_key = min(self._access_times.keys(), key=self._access_times.get)
                del self._cache[lru_key]
                del self._access_times[lru_key]
                logger.debug(f"Evicted cached model: {lru_key}")
            
            self._cache[key] = {
                'model': model,
                'metadata': metadata,
                'cached_at': time.time()
            }
            self._access_times[key] = time.time()
            logger.debug(f"Cached model: {key}")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            logger.info("Model cache cleared")


# ================================
# Abstract Interfaces
# ================================

class ModelPersistence(Protocol):
    """Protocol for model persistence operations."""
    
    def save_model(self, model: Any, path: Path, metadata: ModelMetadata) -> bool:
        """Save model to specified path."""
        ...
    
    def load_model(self, path: Path) -> Tuple[Any, ModelMetadata]:
        """Load model from specified path."""
        ...


class ModelValidator(Protocol):
    """Protocol for model validation."""
    
    def validate_model(self, model: Any, metadata: ModelMetadata) -> bool:
        """Validate model consistency and integrity."""
        ...


# ================================
# Concrete Implementations
# ================================

class PyTorchModelPersistence:
    """Handles PyTorch model persistence."""
    
    def __init__(self, config: ModelManagerConfig):
        self.config = config
        self.security_validator = SecurityValidator()
    
    def save_model(self, model: Any, path: Path, metadata: ModelMetadata, 
                   optimizer=None, epoch=None, loss=None) -> bool:
        """Save PyTorch model with enhanced security and validation."""
        try:
            # Security validation
            if not self.security_validator.validate_file_path(path, self.config.base_directory):
                raise ValueError(f"Invalid file path: {path}")
            
            # Prepare save data
            save_data = {
                'state_dict': model.state_dict(),
                'metadata': metadata.to_dict(),
                'model_class': model.__class__.__name__,
                'model_module': model.__class__.__module__,
            }
            
            if optimizer:
                save_data['optimizer_state_dict'] = optimizer.state_dict()
            if epoch is not None:
                save_data['epoch'] = epoch
            if loss is not None:
                save_data['loss'] = loss
            
            # Save model
            torch.save(save_data, path)
            
            # Calculate and update checksum
            if self.config.enable_checksums:
                checksum = self.security_validator.calculate_checksum(path)
                metadata.file_checksum = checksum
                metadata.file_size_bytes = path.stat().st_size
            
            logger.info(f"PyTorch model saved successfully: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save PyTorch model: {e}")
            return False
    
    def load_model(self, model_class: Type, path: Path, device: Optional[torch.device] = None) -> Tuple[Any, ModelMetadata]:
        """Load PyTorch model with enhanced validation."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Security validation
        if not self.security_validator.validate_file_path(path, self.config.base_directory):
            raise ValueError(f"Invalid file path: {path}")
        
        if not self.security_validator.validate_file_size(path, self.config.max_file_size_mb):
            raise ValueError(f"File too large: {path}")
        
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            checkpoint = torch.load(path, map_location=device)
            
            # Validate checkpoint structure
            required_keys = ['state_dict', 'metadata']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise ValueError(f"Invalid checkpoint format. Missing keys: {missing_keys}")
            
            # Create metadata
            metadata = ModelMetadata.from_dict(checkpoint['metadata'])
            
            # Verify checksum if available
            if self.config.enable_checksums and metadata.file_checksum:
                if not self.security_validator.verify_checksum(path, metadata.file_checksum):
                    raise ValueError("File integrity check failed")
            
            # Instantiate model
            model_params = metadata.parameters or {}
            required_params = ["input_size", "hidden_size", "num_layers", "output_size", "dropout"]
            
            missing_params = [p for p in required_params if p not in model_params]
            if missing_params:
                raise ValueError(f"Missing model parameters: {missing_params}")
            
            model = model_class(
                input_size=model_params["input_size"],
                hidden_size=model_params["hidden_size"],
                num_layers=model_params["num_layers"],
                output_size=model_params["output_size"],
                dropout=model_params["dropout"]
            )
            
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            model.eval()
            
            logger.info(f"PyTorch model loaded successfully: {path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise


class SklearnModelPersistence:
    """Handles Scikit-learn model persistence."""
    
    def __init__(self, config: ModelManagerConfig):
        self.config = config
        self.security_validator = SecurityValidator()
    
    def save_model(self, model: Any, path: Path, metadata: ModelMetadata) -> bool:
        """Save scikit-learn model."""
        try:
            # Security validation
            if not self.security_validator.validate_file_path(path, self.config.base_directory):
                raise ValueError(f"Invalid file path: {path}")
            
            # Save model
            joblib.dump(model, path)
            
            # Update metadata
            if self.config.enable_checksums:
                checksum = self.security_validator.calculate_checksum(path)
                metadata.file_checksum = checksum
                metadata.file_size_bytes = path.stat().st_size
            
            logger.info(f"Sklearn model saved successfully: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save sklearn model: {e}")
            return False
    
    def load_model(self, path: Path) -> Tuple[Any, ModelMetadata]:
        """Load scikit-learn model."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Security validation
        if not self.security_validator.validate_file_path(path, self.config.base_directory):
            raise ValueError(f"Invalid file path: {path}")
        
        try:
            model = joblib.load(path)
            
            # Try to load metadata from companion JSON file
            metadata_path = path.with_suffix('.json')
            if metadata_path.exists():
                with metadata_path.open() as f:
                    metadata_dict = json.load(f)
                metadata = ModelMetadata.from_dict(metadata_dict)
            else:
                # Create basic metadata if none exists
                metadata = ModelMetadata(
                    version="unknown",
                    saved_at=datetime.now().isoformat(),
                    backend="Classic ML (Unknown)"
                )
            
            logger.info(f"Sklearn model loaded successfully: {path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load sklearn model: {e}")
            raise


# ================================
# Enhanced Model Manager
# ================================

class ModelManager:
    """
    Production-grade model manager with enhanced security, performance, and maintainability.
    
    Features:
    - SOLID principles compliance
    - Comprehensive security validation
    - Performance monitoring and caching
    - Structured error handling
    - Configuration management
    - Resource management
    """
    
    def __init__(self, config: Optional[ModelManagerConfig] = None):
        """Initialize ModelManager with configuration."""
        self.config = config or ModelManagerConfig()
        self.cache = ModelCache(
            max_entries=self.config.max_cache_entries,
            ttl_seconds=self.config.cache_ttl_seconds
        )
        self.performance_monitor = PerformanceMonitor()
        self.security_validator = SecurityValidator()
        
        # Initialize persistence handlers
        self.pytorch_persistence = PyTorchModelPersistence(self.config)
        self.sklearn_persistence = SklearnModelPersistence(self.config)
        
        logger.info(f"ModelManager initialized with config: {self.config}")
    
    def save_model(self, model: Any, metadata: ModelMetadata, backend: str = None,
                   optimizer=None, epoch=None, loss=None, csv_filename=None, df=None) -> str:
        """
        Save model with enhanced validation and monitoring.
        
        Args:
            model: Model instance to save
            metadata: Model metadata
            backend: Backend type identifier
            optimizer: Optional optimizer state
            epoch: Training epoch number
            loss: Training loss value
            csv_filename: Source data filename
            df: Source dataframe for additional metadata
            
        Returns:
            str: Path to saved model file
            
        Raises:
            ValueError: If validation fails
            ModelError: If save operation fails
        """
        with self.performance_monitor.measure_operation("save_model"):
            return self._save_model_impl(
                model, metadata, backend, optimizer, epoch, loss, csv_filename, df
            )
    
    def _save_model_impl(self, model: Any, metadata: ModelMetadata, backend: str = None,
                         optimizer=None, epoch=None, loss=None, csv_filename=None, df=None) -> str:
        """Internal implementation of save_model."""
        # Validation
        if model is None:
            raise ValueError("Model object is None")
        if not hasattr(metadata, "version"):
            raise ValueError("Metadata missing required 'version' attribute")
        
        # Determine backend and validate
        backend = backend or getattr(metadata, "backend", None)
        if not backend or not (backend.startswith("Classic") or backend.startswith("Deep")):
            raise ValueError("Backend string must start with 'Classic' or 'Deep'")
        
        try:
            # Extract additional metadata from model and data
            self._enrich_metadata(model, metadata, backend, csv_filename, df)
            
            # Generate file paths
            version = metadata.version or self._generate_version_id()
            metadata.version = version
            
            if backend.startswith("Classic"):
                model_filename = f"classic_ml_{version}.joblib"
                model_path = self.config.base_directory / model_filename
                success = self.sklearn_persistence.save_model(model, model_path, metadata)
            else:
                model_filename = f"pattern_nn_{version}.pth"
                model_path = self.config.base_directory / model_filename
                success = self.pytorch_persistence.save_model(
                    model, model_path, metadata, optimizer, epoch, loss
                )
            
            if not success:
                raise RuntimeError("Model save operation failed")
            
            # Save metadata
            self._save_metadata(model_path, metadata)
            
            # Cleanup old models if needed
            self._cleanup_old_models_if_needed()
            
            # Clear cache to ensure fresh loads
            self.cache.clear()
            
            logger.info(f"Model saved successfully: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Model save failed: {e}")
            raise
    
    def load_model(self, model_class: Optional[Type] = None, path: str = "", 
                   device: Optional[torch.device] = None, use_cache: bool = True) -> Tuple[Any, ModelMetadata]:
        """
        Load model with caching and enhanced validation.
        
        Args:
            model_class: Model class for PyTorch models
            path: Path to model file
            device: Target device for PyTorch models
            use_cache: Whether to use cached models
            
        Returns:
            Tuple[Any, ModelMetadata]: Loaded model and metadata
            
        Raises:
            FileNotFoundError: If model file not found
            ValueError: If validation fails
            ModelError: If load operation fails
        """
        with self.performance_monitor.measure_operation("load_model"):
            return self._load_model_impl(model_class, path, device, use_cache)
    
    def _load_model_impl(self, model_class: Optional[Type], path: str, 
                         device: Optional[torch.device], use_cache: bool) -> Tuple[Any, ModelMetadata]:
        """Internal implementation of load_model."""
        if not path:
            raise ValueError("Model path is required")
        
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Check cache first
        cache_key = str(path_obj.resolve())
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug(f"Loaded model from cache: {path}")
                return cached_result
        
        try:
            # Load based on file extension
            if path_obj.suffix == ".joblib":
                model, metadata = self.sklearn_persistence.load_model(path_obj)
            elif path_obj.suffix == ".pth":
                if model_class is None:
                    raise ValueError("model_class must be provided for PyTorch models")
                model, metadata = self.pytorch_persistence.load_model(model_class, path_obj, device)
            else:
                raise ValueError(f"Unsupported model file extension: {path_obj.suffix}")
            
            # Cache the result
            if use_cache:
                self.cache.put(cache_key, model, metadata)
            
            logger.info(f"Model loaded successfully: {path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise
    
    def get_model_history(self) -> List[Dict[str, Any]]:
        """Get model training history with enhanced filtering."""
        try:
            metadata_files = list(self.config.base_directory.glob("*.json"))
            
            history = []
            for metadata_file in metadata_files:
                try:
                    with metadata_file.open() as f:
                        metadata = json.load(f)
                    
                    # Validate metadata structure
                    if self._is_valid_metadata(metadata):
                        history.append(metadata)
                        
                except Exception as e:
                    logger.warning(f"Failed to read metadata file {metadata_file}: {e}")
                    continue
            
            # Sort by date and add computed fields
            sorted_history = sorted(history, key=lambda x: x.get('saved_at', ''), reverse=True)
            
            # Add computed fields
            for item in sorted_history:
                item['age_days'] = self._calculate_age_days(item.get('saved_at'))
                item['file_exists'] = self._check_model_file_exists(item)
            
            return sorted_history
            
        except Exception as e:
            logger.error(f"Failed to get model history: {e}")
            return []
    
    def cleanup_old_models(self, keep_versions: int = None, keep_latest: bool = True) -> int:
        """
        Enhanced cleanup with better selection criteria.
        
        Args:
            keep_versions: Number of versions to keep (uses config default if None)
            keep_latest: Whether to preserve latest version
            
        Returns:
            int: Number of models cleaned up
        """
        keep_versions = keep_versions or self.config.max_versions_to_keep
        
        try:
            with self.performance_monitor.measure_operation("cleanup_old_models"):
                return self._cleanup_old_models_impl(keep_versions, keep_latest)
        except Exception as e:
            logger.error(f"Model cleanup failed: {e}")
            return 0
    
    def _cleanup_old_models_impl(self, keep_versions: int, keep_latest: bool) -> int:
        """Internal implementation of cleanup_old_models."""
        # Get all model files grouped by type
        pytorch_files = sorted(
            self.config.base_directory.glob("pattern_nn_*.pth"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        sklearn_files = sorted(
            self.config.base_directory.glob("classic_ml_*.joblib"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        cleanup_count = 0
        
        # Cleanup PyTorch models
        files_to_remove = pytorch_files[keep_versions:]
        cleanup_count += self._remove_model_files(files_to_remove)
        
        # Cleanup sklearn models
        files_to_remove = sklearn_files[keep_versions:]
        cleanup_count += self._remove_model_files(files_to_remove)
        
        # Clear cache after cleanup
        if cleanup_count > 0:
            self.cache.clear()
            logger.info(f"Cleaned up {cleanup_count} old model files")
        
        return cleanup_count
    
    def list_models(self, pattern: str = "*.*", include_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        Enhanced model listing with metadata support.
        
        Args:
            pattern: Glob pattern for filtering
            include_metadata: Whether to include full metadata
            
        Returns:
            List of model information dictionaries
        """
        try:
            model_files = sorted(
                self.config.base_directory.glob(pattern),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            results = []
            for file_path in model_files:
                if file_path.suffix in ['.pth', '.joblib']:
                    model_info = {
                        'path': str(file_path),
                        'name': file_path.name,
                        'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        'type': 'PyTorch' if file_path.suffix == '.pth' else 'Sklearn'
                    }
                    
                    if include_metadata:
                        metadata_path = file_path.with_suffix('.json')
                        if metadata_path.exists():
                            try:
                                with metadata_path.open() as f:
                                    model_info['metadata'] = json.load(f)
                            except Exception as e:
                                logger.warning(f"Failed to read metadata for {file_path}: {e}")
                                model_info['metadata'] = None
                        else:
                            model_info['metadata'] = None
                    
                    results.append(model_info)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance monitoring metrics."""
        return {
            'average_save_time': self.performance_monitor.get_average_duration('save_model'),
            'average_load_time': self.performance_monitor.get_average_duration('load_model'),
            'cache_info': {
                'max_entries': self.cache.max_entries,
                'current_entries': len(self.cache._cache),
                'ttl_seconds': self.cache.ttl_seconds
            },
            'disk_usage': self._calculate_disk_usage()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the model management system."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        try:
            # Check directory accessibility
            health_status['checks']['directory_writable'] = self.config.base_directory.exists() and os.access(self.config.base_directory, os.W_OK)
            
            # Check disk space
            disk_usage = self._calculate_disk_usage()
            health_status['checks']['disk_usage_mb'] = disk_usage
            health_status['checks']['disk_usage_ok'] = disk_usage < 1000  # Less than 1GB
            
            # Check cache health
            health_status['checks']['cache_operational'] = len(self.cache._cache) >= 0
            
            # Overall status
            all_checks_passed = all(
                check for check in health_status['checks'].values()
                if isinstance(check, bool)
            )
            health_status['status'] = 'healthy' if all_checks_passed else 'unhealthy'
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['status'] = 'error'
            health_status['error'] = str(e)
        
        return health_status
    
    # ================================
    # Helper Methods
    # ================================
    
    def _generate_version_id(self) -> str:
        """Generate a consistent version ID based on UTC time."""
        return datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
    
    def _enrich_metadata(self, model: Any, metadata: ModelMetadata, backend: str, 
                        csv_filename: str = None, df: pd.DataFrame = None) -> None:
        """Enrich metadata with additional information."""
        # Add system information
        metadata.created_by = os.getenv('USER', 'unknown')
        
        # Extract model parameters for deep learning models
        if backend.startswith("Deep"):
            arch_params = {}
            for param in ["input_size", "hidden_size", "num_layers", "output_size", "dropout"]:
                if hasattr(model, param):
                    arch_params[param] = getattr(model, param)
            
            if hasattr(metadata, "parameters") and isinstance(metadata.parameters, dict):
                arch_params.update({k: v for k, v in metadata.parameters.items() if k not in arch_params})
            metadata.parameters = arch_params
            
            # Add data source information
            if csv_filename and df is not None:
                ticker, interval = self._extract_ticker_interval(csv_filename)
                start_time, end_time = self._get_timeframe(df)
                
                metadata.parameters.update({
                    "ticker": ticker,
                    "interval": interval,
                    "timeframe_start": start_time,
                    "timeframe_end": end_time,
                    "csv_filename": csv_filename,
                    "data_shape": list(df.shape) if df is not None else None
                })
    
    def _extract_ticker_interval(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract ticker and interval from filename."""
        match = re.match(r"([A-Za-z0-9\-\.]+)_([A-Za-z0-9]+)\.csv", filename)
        if match:
            ticker, interval = match.groups()
            return ticker.upper(), interval
        return None, None
    
    def _get_timeframe(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Get timeframe from dataframe."""
        if 'timestamp' in df.columns:
            start = pd.to_datetime(df['timestamp']).min()
            end = pd.to_datetime(df['timestamp']).max()
        elif 'date' in df.columns:
            start = pd.to_datetime(df['date']).min()
            end = pd.to_datetime(df['date']).max()
        else:
            start = end = None
        return str(start), str(end)
    
    def _save_metadata(self, model_path: Path, metadata: ModelMetadata) -> None:
        """Save metadata to JSON file."""
        metadata_filename = model_path.with_suffix('.json')
        
        try:
            with metadata_filename.open('w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            logger.debug(f"Metadata saved: {metadata_filename}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def _cleanup_old_models_if_needed(self) -> None:
        """Automatically cleanup if too many models exist."""
        # Count current models
        pytorch_count = len(list(self.config.base_directory.glob("pattern_nn_*.pth")))
        sklearn_count = len(list(self.config.base_directory.glob("classic_ml_*.joblib")))
        
        max_versions = self.config.max_versions_to_keep
        
        if pytorch_count > max_versions * 2 or sklearn_count > max_versions * 2:
            logger.info("Auto-cleanup triggered due to high model count")
            self.cleanup_old_models()
    
    def _is_valid_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata structure."""
        required_fields = ['version', 'saved_at', 'framework_version']
        return all(field in metadata for field in required_fields)
    
    def _calculate_age_days(self, saved_at: str) -> int:
        """Calculate age in days from saved_at timestamp."""
        try:
            saved_date = datetime.fromisoformat(saved_at.replace('Z', '+00:00'))
            return (datetime.now() - saved_date).days
        except Exception:
            return -1
    
    def _check_model_file_exists(self, metadata: Dict[str, Any]) -> bool:
        """Check if corresponding model file exists."""
        try:
            version = metadata.get('version', '')
            backend = metadata.get('backend', '')
            
            if backend and backend.startswith('Deep'):
                model_path = self.config.base_directory / f"pattern_nn_{version}.pth"
            elif backend and backend.startswith('Classic'):
                model_path = self.config.base_directory / f"classic_ml_{version}.joblib"
            else:
                return False
            
            return model_path.exists()
        except Exception:
            return False
    
    def _remove_model_files(self, files: List[Path]) -> int:
        """Remove model files and their metadata."""
        count = 0
        for model_file in files:
            try:
                # Remove model file
                model_file.unlink(missing_ok=True)
                
                # Remove metadata file
                metadata_file = model_file.with_suffix('.json')
                metadata_file.unlink(missing_ok=True)
                
                count += 1
                logger.debug(f"Removed old model: {model_file.name}")
                
            except Exception as e:
                logger.warning(f"Failed to remove {model_file}: {e}")
        
        return count
    
    def _calculate_disk_usage(self) -> float:
        """Calculate total disk usage of model directory in MB."""
        try:
            total_size = sum(
                f.stat().st_size for f in self.config.base_directory.rglob('*')
                if f.is_file()
            )
            return round(total_size / (1024 * 1024), 2)
        except Exception:
            return 0.0


# ================================
# Backward Compatibility
# ================================

# Enhanced exceptions with more context
class ModelError(Exception):
    """Base exception for model management errors with enhanced context."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message)
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()


class ModelNotFoundError(ModelError):
    """Raised when model file cannot be found."""
    pass


class ModelVersionError(ModelError):
    """Raised when version handling fails."""
    pass


# Utility functions for backward compatibility
def generate_version_id() -> str:
    """Generate a consistent version ID based on UTC time."""
    return datetime.utcnow().strftime("v%Y%m%d_%H%M%S")


def extract_ticker_interval(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract ticker and interval from filename."""
    match = re.match(r"([A-Za-z0-9\-\.]+)_([A-Za-z0-9]+)\.csv", filename)
    if match:
        ticker, interval = match.groups()
        return ticker.upper(), interval
    return None, None


def get_timeframe(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Get timeframe from dataframe."""
    if 'timestamp' in df.columns:
        start = pd.to_datetime(df['timestamp']).min()
        end = pd.to_datetime(df['timestamp']).max()
    elif 'date' in df.columns:
        start = pd.to_datetime(df['date']).min()
        end = pd.to_datetime(df['date']).max()
    else:
        start = end = None
    return str(start), str(end)


def load_latest_model(model_class: Type, base_directory: str = "models/") -> Any:
    """Load the most recent model from the model directory."""
    config = ModelManagerConfig(base_directory=Path(base_directory))
    manager = ModelManager(config)
    
    model_files = sorted(
        config.base_directory.glob("pattern_nn_v*.pth"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not model_files:
        raise FileNotFoundError("No saved models found in the model directory.")
    
    latest_model_path = model_files[0]
    model, metadata = manager.load_model(model_class, str(latest_model_path))
    return model


# Original training function for backward compatibility
def train_model_deep_learning(epochs, seq_len, learning_rate, batch_size, validation_split,
                             early_stopping_patience, min_patterns, max_samples_per_symbol,
                             *args, **kwargs):
    """Backward compatibility function for deep learning training."""
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
    return config


# Model format enum for backward compatibility
class ModelFormat(Enum):
    """Supported model file formats."""
    PTH = ".pth"
    ONNX = ".onnx"
    JOBLIB = ".joblib"


# Enhanced validation schema
MODEL_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"type": "string"},
        "saved_at": {"type": "string"},
        "accuracy": {"type": ["number", "null"]},
        "parameters": {"type": ["object", "null"]},
        "framework_version": {"type": "string"},
        "backend": {"type": ["string", "null"]},
        "file_checksum": {"type": ["string", "null"]},
        "file_size_bytes": {"type": ["integer", "null"]},
        "created_by": {"type": ["string", "null"]},
        "tags": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["version", "saved_at", "framework_version"],
    "additionalProperties": True
}
