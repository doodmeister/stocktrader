"""
Enhanced Model Visualizer & Comparator Dashboard

This module provides a production-grade Streamlit dashboard for visualizing and comparing
machine learning models. It features robust error handling, comprehensive logging,
security validation, performance optimization, and modular architecture following
SOLID principles.

Author: Production Code Refactor
Version: 2.0.0
Last Modified: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import time
from typing import Any, Dict, Optional, Tuple, List, Protocol
from dataclasses import asdict
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
import threading
import warnings
import sys
from io import StringIO # Import StringIO directly

# Core imports
from train.model_manager import ModelManager
from patterns.patterns_nn import PatternNN
from core.streamlit.dashboard_utils import setup_page, handle_streamlit_error
from core.streamlit.session_manager import SessionManager # Added

# Logging setup
from utils.logger import get_dashboard_logger

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Initialize logger
logger = get_dashboard_logger(__name__) # Added logger initialization

# Page setup
setup_page(
    title="📊 Model Visualizer & Comparator",
    logger_name=__name__,
    sidebar_title="Model Controls"
)

# Initialize SessionManager
session_manager = SessionManager(namespace_prefix="model_visualizer") # Added

class ModelType(Enum):
    """Enumeration of supported model types."""
    PATTERN_NN = "PatternNN"
    CLASSIC_ML = "Classic ML"
    UNKNOWN = "Unknown"


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


# --- Configuration Management ---

class ConfigurationManager:
    """Manages configuration settings for the model visualizer."""
    
    def __init__(self):
        self.max_models_to_compare = 10
        self.max_file_size_mb = 500
        self.cache_ttl_seconds = 3600
        self.allowed_extensions = {".pth", ".joblib", ".pkl"}
        self.max_weight_display_params = 1000000
        self.performance_monitoring = True
        
    def validate_model_count(self, count: int) -> bool:
        """Validate the number of models selected for comparison."""
        return 1 <= count <= self.max_models_to_compare
    
    def is_allowed_extension(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        return any(filename.endswith(ext) for ext in self.allowed_extensions)


# --- Security and Validation Services ---

class ValidationService:
    """Handles input validation and sanitization."""
    
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.logger = get_dashboard_logger(f"{__name__}.ValidationService")
    
    def validate_filename(self, filename: str) -> bool:
        """
        Validate filename for security and format compliance.
        
        Args:
            filename: The filename to validate
            
        Returns:
            bool: True if filename is valid
            
        Raises:
            ValidationError: If filename is invalid
        """
        if not filename:
            raise ValidationError("Filename cannot be empty")
        
        if not isinstance(filename, str):
            raise ValidationError("Filename must be a string")
        
        # Check for path traversal attempts
        if ".." in filename or "/" in filename or "\\" in filename:
            raise SecurityError("Invalid filename: path traversal detected")
        
        # Check extension
        if not self.config.is_allowed_extension(filename):
            raise ValidationError(f"Unsupported file extension. Allowed: {self.config.allowed_extensions}")
        
        # Check length
        if len(filename) > 255:
            raise ValidationError("Filename too long")
        
        return True
    
    def validate_model_selection(self, selected_files: List[str], available_files: List[str]) -> bool:
        """
        Validate model selection.
        
        Args:
            selected_files: List of selected model files
            available_files: List of available model files
            
        Returns:
            bool: True if selection is valid
            
        Raises:
            ValidationError: If selection is invalid
        """
        if not selected_files:
            raise ValidationError("At least one model must be selected")
        
        if not self.config.validate_model_count(len(selected_files)):
            raise ValidationError(f"Too many models selected. Maximum: {self.config.max_models_to_compare}")
        
        for filename in selected_files:
            if filename not in available_files:
                raise SecurityError(f"Selected file not in available list: {filename}")
            self.validate_filename(filename)
        
        return True
    
    def sanitize_display_data(self, data: Any) -> Any:
        """
        Sanitize data for safe display.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data safe for display
        """
        if isinstance(data, dict):
            return {str(k)[:100]: self.sanitize_display_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_display_data(item) for item in data[:100]]  # Limit list size
        elif isinstance(data, str):
            return data[:1000]  # Limit string length
        elif isinstance(data, (int, float, bool, type(None))):
            return data
        else:
            return str(data)[:1000]


# --- Caching and Performance Services ---

class CacheManager:
    """Manages caching with TTL and memory optimization."""
    
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.logger = get_dashboard_logger(f"{__name__}.CacheManager")
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.Lock()
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._timestamps:
            return True
        return (time.time() - self._timestamps[key]) > self.config.cache_ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        with self._lock:
            if key in self._cache and not self._is_expired(key):
                self.logger.debug(f"Cache hit for key: {key[:10]}...")
                return self._cache[key]
            elif key in self._cache:
                # Remove expired entry
                del self._cache[key]
                del self._timestamps[key]
                self.logger.debug(f"Cache expired for key: {key[:10]}...")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with timestamp."""
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self.logger.debug(f"Cache set for key: {key[:10]}...")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self.logger.info("Cache cleared")


class PerformanceMonitor:
    """Monitors performance metrics."""
    
    def __init__(self):
        self.logger = get_dashboard_logger(f"{__name__}.PerformanceMonitor")
    
    @contextmanager
    def measure_time(self, operation_name: str):
        """Context manager to measure operation time."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.logger.info(f"Operation '{operation_name}' took {duration:.3f} seconds")
            
            # Display in Streamlit if in debug mode
            if st.session_state.get('debug_mode', False):
                st.sidebar.text(f"⏱️ {operation_name}: {duration:.3f}s")


# --- Model Loading and Processing Services ---

class ModelLoaderProtocol(Protocol):
    """Protocol for model loader implementations."""
    
    def load_model(self, model_class: Optional[type], model_file: str) -> Tuple[Any, Dict]:
        """Load a model and its metadata."""
        pass
    
    def get_model_type(self, metadata: Dict, model_file: str) -> ModelType:
        """Determine the model type."""
        pass


class SecureModelLoader:
    """Secure model loader with comprehensive error handling."""
    
    def __init__(self, config: ConfigurationManager, validator: ValidationService, 
                 cache_manager: CacheManager, performance_monitor: PerformanceMonitor):
        self.config = config
        self.validator = validator
        self.cache_manager = cache_manager
        self.performance_monitor = performance_monitor
        self.logger = get_dashboard_logger(f"{__name__}.SecureModelLoader")
        self.model_manager = None
    
    def _get_model_manager(self) -> ModelManager:
        """Lazy initialization of ModelManager."""
        if self.model_manager is None:
            self.model_manager = ModelManager()
        return self.model_manager
    
    @st.cache_resource(show_spinner=False, ttl=3600)
    # Corrected type hint to match original intention, assuming Any for model and Dict for metadata.
    def load_model(self, model_class: Optional[type], model_file: str) -> Tuple[Any, Dict]: 
        """
        Securely load a model and its metadata with comprehensive error handling.
        
        Args:
            model_class: The model class (for PyTorch models)
            model_file: Path to the model file
            
        Returns:
            Tuple of (model, metadata). If loading fails, returns (None, {}).
        """
        model_to_return: Any = None  # Initialize with None, as per error case
        metadata_to_return: Dict = {} # Initialize with empty dict, as per error case

        try:
            self.validator.validate_filename(model_file)
            cache_key = self.cache_manager._generate_cache_key(str(model_class), model_file, "load_model")
            
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.logger.info(f"Loading model {model_file} from cache")
                if isinstance(cached_result, tuple) and len(cached_result) == 2 and isinstance(cached_result[1], dict):
                    # Ensure the first element can be Any (including None), and second is Dict
                    model_to_return, metadata_to_return = cached_result[0], cached_result[1]
                    return model_to_return, metadata_to_return
                else:
                    self.logger.warning(f"Invalid cached result for {model_file}, reloading.")

            with self.performance_monitor.measure_time(f"Loading model {model_file}"):
                model_manager = self._get_model_manager()
                model, metadata_raw = model_manager.load_model(model_class, model_file)
                
                if model is None:
                    self.logger.error(f"Model loading returned None for {model_file}")
                    # model_to_return is already None, metadata_to_return is already {}
                else:
                    model_to_return = model # model is Any
                    if isinstance(metadata_raw, dict):
                        metadata_to_return = metadata_raw
                    elif metadata_raw is not None:
                        try:
                            metadata_to_return = asdict(metadata_raw)
                        except Exception as e:
                            self.logger.warning(f"Could not convert metadata to dict: {e}")
                            metadata_to_return = {} # Default to empty dict
                    # If metadata_raw was None, metadata_to_return remains {}
                    
                    metadata_to_return = self.validator.sanitize_display_data(metadata_to_return)
                    
                    # Cache the successfully loaded or processed result
                    result_to_cache: Tuple[Any, Dict] = (model_to_return, metadata_to_return)
                    self.cache_manager.set(cache_key, result_to_cache)
                    self.logger.info(f"Successfully loaded model: {model_file}")
        
        except Exception as e:
            error_msg = f"Failed to load model {model_file}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            st.error(f"❌ {error_msg}")
            # On exception, ensure we return (None, {})
            model_to_return = None
            metadata_to_return = {}
        
        # All paths lead here or return earlier from cache.
        # This ensures (Any, Dict) is always returned.
        return model_to_return, metadata_to_return

    # Ensure get_model_type always returns a ModelType enum member.
    def get_model_type(self, metadata: Dict, model_file: str) -> ModelType:
        """
        Determine the model type from metadata or file extension.
        
        Args:
            metadata: Model metadata dictionary
            model_file: Path to the model file
        Returns:
            A ModelType enum member.
        """
        determined_type: ModelType = ModelType.UNKNOWN # Default value

        try:
            if isinstance(metadata, dict) and "model_type" in metadata:
                model_type_str = str(metadata["model_type"])
                try:
                    # Check against enum values first for exact match
                    if model_type_str == ModelType.PATTERN_NN.value:
                        determined_type = ModelType.PATTERN_NN
                    elif model_type_str == ModelType.CLASSIC_ML.value:
                        determined_type = ModelType.CLASSIC_ML
                    # Add other ModelType values here if necessary
                    
                    # If a match was found, log and return
                    if determined_type != ModelType.UNKNOWN:
                        self.logger.info(f"Model type determined as {determined_type.value} from metadata value for {model_file}.")
                        return determined_type

                    # If not matched by specific value, try direct enum conversion by string (name or value)
                    # This might catch cases where model_type_str is "PATTERN_NN" instead of "PatternNN"
                    converted_type = ModelType(model_type_str) # This can raise ValueError
                    self.logger.info(f"Model type determined as {converted_type.value} from metadata string '{model_type_str}' for {model_file}.")
                    return converted_type

                except ValueError:
                    self.logger.warning(
                        f"Unknown or invalid model type string '{model_type_str}' in metadata for file {model_file}. Will try file extension."
                    )
            
            # Fallback to file extension if not determined from metadata or if metadata check failed
            if model_file.endswith(('.pt', '.pth')):
                determined_type = ModelType.PATTERN_NN
                self.logger.info(f"Model file {model_file} has .pt/.pth extension, determined as {determined_type.value}.")
            elif model_file.endswith('.joblib'):
                determined_type = ModelType.CLASSIC_ML
                self.logger.info(f"Model file {model_file} has .joblib extension, determined as {determined_type.value}.")
            elif model_file.endswith(('.h5', '.keras')):
                # This case was previously UNKNOWN, keeping it consistent.
                determined_type = ModelType.UNKNOWN 
                self.logger.warning(f"Model file {model_file} has .h5/.keras extension. Defaulting to {determined_type.value} as it's not directly mapped.")
            else:
                # If no specific extension matches and not found in metadata, it remains UNKNOWN or its last set value.
                if determined_type == ModelType.UNKNOWN: # Log only if it's still unknown at this point
                    self.logger.warning(f"Could not determine model type for {model_file} from metadata or extension. Defaulting to {determined_type.value}.")
        
        except Exception as e:
            self.logger.error(f"Error determining model type for {model_file}: {e}", exc_info=True)
            st.error(f"An error occurred while determining the model type for {model_file}.")
            determined_type = ModelType.UNKNOWN # Ensure it's UNKNOWN on any unexpected error
        
        return determined_type # This will be ModelType.UNKNOWN if no other type was determined.



