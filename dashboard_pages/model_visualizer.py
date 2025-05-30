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
import traceback
from typing import Any, Dict, Optional, Tuple, List, Union, Protocol
from dataclasses import asdict
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from enum import Enum
import threading
from functools import wraps
import warnings
import io
import sys

# Core imports
from train.model_manager import ModelManager
from patterns.patterns_nn import PatternNN
from core.dashboard_utils import setup_page, handle_streamlit_error
from core.session_manager import create_session_manager, show_session_debug_info

# Logging setup
from utils.logger import get_dashboard_logger

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


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
    def load_model(self, model_class: Optional[type], model_file: str) -> Tuple[Any, Dict]:
        """
        Securely load a model and its metadata with comprehensive error handling.
        
        Args:
            model_class: The model class (for PyTorch models)
            model_file: Path to the model file
            
        Returns:
            Tuple of (model, metadata) or (None, {}) if loading fails
            
        Raises:
            ModelLoadError: If model loading fails
            ValidationError: If validation fails
            SecurityError: If security validation fails
        """
        # Input validation
        self.validator.validate_filename(model_file)
        
        # Generate cache key
        cache_key = self.cache_manager._generate_cache_key(
            str(model_class), model_file, "load_model"
        )
        
        # Check cache first
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            self.logger.info(f"Loading model {model_file} from cache")
            return cached_result
        
        try:
            with self.performance_monitor.measure_time(f"Loading model {model_file}"):
                model_manager = self._get_model_manager()
                
                # Attempt to load the model
                model, metadata = model_manager.load_model(model_class, model_file)
                
                # Validate loaded data
                if model is None:
                    raise ModelLoadError(f"Model loading returned None for {model_file}")
                
                # Convert metadata to dict if needed
                if not isinstance(metadata, dict):
                    try:
                        metadata = asdict(metadata)
                    except Exception as e:
                        self.logger.warning(f"Could not convert metadata to dict: {e}")
                        metadata = {}
                
                # Sanitize metadata for security
                metadata = self.validator.sanitize_display_data(metadata)
                
                result = (model, metadata)
                
                # Cache the result
                self.cache_manager.set(cache_key, result)
                
                self.logger.info(f"Successfully loaded model: {model_file}")
                return result
                
        except Exception as e:
            error_msg = f"Failed to load model {model_file}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Display user-friendly error
            st.error(f"❌ {error_msg}")
            
            # Return empty result instead of raising for graceful degradation
            return None, {}
    
    def get_model_type(self, metadata: Dict, model_file: str) -> ModelType:
        """
        Determine the model type from metadata or file extension.
        
        Args:
            metadata: Model metadata dictionary
            model_file: Path to the model file
            
        Returns:
            ModelType enum value
        """
        try:
            # Check metadata first
            if isinstance(metadata, dict) and "model_type" in metadata:
                model_type_str = metadata["model_type"]
                try:
                    return ModelType(model_type_str)
                except ValueError:
                    self.logger.warning(f"Unknown model type in metadata: {model_type_str}")
            
            # Fall back to file extension
            if model_file.endswith(".pth"):
                return ModelType.PATTERN_NN
            elif model_file.endswith((".joblib", ".pkl")):
                return ModelType.CLASSIC_ML
            else:
                self.logger.warning(f"Unknown file extension for {model_file}")
                return ModelType.UNKNOWN
                
        except Exception as e:
            self.logger.error(f"Error determining model type for {model_file}: {e}")
            return ModelType.UNKNOWN


# --- Model Display Handlers ---

class ModelDisplayHandler(ABC):
    """Abstract base class for model display handlers."""
    
    def __init__(self, validator: ValidationService, performance_monitor: PerformanceMonitor):
        self.validator = validator
        self.performance_monitor = performance_monitor
        self.logger = get_dashboard_logger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def display_model(self, model: Any, metadata: Dict, model_file: str) -> Optional[Dict]:
        """Display model information and return metrics if available."""
        pass
    
    def _safe_display_json(self, data: Any, title: str) -> None:
        """Safely display JSON data with error handling."""
        try:
            sanitized_data = self.validator.sanitize_display_data(data)
            if sanitized_data:
                st.subheader(title)
                st.json(sanitized_data)
            else:
                st.info(f"No {title.lower()} available")
        except Exception as e:
            self.logger.error(f"Error displaying {title}: {e}")
            st.error(f"Could not display {title}")


class PatternNNDisplayHandler(ModelDisplayHandler):
    """Handler for displaying PatternNN models."""
    
    def display_model(self, model: Any, metadata: Dict, model_file: str) -> Optional[Dict]:
        """
        Display PatternNN model architecture, parameters, and metrics.
        
        Args:
            model: The PatternNN model instance
            metadata: Model metadata dictionary
            model_file: Path to the model file
            
        Returns:
            Dictionary of metrics if available, None otherwise
        """
        try:
            with self.performance_monitor.measure_time(f"Displaying PatternNN model {model_file}"):
                st.subheader("🧠 PatternNN Model Architecture")
                
                # Display model architecture and parameters in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Model Architecture:**")
                    try:
                        # Capture model string representation safely
                        with io.StringIO() as buf:
                            old_stdout = sys.stdout
                            sys.stdout = buf
                            try:
                                model_str = str(model)
                            finally:
                                sys.stdout = old_stdout
                            model_output = buf.getvalue()
                        
                        if model_output:
                            st.text(model_output[:2000])  # Limit output length
                        else:
                            st.text(model_str[:2000] if model_str else "Model architecture not available")
                    except Exception as e:
                        self.logger.error(f"Error displaying model architecture: {e}")
                        st.error("Could not display model architecture")
                
                with col2:
                    self._safe_display_json(metadata.get("parameters", {}), "Model Parameters")
                
                # Model weights section with safety checks
                self._display_model_weights(model, model_file)
                
                # Display metrics
                metrics = metadata.get("metrics")
                if metrics:
                    self._safe_display_json(metrics, "Performance Metrics")
                    return metrics
                else:
                    st.info("No performance metrics available")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error displaying PatternNN model {model_file}: {e}")
            st.error(f"Could not display model: {str(e)}")
            return None
    
    def _display_model_weights(self, model: Any, model_file: str) -> None:
        """Safely display model weights with memory considerations."""
        try:
            if self.session_manager.create_checkbox(f"Show weights for {model_file}", f"weights_{model_file}"):
                st.subheader("⚖️ Model Weights")
                
                if not hasattr(model, 'named_parameters'):
                    st.warning("Model does not have named_parameters method")
                    return
                
                # Count total parameters first
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                st.info(f"Total trainable parameters: {total_params:,}")
                
                if total_params > 1000000:  # 1M parameters
                    st.warning("⚠️ Large model detected. Showing limited weight information.")
                    show_weights = self.session_manager.create_checkbox("Show anyway (may be slow)", f"force_weights_{model_file}")
                    if not show_weights:
                        return
                
                # Display weights with pagination
                params_list = list(model.named_parameters())
                max_display = min(20, len(params_list))  # Limit to 20 layers
                
                if len(params_list) > max_display:
                    st.info(f"Showing first {max_display} of {len(params_list)} parameter groups")
                
                for i, (name, param) in enumerate(params_list[:max_display]):
                    with st.expander(f"📊 {name} - Shape: {tuple(param.shape)}"):
                        try:
                            param_data = param.data.cpu().numpy()
                            st.write(f"**Statistics:**")
                            st.write(f"- Mean: {param_data.mean():.6f}")
                            st.write(f"- Std: {param_data.std():.6f}")
                            st.write(f"- Min: {param_data.min():.6f}")
                            st.write(f"- Max: {param_data.max():.6f}")
                            
                            # Show actual values only for small tensors
                            if param.numel() <= 100:
                                st.write("**Values:**")
                                st.write(param_data)
                            else:
                                st.info(f"Tensor too large to display ({param.numel()} elements)")
                                
                        except Exception as e:
                            self.logger.error(f"Error displaying parameter {name}: {e}")
                            st.error(f"Could not display parameter {name}")
                            
        except Exception as e:
            self.logger.error(f"Error in weight display for {model_file}: {e}")
            st.error("Could not display model weights")


class ClassicMLDisplayHandler(ModelDisplayHandler):
    """Handler for displaying Classic ML models."""
    
    def display_model(self, model: Any, metadata: Dict, model_file: str) -> Optional[Dict]:
        """
        Display Classic ML model information, parameters, and feature importance.
        
        Args:
            model: The ML model instance
            metadata: Model metadata dictionary
            model_file: Path to the model file
            
        Returns:
            Dictionary of metrics if available, None otherwise
        """
        try:
            with self.performance_monitor.measure_time(f"Displaying Classic ML model {model_file}"):
                st.subheader("🤖 Classic ML Model")
                
                # Display model summary and parameters in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Model Summary:**")
                    try:
                        model_str = str(model)
                        st.text(model_str[:1000])  # Limit output length
                    except Exception as e:
                        self.logger.error(f"Error displaying model summary: {e}")
                        st.error("Could not display model summary")
                
                with col2:
                    self._safe_display_json(metadata.get("parameters", {}), "Model Parameters")
                
                # Feature importance or coefficients
                self._display_feature_analysis(model, metadata)
                
                # Display metrics
                metrics = metadata.get("metrics")
                if metrics:
                    self._safe_display_json(metrics, "Performance Metrics")
                    return metrics
                else:
                    st.info("No performance metrics available")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error displaying Classic ML model {model_file}: {e}")
            st.error(f"Could not display model: {str(e)}")
            return None
    
    def _display_feature_analysis(self, model: Any, metadata: Dict) -> None:
        """Display feature importance or coefficients."""
        try:
            if hasattr(model, "feature_importances_"):
                st.subheader("📊 Feature Importance")
                importances = model.feature_importances_
                features = metadata.get("parameters", {}).get("features", [])
                
                if features and len(features) == len(importances):
                    # Create importance DataFrame
                    df = pd.DataFrame({
                        "Feature": features, 
                        "Importance": importances
                    }).sort_values("Importance", ascending=False)
                    
                    # Display top features
                    st.write("**Top 10 Most Important Features:**")
                    st.dataframe(df.head(10))
                    
                    # Visualization
                    if len(df) <= 20:  # Only chart if reasonable number of features
                        st.bar_chart(df.set_index("Feature")["Importance"])
                    else:
                        st.bar_chart(df.head(20).set_index("Feature")["Importance"])
                        st.info(f"Showing top 20 of {len(df)} features")
                else:
                    st.write("**Feature Importances (raw):**")
                    st.write(importances)
                    
            elif hasattr(model, "coef_"):
                st.subheader("📈 Model Coefficients")
                try:
                    coef = model.coef_
                    if coef.ndim == 1:
                        st.write(f"Coefficients shape: {coef.shape}")
                        if len(coef) <= 50:  # Reasonable number to display
                            st.write(coef)
                        else:
                            st.write("**Coefficient Statistics:**")
                            st.write(f"- Mean: {coef.mean():.6f}")
                            st.write(f"- Std: {coef.std():.6f}")
                            st.write(f"- Min: {coef.min():.6f}")
                            st.write(f"- Max: {coef.max():.6f}")
                            st.info(f"Too many coefficients to display ({len(coef)})")
                    else:
                        st.write(f"Coefficients shape: {coef.shape}")
                        st.write(coef)
                except Exception as e:
                    self.logger.error(f"Error displaying coefficients: {e}")
                    st.error("Could not display coefficients")
            else:
                st.info("No feature importance or coefficients available for this model")
                
        except Exception as e:
            self.logger.error(f"Error in feature analysis: {e}")
            st.error("Could not display feature analysis")


# --- Metrics Comparison Service ---

class MetricsComparisonService:
    """Service for comparing metrics across multiple models."""
    
    def __init__(self, validator: ValidationService, performance_monitor: PerformanceMonitor):
        self.validator = validator
        self.performance_monitor = performance_monitor
        self.logger = get_dashboard_logger(f"{__name__}.MetricsComparisonService")
    
    def compare_metrics(self, all_metrics: Dict[str, Dict]) -> None:
        """
        Visualize metric comparisons across models with enhanced error handling.
        
        Args:
            all_metrics: Dictionary mapping model names to their metrics
        """
        if not all_metrics:
            st.info("No metrics available for comparison")
            return
        
        if len(all_metrics) == 1:
            st.info("Select multiple models to see comparison charts")
            return
        
        try:
            with self.performance_monitor.measure_time("Generating metric comparisons"):
                st.markdown("## 📊 Metric Comparison Across Models")
                
                # Sanitize metrics data
                sanitized_metrics = {}
                for model_name, metrics in all_metrics.items():
                    sanitized_metrics[model_name] = self.validator.sanitize_display_data(metrics)
                
                # Create comparison DataFrame
                try:
                    metric_df = pd.DataFrame(sanitized_metrics).T
                    
                    if metric_df.empty:
                        st.warning("No comparable metrics found")
                        return
                    
                    # Display summary table
                    st.subheader("📋 Metrics Summary Table")
                    st.dataframe(metric_df.round(4))
                    
                    # Generate individual metric charts
                    self._generate_metric_charts(metric_df)
                    
                    # Generate comparison insights
                    self._generate_comparison_insights(metric_df)
                    
                except Exception as e:
                    self.logger.error(f"Error creating comparison DataFrame: {e}")
                    st.error("Could not create metrics comparison table")
                    
                    # Fallback: show individual metrics
                    self._display_individual_metrics(sanitized_metrics)
                    
        except Exception as e:
            self.logger.error(f"Error in metrics comparison: {e}")
            st.error("Could not generate metrics comparison")
    
    def _generate_metric_charts(self, metric_df: pd.DataFrame) -> None:
        """Generate individual charts for each metric."""
        for metric in metric_df.columns:
            try:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(metric_df[metric]):
                    continue
                
                # Remove NaN values
                clean_data = metric_df[[metric]].dropna()
                
                if clean_data.empty:
                    continue
                
                with st.expander(f"📈 {metric} Comparison"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        try:
                            st.bar_chart(clean_data)
                        except Exception:
                            # Fallback to table if chart fails
                            st.dataframe(clean_data)
                    
                    with col2:
                        # Show metric statistics
                        st.write("**Statistics:**")
                        st.write(f"Best: {clean_data[metric].max():.4f}")
                        st.write(f"Worst: {clean_data[metric].min():.4f}")
                        st.write(f"Mean: {clean_data[metric].mean():.4f}")
                        st.write(f"Std: {clean_data[metric].std():.4f}")
                        
                        # Highlight best model
                        best_model = clean_data[metric].idxmax()
                        st.success(f"🏆 Best: {best_model}")
                        
            except Exception as e:
                self.logger.warning(f"Could not plot metric '{metric}': {e}")
                continue
    
    def _generate_comparison_insights(self, metric_df: pd.DataFrame) -> None:
        """Generate insights from metric comparisons."""
        try:
            st.subheader("💡 Comparison Insights")
            
            numeric_cols = metric_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                st.info("No numeric metrics available for insights")
                return
            
            insights = []
            
            # Find best performing model for each metric
            for metric in numeric_cols:
                clean_data = metric_df[metric].dropna()
                if not clean_data.empty:
                    best_model = clean_data.idxmax()
                    best_value = clean_data.max()
                    insights.append(f"**{metric}**: {best_model} ({best_value:.4f})")
            
            if insights:
                st.write("🏆 **Best performing models by metric:**")
                for insight in insights:
                    st.write(f"- {insight}")
            
            # Overall performance summary
            if len(numeric_cols) > 1:
                # Normalize metrics and calculate overall score
                normalized_df = (metric_df[numeric_cols] - metric_df[numeric_cols].min()) / (
                    metric_df[numeric_cols].max() - metric_df[numeric_cols].min()
                )
                overall_scores = normalized_df.mean(axis=1).sort_values(ascending=False)
                
                st.write("🎯 **Overall performance ranking:**")
                for i, (model, score) in enumerate(overall_scores.items(), 1):
                    st.write(f"{i}. {model} (score: {score:.3f})")
                    
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            st.error("Could not generate comparison insights")
    
    def _display_individual_metrics(self, metrics: Dict[str, Dict]) -> None:
        """Fallback method to display individual metrics."""
        st.subheader("📊 Individual Model Metrics")
        for model_name, model_metrics in metrics.items():
            with st.expander(f"Metrics for {model_name}"):
                st.json(model_metrics)


# --- Main Dashboard Service ---

class ModelVisualizerService:
    """Main service orchestrating the model visualization dashboard."""
    
    def __init__(self):
        # Initialize configuration and core services
        self.config = ConfigurationManager()
        self.validator = ValidationService(self.config)
        self.cache_manager = CacheManager(self.config)
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize SessionManager for conflict-free widget handling
        self.session_manager = create_session_manager("model_visualizer_service")
        
        # Initialize model services
        self.model_loader = SecureModelLoader(
            self.config, self.validator, self.cache_manager, self.performance_monitor
        )
        
        # Initialize display handlers
        self.display_handlers = {
            ModelType.PATTERN_NN: PatternNNDisplayHandler(self.validator, self.performance_monitor),
            ModelType.CLASSIC_ML: ClassicMLDisplayHandler(self.validator, self.performance_monitor)
        }
        
        # Initialize comparison service
        self.metrics_service = MetricsComparisonService(self.validator, self.performance_monitor)
        
        # Setup logging
        self.logger = get_dashboard_logger(f"{__name__}.ModelVisualizerService")
        
        self.logger.info("ModelVisualizerService initialized successfully")
    
    def run_dashboard(self) -> None:
        """Main method to run the dashboard application."""
        try:
            with self.performance_monitor.measure_time("Dashboard execution"):
                self._setup_sidebar_controls()
                self._display_main_content()
                
        except Exception as e:
            self.logger.error(f"Dashboard execution failed: {e}", exc_info=True)
            handle_streamlit_error(e, "Model Visualizer Dashboard")
    
    def _setup_sidebar_controls(self) -> None:
        """Setup sidebar controls and options."""
        st.sidebar.markdown("### ⚙️ Options")
        
        # Debug mode toggle
        debug_mode = st.sidebar.checkbox("Debug Mode", value=False, key="debug_mode")
        if debug_mode:
            st.sidebar.info("Debug mode enabled - performance metrics will be shown")
        
        # Cache controls
        if st.sidebar.button("🗑️ Clear Cache"):
            self.cache_manager.clear()
            st.sidebar.success("Cache cleared!")
            st.experimental_rerun()
        
        # Configuration display
        with st.sidebar.expander("📋 Configuration"):
            st.write(f"Max models to compare: {self.config.max_models_to_compare}")
            st.write(f"Allowed extensions: {', '.join(self.config.allowed_extensions)}")
            st.write(f"Cache TTL: {self.config.cache_ttl_seconds}s")
    
    def _display_main_content(self) -> None:
        """Display the main dashboard content."""
        st.title("🔍 Model Visualizer & Comparator")
        st.markdown("""
        **Enhanced Model Analysis Dashboard**
        
        This tool provides comprehensive visualization and comparison of your trained models.
        Select one or more models below to analyze their architecture, performance metrics,
        and compare them side-by-side.
        """)
        
        # Get available models
        model_files = self._get_available_models()
        
        if not model_files:
            st.warning("⚠️ No trained models found. Train and save a model first.")
            st.info("Supported formats: .pth (PyTorch), .joblib, .pkl (scikit-learn)")
            return
        
        # Model selection
        selected_files = self._handle_model_selection(model_files)
        
        if not selected_files:
            st.info("👆 Select one or more models above to begin analysis")
            return
        
        # Process and display selected models
        self._process_selected_models(selected_files)
    
    def _get_available_models(self) -> List[str]:
        """Get list of available model files with error handling."""
        try:
            with self.performance_monitor.measure_time("Listing available models"):
                model_manager = self.model_loader._get_model_manager()
                all_files = model_manager.list_models()
                
                # Filter for supported model files
                model_files = [
                    f for f in all_files 
                    if self.config.is_allowed_extension(f)
                ]
                
                self.logger.info(f"Found {len(model_files)} available models")
                return model_files
                
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            st.error("❌ Could not list models. Please check model directory and permissions.")
            return []
    
    def _handle_model_selection(self, model_files: List[str]) -> List[str]:
        """Handle model selection with validation."""
        try:
            # Display model count info
            st.info(f"📁 Found {len(model_files)} trained models")
              # Model selection widget
            selected_files = self.session_manager.create_multiselect(
                "**Select Model Files to Analyze**",
                options=model_files,
                default=[model_files[0]] if model_files else [],
                multiselect_name="model_selection",
                help=f"Select up to {self.config.max_models_to_compare} models for comparison"
            )
            
            if selected_files:
                # Validate selection
                try:
                    self.validator.validate_model_selection(selected_files, model_files)
                    self.logger.info(f"Selected {len(selected_files)} models for analysis")
                except (ValidationError, SecurityError) as e:
                    st.error(f"❌ Selection error: {str(e)}")
                    return []
            
            return selected_files
            
        except Exception as e:
            self.logger.error(f"Error in model selection: {e}")
            st.error("❌ Error in model selection interface")
            return []
    
    def _process_selected_models(self, selected_files: List[str]) -> None:
        """Process and display information for selected models."""
        all_metrics = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_file in enumerate(selected_files):
            try:
                # Update progress
                progress = (i + 1) / len(selected_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {model_file}...")
                
                # Process individual model
                metrics = self._process_single_model(model_file)
                
                if metrics:
                    all_metrics[model_file] = metrics
                    
            except Exception as e:
                self.logger.error(f"Error processing model {model_file}: {e}")
                st.error(f"❌ Error processing {model_file}: {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Generate comparison if multiple models
        if len(all_metrics) > 1:
            st.markdown("---")
            self.metrics_service.compare_metrics(all_metrics)
        elif len(selected_files) > 1:
            st.info("📊 Comparison will be available once multiple models are successfully loaded")
    
    def _process_single_model(self, model_file: str) -> Optional[Dict]:
        """Process and display a single model."""
        try:
            st.markdown(f"## 📋 Model Analysis: `{model_file}`")
            
            # Determine model type and load
            model_type = self._determine_model_type(model_file)
            
            if model_type == ModelType.PATTERN_NN:
                model, metadata = self.model_loader.load_model(PatternNN, model_file)
            else:
                model, metadata = self.model_loader.load_model(None, model_file)
            
            # Display model type
            st.markdown(f"**Detected Model Type:** `{model_type.value}`")
            
            if model is None:
                st.error(f"❌ Failed to load model: {model_file}")
                return None
            
            # Get appropriate display handler
            handler = self.display_handlers.get(model_type)
            
            if handler is None:
                st.error(f"❌ No display handler available for model type: {model_type.value}")
                return None
            
            # Display model using appropriate handler
            metrics = handler.display_model(model, metadata, model_file)
            
            # Add separator
            st.markdown("---")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error processing single model {model_file}: {e}")
            st.error(f"❌ Error processing {model_file}: {str(e)}")
            return None
    
    def _determine_model_type(self, model_file: str) -> ModelType:
        """Determine model type from filename."""
        # For initial type determination, we can use filename
        # The actual metadata will be checked during loading
        if model_file.endswith(".pth"):
            return ModelType.PATTERN_NN
        elif model_file.endswith((".joblib", ".pkl")):
            return ModelType.CLASSIC_ML
        else:
            return ModelType.UNKNOWN


# --- Main Dashboard Class ---

class ModelVisualizerDashboard:
    """
    Production-grade Model Visualizer Dashboard.
    
    This class provides a comprehensive, secure, and performance-optimized
    dashboard for visualizing and comparing machine learning models.
    
    Features:
    - Robust error handling and input validation
    - Security measures against malicious inputs
    - Performance optimization with caching
    - Modular architecture following SOLID principles    - Comprehensive logging and monitoring
    - Responsive UI with progressive loading
    - Multi-model comparison capabilities
    """
    
    def __init__(self):
        """Initialize the dashboard with all required services."""
        self.logger = get_dashboard_logger(__name__)
        
        try:
            # Initialize the main service
            self.service = ModelVisualizerService()
            
            # Initialize SessionManager for conflict-free widget handling
            self.session_manager = create_session_manager("model_visualizer")
            
            self.logger.info("ModelVisualizerDashboard initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dashboard: {e}", exc_info=True)
            raise
    
    def run(self) -> None:
        """
        Main entry point for the dashboard application.
        
        This method sets up the page configuration and runs the main dashboard logic
        with comprehensive error handling.
        """
        try:
            # Setup page configuration
            setup_page(
                title="🔍 Model Visualizer & Comparator",
                logger_name=__name__,
                sidebar_title="Model Analysis & Comparison"
            )
            
            # Run the main dashboard
            self.service.run_dashboard()
            
        except Exception as e:
            self.logger.error(f"Dashboard execution failed: {e}", exc_info=True)
            handle_streamlit_error(e, "Model Visualizer Dashboard")


# --- Entry Point ---

if __name__ == "__main__":
    try:
        dashboard = ModelVisualizerDashboard()
        dashboard.run()
    except Exception as e:
        # Final fallback error handling
        get_dashboard_logger(__name__).error(f"Critical error in main execution: {e}", exc_info=True)
        st.error("❌ Critical error occurred. Please check logs and try again.")
        st.exception(e)



