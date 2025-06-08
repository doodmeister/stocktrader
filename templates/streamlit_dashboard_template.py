"""
Streamlit Dashboard Page Template

This template provides a comprehensive foundation for creating production-grade 
Streamlit dashboard pages based on successful patterns from existing code.

USAGE INSTRUCTIONS:
1. Copy this template to create a new dashboard page
2. Replace 'template' with your actual page name throughout the file
3. Update the PAGE_CONFIG dictionary with your page settings
4. Implement your specific validation logic in TemplateValidator
5. Add your main dashboard functionality in the render_main_content method
6. Update the class docstring and method documentation
7. Add any additional imports specific to your page functionality

TEMPLATE FEATURES:
- Session state initialization with validation
- Error handling with decorators and try-catch blocks
- Performance monitoring and timing
- Security validation and input sanitization
- Caching configuration with TTL support
- Modular class-based architecture following SOLID principles
- Comprehensive logging and debugging support
- Configuration management with validation
- Directory setup and file handling utilities
- Production-ready structure for enterprise applications

REQUIREMENTS:
- Ensure all dependencies are installed (streamlit, pandas, etc.)
- Core utilities must be available (core.dashboard_utils, utils.logger, utils.io)
- Proper logging configuration in utils.logger
- Dashboard utilities and error handling in core.dashboard_utils
"""

# =============================================================================
# IMPORTS - Standard Library First, Third-party Second, Local Last
# =============================================================================

# Standard library imports
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
from functools import wraps

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Local imports - Update paths based on your project structure
try:
    from utils.logger import get_logger
    from core.streamlit.dashboard_utils import (
        setup_page, 
        render_performance_metrics,
        validate_session_state,
        safe_execute,
        DashboardException
    )
    from utils.io import ensure_directory_exists, safe_file_read, safe_file_write
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# =============================================================================
# LOGGER SETUP
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

PAGE_CONFIG = {
    "page_title": "Template Dashboard",  # Update with your page title
    "page_icon": "📊",  # Update with your preferred icon
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Session state keys for this page
SESSION_STATE_KEYS = {
    "template_initialized": False,
    "template_config": {},
    "template_data": None,
    "template_cache": {},
    "template_errors": [],
    "template_last_update": None,
    "template_performance_metrics": {}
}

# Cache configuration
CACHE_CONFIG = {
    "default_ttl": 300,  # 5 minutes
    "max_entries": 100,
    "clear_on_error": True
}

# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

class TemplateConfig:
    """
    Configuration management for the template dashboard page.
    
    This class handles all configuration settings, validation, and provides
    defaults for the dashboard functionality.
    """
    
    def __init__(self):
        """Initialize configuration with default values."""
        self.config = {
            # Data settings
            "max_records": 10000,
            "data_refresh_interval": 300,  # 5 minutes
            "enable_caching": True,
            
            # UI settings
            "show_debug_info": False,
            "show_performance_metrics": True,
            "theme": "default",
            
            # Security settings
            "validate_inputs": True,
            "max_file_size": 50 * 1024 * 1024,  # 50MB
            "allowed_file_types": [".csv", ".xlsx", ".json"],
            
            # Performance settings
            "enable_profiling": False,
            "max_execution_time": 30,  # seconds
            
            # Directory settings - Update paths for your project
            "data_dir": Path("data"),
            "output_dir": Path("output"),
            "cache_dir": Path("cache"),
            "log_dir": Path("logs")
        }
        
        self._validate_config()
        self._setup_directories()
    
    def _validate_config(self) -> None:
        """Validate configuration settings."""
        try:
            # Validate numeric values
            if self.config["max_records"] <= 0:
                raise ValueError("max_records must be positive")
            
            if self.config["data_refresh_interval"] < 0:
                raise ValueError("data_refresh_interval must be non-negative")
            
            if self.config["max_file_size"] <= 0:
                raise ValueError("max_file_size must be positive")
            
            logger.info("Configuration validation successful")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise DashboardException(f"Invalid configuration: {e}")
    
    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        try:
            for dir_key in ["data_dir", "output_dir", "cache_dir", "log_dir"]:
                directory = self.config[dir_key]
                ensure_directory_exists(directory)
                logger.debug(f"Directory ensured: {directory}")
            
        except Exception as e:
            logger.error(f"Directory setup failed: {e}")
            raise DashboardException(f"Failed to setup directories: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.config.get(key, default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        try:
            self.config.update(updates)
            self._validate_config()
            logger.info(f"Configuration updated: {list(updates.keys())}")
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            raise DashboardException(f"Failed to update configuration: {e}")

# =============================================================================
# VALIDATION CLASS
# =============================================================================

class TemplateValidator:
    """
    Input validation and security checks for the template dashboard.
    
    This class provides comprehensive validation for user inputs, file uploads,
    and data integrity to ensure security and reliability.
    """
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
        """
        Validate a pandas DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            bool: True if valid, raises exception if invalid
            
        Raises:
            DashboardException: If validation fails
        """
        try:
            if df is None or df.empty:
                raise DashboardException("DataFrame is None or empty")
            
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    raise DashboardException(f"Missing required columns: {missing_columns}")
            
            # Check for reasonable size
            if len(df) > 1000000:  # 1M rows
                raise DashboardException("DataFrame too large (>1M rows)")
            
            if df.memory_usage(deep=True).sum() > 500 * 1024 * 1024:  # 500MB
                raise DashboardException("DataFrame too large (>500MB)")
            
            logger.debug(f"DataFrame validation successful: {df.shape}")
            return True
            
        except Exception as e:
            logger.error(f"DataFrame validation failed: {e}")
            raise DashboardException(f"Invalid DataFrame: {e}")
    
    @staticmethod
    def validate_file_upload(uploaded_file) -> bool:
        """
        Validate uploaded file for security and size constraints.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            bool: True if valid, raises exception if invalid
            
        Raises:
            DashboardException: If validation fails
        """
        try:
            if uploaded_file is None:
                raise DashboardException("No file uploaded")
            
            # Check file size
            if uploaded_file.size > 50 * 1024 * 1024:  # 50MB
                raise DashboardException("File too large (>50MB)")
            
            # Check file extension
            file_ext = Path(uploaded_file.name).suffix.lower()
            allowed_extensions = [".csv", ".xlsx", ".json", ".txt"]
            if file_ext not in allowed_extensions:
                raise DashboardException(f"File type not allowed. Allowed: {allowed_extensions}")
            
            # Basic filename validation
            if any(char in uploaded_file.name for char in ['<', '>', ':', '"', '|', '?', '*']):
                raise DashboardException("Invalid characters in filename")
            
            logger.debug(f"File validation successful: {uploaded_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            raise DashboardException(f"Invalid file: {e}")
    
    @staticmethod
    def validate_numeric_input(value: Union[int, float], min_val: Optional[float] = None, 
                             max_val: Optional[float] = None) -> bool:
        """
        Validate numeric input with optional range constraints.
        
        Args:
            value: Numeric value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            bool: True if valid, raises exception if invalid
            
        Raises:
            DashboardException: If validation fails
        """
        try:
            if not isinstance(value, (int, float)):
                raise DashboardException(f"Value must be numeric, got {type(value)}")
            
            if pd.isna(value) or not np.isfinite(value):
                raise DashboardException("Value must be finite")
            
            if min_val is not None and value < min_val:
                raise DashboardException(f"Value {value} below minimum {min_val}")
            
            if max_val is not None and value > max_val:
                raise DashboardException(f"Value {value} above maximum {max_val}")
            
            return True
            
        except Exception as e:
            logger.error(f"Numeric validation failed: {e}")
            raise DashboardException(f"Invalid numeric value: {e}")

# =============================================================================
# ERROR HANDLING DECORATOR
# =============================================================================

def handle_dashboard_errors(func: Callable) -> Callable:
    """
    Decorator for handling dashboard errors gracefully.
    
    This decorator provides consistent error handling across dashboard methods,
    logging errors and displaying user-friendly messages.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Callable: Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log performance metrics
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            
            # Store metrics in session state if available
            if hasattr(st.session_state, 'template_performance_metrics'):
                st.session_state.template_performance_metrics[func.__name__] = execution_time
            
            return result
            
        except DashboardException as e:
            logger.error(f"Dashboard error in {func.__name__}: {e}")
            st.error(f"Dashboard Error: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            st.error("An unexpected error occurred. Please check the logs.")
            return None
    
    return wrapper

# =============================================================================
# MAIN DASHBOARD CLASS
# =============================================================================

class TemplateDashboard:
    """
    Main template dashboard class following SOLID principles.
    
    This class provides a complete template for creating production-grade
    Streamlit dashboard pages with proper error handling, validation,
    performance monitoring, and modular architecture.
    
    Features:
    - Session state management with validation
    - Configuration management with validation
    - Error handling with decorators and logging
    - Performance monitoring and metrics
    - Security validation and input sanitization
    - Caching with TTL support
    - Modular method organization
    - Comprehensive documentation
    
    Usage:
        dashboard = TemplateDashboard()
        dashboard.run()
    """
    
    def __init__(self):
        """Initialize the template dashboard."""
        self.config = TemplateConfig()
        self.validator = TemplateValidator()
        self._initialize_session_state()
        
        logger.info("Template dashboard initialized")
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state with default values."""
        try:
            for key, default_value in SESSION_STATE_KEYS.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value
                    logger.debug(f"Initialized session state key: {key}")
            
            # Mark as initialized
            st.session_state.template_initialized = True
            st.session_state.template_last_update = datetime.now()
            
            logger.info("Session state initialization completed")
            
        except Exception as e:
            logger.error(f"Session state initialization failed: {e}")
            raise DashboardException(f"Failed to initialize session state: {e}")
    
    @handle_dashboard_errors
    def setup_page_layout(self) -> None:
        """Setup the basic page layout and configuration."""
        # Use the centralized page setup utility
        setup_page(**PAGE_CONFIG)
        
        # Add custom CSS if needed
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        </style>
        """, unsafe_allow_html=True)
        
        logger.debug("Page layout setup completed")
    
    @handle_dashboard_errors
    def render_header(self) -> None:
        """Render the page header with title and description."""
        st.markdown("""
        <div class="main-header">
            <h1>📊 Template Dashboard</h1>
            <p>A comprehensive template for Streamlit dashboard development</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add navigation or breadcrumbs if needed
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("🚀 **Template Features**: Session Management, Error Handling, Performance Monitoring, Security Validation")
    
    @handle_dashboard_errors
    def render_sidebar(self) -> None:
        """Render the sidebar with controls and configuration options."""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Configuration controls
            with st.expander("🔧 Settings", expanded=False):
                show_debug = st.checkbox(
                    "Show Debug Information",
                    value=self.config.get("show_debug_info", False),
                    help="Display detailed debug information and logs"
                )
                
                show_metrics = st.checkbox(
                    "Show Performance Metrics",
                    value=self.config.get("show_performance_metrics", True),
                    help="Display performance timing metrics"
                )
                
                max_records = st.number_input(
                    "Max Records",
                    min_value=100,
                    max_value=100000,
                    value=self.config.get("max_records", 10000),
                    step=1000,
                    help="Maximum number of records to process"
                )
                
                # Update configuration
                self.config.update({
                    "show_debug_info": show_debug,
                    "show_performance_metrics": show_metrics,
                    "max_records": max_records
                })
            
            # Data upload section
            with st.expander("📁 Data Upload", expanded=False):
                uploaded_file = st.file_uploader(
                    "Upload Data File",
                    type=["csv", "xlsx", "json"],
                    help="Upload a data file for analysis"
                )
                
                if uploaded_file is not None:
                    try:
                        self.validator.validate_file_upload(uploaded_file)
                        st.success(f"✅ File validated: {uploaded_file.name}")
                        # Process the file here
                        self._process_uploaded_file(uploaded_file)
                        
                    except DashboardException as e:
                        st.error(f"❌ File validation failed: {e}")
            
            # Session state information
            if self.config.get("show_debug_info", False):
                with st.expander("🔍 Debug Information", expanded=False):
                    st.write("**Session State Keys:**")
                    for key in SESSION_STATE_KEYS.keys():
                        if key in st.session_state:
                            st.write(f"- {key}: {type(st.session_state[key])}")
                    
                    st.write("**Last Update:**")
                    st.write(st.session_state.get("template_last_update", "Not set"))
    
    def _process_uploaded_file(self, uploaded_file) -> None:
        """Process uploaded file and store in session state."""
        try:
            # Read file based on type
            file_ext = Path(uploaded_file.name).suffix.lower()
            
            if file_ext == ".csv":
                df = pd.read_csv(uploaded_file)
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(uploaded_file)
            elif file_ext == ".json":
                df = pd.read_json(uploaded_file)
            else:
                raise DashboardException(f"Unsupported file type: {file_ext}")
            
            # Validate the DataFrame
            self.validator.validate_dataframe(df)
            
            # Store in session state
            st.session_state.template_data = df
            st.session_state.template_last_update = datetime.now()
            
            logger.info(f"File processed successfully: {uploaded_file.name}, shape: {df.shape}")
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            raise DashboardException(f"Failed to process file: {e}")
    
    @handle_dashboard_errors
    def render_main_content(self) -> None:
        """
        Render the main dashboard content.
        
        *** REPLACE THIS METHOD WITH YOUR SPECIFIC FUNCTIONALITY ***
        
        This is where you implement your specific dashboard features:
        - Data visualization
        - Interactive controls
        - Analysis tools
        - Reports and exports
        - Custom functionality
        
        Example implementations:
        - Trading dashboard: render trading controls, charts, portfolio metrics
        - Data analysis: render data exploration tools, statistical analysis
        - Model monitoring: render model performance metrics, predictions
        """
        
        # Main content area
        st.header("📈 Main Dashboard Content")
        
        # Check if data is available
        if st.session_state.template_data is not None:
            df = st.session_state.template_data
            
            # Data overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(df))
            
            with col2:
                st.metric("Columns", len(df.columns))
            
            with col3:
                memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                st.metric("Memory (MB)", f"{memory_mb:.2f}")
            
            with col4:
                st.metric("Data Types", df.dtypes.nunique())
            
            # Data display
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head(100), use_container_width=True)
            
            # Basic analysis
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                with st.expander("📈 Quick Analysis", expanded=False):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_cols) >= 1:
                        col = st.selectbox("Select Column for Analysis", numeric_cols)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Basic statistics
                            st.write("**Statistics:**")
                            stats = df[col].describe()
                            st.dataframe(stats)
                        
                        with col2:
                            # Simple histogram
                            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # No data available - show placeholder content
            st.info("👆 Upload a data file using the sidebar to get started")
            
            # Show template examples
            with st.expander("📝 Template Examples", expanded=True):
                st.write("**This template provides:**")
                
                examples = [
                    "✅ Session state management with validation",
                    "✅ Error handling with decorators and logging",
                    "✅ Performance monitoring and metrics",
                    "✅ Security validation for file uploads",
                    "✅ Configuration management with validation",
                    "✅ Modular class-based architecture",
                    "✅ Comprehensive documentation",
                    "✅ Production-ready structure"
                ]
                
                for example in examples:
                    st.write(example)
                
                st.code("""
# Example usage in your custom dashboard:

class MyCustomDashboard(TemplateDashboard):
    def render_main_content(self):
        # Your custom implementation here
        st.header("My Custom Dashboard")
        
        # Add your specific functionality:
        # - Custom visualizations
        # - Interactive controls
        # - Analysis tools
        # - Data processing
        # - Export functionality
        
        pass

# Run your dashboard
dashboard = MyCustomDashboard()
dashboard.run()
                """, language="python")
    
    @handle_dashboard_errors
    def render_footer(self) -> None:
        """Render the page footer with performance metrics and information."""
        st.markdown("---")
        
        # Performance metrics
        if self.config.get("show_performance_metrics", True):
            render_performance_metrics(st.session_state.get("template_performance_metrics", {}))
        
        # Footer information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.caption("🔧 Template Dashboard v1.0")
        
        with col2:
            if st.session_state.template_last_update:
                st.caption(f"🕒 Last Update: {st.session_state.template_last_update.strftime('%H:%M:%S')}")
        
        with col3:
            st.caption("📊 Built with Streamlit")
    
    @handle_dashboard_errors
    def run(self) -> None:
        """
        Main entry point to run the dashboard.
        
        This method orchestrates the entire dashboard rendering process:
        1. Setup page layout and configuration
        2. Render header with title and navigation
        3. Render sidebar with controls and configuration
        4. Render main content area
        5. Render footer with metrics and information
        """
        try:
            # Validate session state
            validate_session_state(SESSION_STATE_KEYS.keys())
            
            # Setup and render dashboard components
            self.setup_page_layout()
            self.render_header()
            self.render_sidebar()
            self.render_main_content()
            self.render_footer()
            
            logger.info("Dashboard rendering completed successfully")
            
        except Exception as e:
            logger.error(f"Dashboard execution failed: {e}", exc_info=True)
            st.error("❌ Dashboard failed to load. Please refresh the page or check the logs.")
            
            # Show error details in debug mode
            if self.config.get("show_debug_info", False):
                st.exception(e)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the template dashboard.
    
    This is the entry point for the dashboard application. It creates an instance
    of the TemplateDashboard class and runs it.
    
    Usage:
        Run this script directly or import and call main() from another module.
    """
    try:
        # Create and run dashboard
        dashboard = TemplateDashboard()
        dashboard.run()
        
    except Exception as e:
        # Final fallback error handling
        logger.error(f"Critical error in main: {e}", exc_info=True)
        st.error("❌ Critical error occurred. Please contact support.")
        st.stop()

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()

# =============================================================================
# TEMPLATE CUSTOMIZATION NOTES
# =============================================================================

"""
TO CUSTOMIZE THIS TEMPLATE FOR YOUR SPECIFIC DASHBOARD:

1. **Replace Template Names:**
   - Change 'Template' to your actual dashboard name throughout
   - Update PAGE_CONFIG with your specific settings
   - Modify SESSION_STATE_KEYS for your data needs

2. **Update Imports:**
   - Add specific libraries for your dashboard functionality
   - Import your custom modules and utilities
   - Update import paths based on your project structure

3. **Customize Configuration:**
   - Modify TemplateConfig class with your specific settings
   - Add validation rules specific to your domain
   - Update directory paths and file handling

4. **Implement Main Functionality:**
   - Replace render_main_content() with your specific features
   - Add custom methods for your dashboard logic
   - Implement data processing and visualization

5. **Enhance Validation:**
   - Add domain-specific validation in TemplateValidator
   - Implement custom security checks
   - Add business logic validation

6. **Customize UI:**
   - Update CSS styles for your branding
   - Modify layout and components
   - Add custom interactive elements

7. **Add Error Handling:**
   - Implement domain-specific error handling
   - Add custom exception types
   - Enhance logging for your use case

8. **Performance Optimization:**
   - Add caching for expensive operations
   - Implement data streaming for large datasets
   - Add progress indicators for long operations

9. **Testing and Documentation:**
   - Add unit tests for your custom methods
   - Update documentation and docstrings
   - Add usage examples and tutorials

10. **Deployment Preparation:**
    - Add environment-specific configuration
    - Implement health checks and monitoring
    - Add deployment scripts and documentation
"""
