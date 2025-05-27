# Streamlit Dashboard Template Documentation

## Overview

This comprehensive template provides a production-ready foundation for creating Streamlit dashboard pages based on successful patterns identified from existing code. The template follows enterprise-grade practices with proper error handling, security validation, performance monitoring, and modular architecture.

## Features

### 🏗️ **Architecture Features**

- **SOLID Principles**: Modular class-based design with single responsibility
- **Error Handling**: Comprehensive error management with decorators and logging
- **Session Management**: Robust session state initialization and validation
- **Performance Monitoring**: Built-in timing and metrics collection
- **Security Validation**: Input sanitization and file upload security
- **Configuration Management**: Centralized config with validation
- **Caching Support**: TTL-based caching for performance optimization

### 🔒 **Security Features**

- File upload validation with size and type restrictions
- Input sanitization and validation
- Path traversal protection
- Memory usage monitoring
- Execution timeout protection

### 📊 **Production Features**

- Comprehensive logging and debugging support
- Performance metrics collection and display
- Error recovery and graceful degradation
- Memory usage optimization
- Resource cleanup and management

## Quick Start

### 1. Copy and Rename Template

```bash
cp templates/streamlit_dashboard_template.py dashboard_pages/my_dashboard.py
```

### 2. Basic Customization

Replace these key elements in your new file:

```python
# Change class name
class MyDashboard(TemplateDashboard):  # Instead of TemplateDashboard

# Update page configuration
PAGE_CONFIG = {
    "page_title": "My Dashboard",
    "page_icon": "🚀",  # Your icon
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Update session state keys
SESSION_STATE_KEYS = {
    "my_dashboard_initialized": False,
    "my_dashboard_config": {},
    "my_dashboard_data": None,
    # ... add your specific keys
}
```

### 3. Implement Main Functionality

Replace the `render_main_content()` method with your specific dashboard logic:

```python
@handle_dashboard_errors
def render_main_content(self) -> None:
    """Your specific dashboard functionality."""
    st.header("🚀 My Dashboard")
    
    # Your implementation here:
    # - Data visualization
    # - Interactive controls
    # - Analysis tools
    # - Custom functionality
    pass
```

## Template Structure

### 📁 **File Organization**

``` python
streamlit_dashboard_template.py
├── Imports (Standard → Third-party → Local)
├── Logger Setup
├── Page Configuration
├── Configuration Class
├── Validation Class  
├── Error Handling Decorator
├── Main Dashboard Class
├── Main Execution Functions
└── Customization Notes
```

### 🏛️ **Core Classes**

#### `TemplateConfig`

- Manages all configuration settings
- Provides validation for config values
- Handles directory setup and path management
- Supports dynamic configuration updates

#### `TemplateValidator`

- Validates DataFrames and data integrity
- Secures file uploads with size/type checking
- Validates numeric inputs with range constraints
- Provides extensible validation framework

#### `TemplateDashboard`

- Main dashboard controller class
- Orchestrates page rendering and state management
- Provides modular method organization
- Implements error handling and performance monitoring

## Configuration Management

### Default Configuration

The template includes comprehensive default settings:

```python
config = {
    # Data settings
    "max_records": 10000,
    "data_refresh_interval": 300,
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
    "max_execution_time": 30,
    
    # Directory settings
    "data_dir": Path("data"),
    "output_dir": Path("output"),
    "cache_dir": Path("cache"),
    "log_dir": Path("logs")
}
```

### Custom Configuration

Add your specific configuration in the `TemplateConfig` class:

```python
class MyDashboardConfig(TemplateConfig):
    def __init__(self):
        super().__init__()
        # Add your specific configuration
        self.config.update({
            "api_endpoint": "https://api.example.com",
            "refresh_rate": 60,
            "chart_theme": "plotly_dark"
        })
```

## Error Handling

### Decorator Pattern

The template uses a decorator for consistent error handling:

```python
@handle_dashboard_errors
def my_method(self):
    """Your method with automatic error handling."""
    # Your code here - errors are automatically caught and displayed
    pass
```

### Custom Exception Handling

Add specific error handling for your domain:

```python
try:
    # Your risky operation
    result = risky_operation()
except MyCustomException as e:
    logger.error(f"Custom error: {e}")
    st.error(f"Operation failed: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    st.error("An unexpected error occurred")
```

## Security Best Practices

### File Upload Security

The template validates uploaded files:

- File size limits (default 50MB)
- File type restrictions
- Filename sanitization
- Content validation

### Input Validation

All inputs are validated:

- Numeric range checking
- DataFrame structure validation
- Memory usage monitoring
- SQL injection prevention

### Data Sanitization

```python
# Validate user input
self.validator.validate_numeric_input(
    value=user_input,
    min_val=0,
    max_val=1000000
)

# Validate uploaded data
self.validator.validate_dataframe(
    df=uploaded_data,
    required_columns=["date", "value"]
)
```

## Performance Optimization

### Caching Strategy

```python
# Built-in caching configuration
CACHE_CONFIG = {
    "default_ttl": 300,  # 5 minutes
    "max_entries": 100,
    "clear_on_error": True
}

# Use Streamlit caching for expensive operations
@st.cache_data(ttl=300)
def expensive_computation(data):
    return process_data(data)
```

### Performance Monitoring

The template automatically tracks:

- Method execution times
- Memory usage
- Session state size
- Error frequencies

## Common Customization Patterns

### 1. Trading Dashboard

```python
class TradingDashboard(TemplateDashboard):
    def render_main_content(self):
        # Portfolio metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Value", "$125,000")
        with col2:
            st.metric("Daily P&L", "+$2,500")
        with col3:
            st.metric("Win Rate", "68%")
        
        # Trading charts
        self.render_price_chart()
        self.render_portfolio_composition()
        
    def render_price_chart(self):
        # Your chart implementation
        pass
```

### 2. Data Analysis Dashboard

```python
class DataAnalysisDashboard(TemplateDashboard):
    def render_main_content(self):
        # Data exploration tools
        self.render_data_explorer()
        self.render_statistical_analysis()
        self.render_correlation_matrix()
        
    def render_data_explorer(self):
        # Your data exploration tools
        pass
```

### 3. Model Monitoring Dashboard

```python
class ModelMonitoringDashboard(TemplateDashboard):
    def render_main_content(self):
        # Model performance metrics
        self.render_model_metrics()
        self.render_prediction_analysis()
        self.render_feature_importance()
        
    def render_model_metrics(self):
        # Your model monitoring implementation
        pass
```

## Integration with Existing Codebase

### Import Existing Utilities

The template expects these utilities to be available:

```python
from utils.logger import get_logger
from core.dashboard_utils import (
    setup_page, 
    render_performance_metrics,
    validate_session_state,
    safe_execute,
    DashboardException
)
from utils.io import ensure_directory_exists, safe_file_read, safe_file_write
```

### Adapt to Your Project Structure

Update import paths based on your project structure:

```python
# If your utilities are in different locations
from your_project.utils.logger import get_logger
from your_project.core.dashboard_utils import setup_page
```

## Testing Guidelines

### Unit Testing Template Methods

```python
import unittest
from unittest.mock import patch, MagicMock
from your_dashboard import MyDashboard

class TestMyDashboard(unittest.TestCase):
    def setUp(self):
        self.dashboard = MyDashboard()
    
    def test_config_validation(self):
        # Test configuration validation
        self.assertTrue(self.dashboard.config.get("max_records") > 0)
    
    @patch('streamlit.session_state')
    def test_session_state_initialization(self, mock_session_state):
        # Test session state setup
        self.dashboard._initialize_session_state()
        # Add assertions
```

### Integration Testing

```python
def test_full_dashboard_flow():
    # Test complete dashboard workflow
    dashboard = MyDashboard()
    
    # Test initialization
    dashboard._initialize_session_state()
    
    # Test configuration
    assert dashboard.config.get("max_records") == 10000
    
    # Test validation
    test_df = pd.DataFrame({"col1": [1, 2, 3]})
    assert dashboard.validator.validate_dataframe(test_df)
```

## Deployment Considerations

### Environment Configuration

```python
# Add environment-specific settings
if os.getenv("ENVIRONMENT") == "production":
    config.update({
        "show_debug_info": False,
        "enable_profiling": False,
        "max_records": 50000
    })
elif os.getenv("ENVIRONMENT") == "development":
    config.update({
        "show_debug_info": True,
        "enable_profiling": True,
        "max_records": 1000
    })
```

### Health Checks

```python
def health_check(self) -> bool:
    """Verify dashboard health for monitoring."""
    try:
        # Check critical dependencies
        assert self.config is not None
        assert st.session_state.template_initialized
        
        # Check directory access
        for dir_path in ["data_dir", "output_dir", "cache_dir"]:
            directory = self.config.get(dir_path)
            assert directory.exists() and directory.is_dir()
        
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
```

## Best Practices Summary

### ✅ **Do's**

- Use the error handling decorator for all dashboard methods
- Validate all user inputs and uploaded files
- Initialize session state properly with default values
- Log important operations and errors
- Use configuration management for settings
- Monitor performance with built-in metrics
- Follow the modular class structure
- Document your custom methods thoroughly

### ❌ **Don'ts**

- Don't bypass validation for user inputs
- Don't store sensitive data in session state
- Don't ignore error handling patterns
- Don't hardcode configuration values
- Don't skip performance monitoring
- Don't mix business logic with UI rendering
- Don't forget to update documentation

## Support and Extensions

### Getting Help

1. Check the extensive code comments in the template
2. Review existing dashboard implementations in your codebase
3. Refer to the customization notes at the end of the template file
4. Use the debugging features built into the template

### Contributing Improvements

When you develop useful patterns or enhancements:

1. Document the pattern clearly
2. Add proper error handling and validation
3. Include performance considerations
4. Update this documentation
5. Share with the team for template evolution

## Template Evolution

This template is designed to evolve with your project needs. Key areas for enhancement:

- Additional validation patterns
- More sophisticated error handling
- Enhanced performance monitoring
- Advanced caching strategies
- Improved security features
- Better testing utilities
- Enhanced documentation
