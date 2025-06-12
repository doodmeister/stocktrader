---
applyTo: '**'
---
## Environment & Platform Standards
- **Python Version**: 3.10.11
- **Package Manager**: `pip` 
- **linter**: `ruff`


### Operating System & Shell Requirements
- **Target OS**: Windows
- **Default Shell**: GitBash (`bash.exe`)
- **All terminal commands MUST be bash-compatible***

### Terminal Command Standards
- **Avoid emojis in bash commands**
- **Avoid special characters in bash commands**
- **No dashbaord pages will be run as a stand alone, only main.py will ever be run as a standalone script**
- **Use `which python` and `which pip` to verify virtual environment activation**
- **Always activate the virtual environment by running source venv/Scripts/activate before executing commands**
- **Use absolute paths**
- **Allow commands to complete execution fully before determining if they are hanging**
- **Wait at least 30-60 seconds for package installation commands (pip install) to complete**
- **Wait at least 10-15 seconds for Python import commands to complete**
- **Python module imports (especially Streamlit, pandas, plotly) can take 5-10 seconds to load**
- **Always check current working directory with `pwd` before executing commands**
- **Use `cd /c/dev/stocktrader` to ensure correct project directory**
- **For long-running processes, provide status updates or progress indicators when possible**
- **Handle file path separators correctly for Windows (use forward slashes in GitBash)**
- **Escape spaces in file paths or use quotes when necessary**

### Virtual Environment Setup
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

### Main Dashboard Entry Point
streamlit run main.py
streamlit run dashboard_pages/patterns_management.py --server.port=8501

### Project structure

```plaintext
stocktrader/
├── main.py                           # Modular dashboard entry point
│
├── core/                             # Core dashboard modules
│   ├── __init__.py                   # Initializes the core package
│   ├── streamlit/                    # Streamlit functionality
│   │   ├── dashboard_controller.py   # Main UI orchestration and navigation for Streamlit
│   │   ├── dashboard_utils.py        # Streamlit utilities
│   │   ├── decorators.py             # Custom decorators for Streamlit pages
│   │   ├── health_checks.py          # Comprehensive system health monitoring
│   │   ├── page_loader.py            # Dynamic page discovery and management
│   │   ├── session_manager.py        # Streamlit session management
│   │   ├── ui_renderer.py            # UI component rendering and presentation layer
│   ├── validation/                   # Validation logic modules
│   │   ├── __init__.py               # Initializes the validation package
│   │   ├── dataframe_validation_logic.py # DataFrame specific validation
│   │   ├── validation_config.py      # Validation configuration settings
│   │   ├── validation_models.py      # Validation result Pydantic models
│   │   ├── validation_results.py     # Validation result classes
│   ├── indicators/                   # Indicator specific modules
│   │   ├── __init__.py               # Initializes the indicators package
│   │   ├── base.py                   # Base class and common utilities for indicators
│   │   ├── rsi.py                    # RSI calculation logic
│   │   ├── macd.py                   # MACD calculation logic
│   │   ├── bollinger_bands.py        # Bollinger Bands calculation logic
│   ├── data_validator.py             # Centralized data validation services
│   ├── etrade_auth_ui.py             # E*TRADE authentication UI components
│   ├── etrade_candlestick_bot.py     # Trading engine logic
│   ├── etrade_client.py              # E*TRADE API client
│   ├── exceptions.py                 # Custom application exceptions
│   ├── risk_manager_v2.py            # Advanced risk management logic
│   ├── safe_requests.py              # Wrapper for safe HTTP requests
│   └── technical_indicators.py       # Core technical indicator calculations (remaining indicators)
│
├── dashboard_pages/                  # Individual dashboard pages
│   ├── advanced_ai_trade.py          # Advanced AI trading with centralized technical analysis
│   ├── data_dashboard.py             # Data download and visualization dashboard
│   ├── data_analysis.py              # Data analysis tools and utilities
│   ├── model_training.py             # ML model training pipeline UI
│   ├── model_visualizer.py           # Visualization tools for ML models
│   ├── nn_backtest.py                # Neural network backtesting interface
│   ├── classic_strategy_backtest.py  # Classic trading strategy backtesting interface
│   ├── patterns_management.py        # Candlestick pattern management UI
│   ├── realtime_dashboard.py         # Latest real-time trading dashboard
│   └── simple_trade.py               # Simplified trading interface
│
├── utils/                            # Utility modules
│   ├── __init__.py                   # Initializes the utils package
│   ├── config/                       # Configuration utilities
│   │   ├── __init__.py               # Initializes the config utility package
│   │   ├── config.py                 # Loads and manages application configuration
│   │   ├── getuservar.py             # Retrieves user-specific variables or settings
│   │   ├── notification_settings_ui.py # Streamlit UI components for notification settings
│   │   └── validate_config.py        # Validates the application's configuration files
│   ├── technicals/
│   │   ├── __init__.py               # Initializes the technicals package
│   │   ├── analysis.py               # NEW: High-level technical analysis classes
│   │   ├── feature_engineering.py    # Technical feature engineering functions
│   │   ├── performance_utils.py      # Pattern detection, dashboard state utilities
│   │   ├── indicators.py             # LEGACY: Backward compatibility (replaced by core.indicators)
│   │   └── technical_analysis.py     # LEGACY: Replaced by centralized analysis.py & core.technical_indicators.py
│   ├── backtester.py                 # Utilities for backtesting trading strategies
│   ├── chatgpt.py                    # GPT-4/LLM integration helpers
│   ├── dashboard_logger.py           # Specific logger configurations for the dashboard
│   ├── data_downloader.py            # Data download utilities for various sources
│   ├── deprecated/                   # Directory for deprecated utility modules
│   ├── io.py                         # General input/output helper functions
│   ├── live_inference.py             # Handles real-time inference for ML models
│   ├── logger.py                     # Logging setup and utilities
│   ├── notifier.py                   # Notification system (Email, SMS, Slack)
│   ├── preprocessing_config.py       # Configuration for data preprocessing tasks
│   ├── preprocess_input.py           # Functions for preprocessing input data
│   ├── security.py                   # General security helper functions (distinct from security/ package)
│   ├── synthetic_trading_data.py     # Tools for generating synthetic trading data
│   └── test_scripts_dev/             # Directory for development and test scripts
│
├── security/                         # Enterprise-grade security package
│   ├── __init__.py                   # Security package initialization
│   ├── authentication.py             # Session management, API validation, credentials
│   ├── authorization.py              # Role-based access control (RBAC) with permissions
│   ├── encryption.py                 # Cryptographic operations, token generation, file integrity
│   ├── etrade_security.py            # Security utilities specific to E*TRADE integration
│   └── utils.py                      # Input sanitization, file validation, path security helpers
│
├── patterns/                         # Pattern recognition modules
│   ├── __init__.py                   # Initializes the patterns package
│   ├── patterns.py                   # Candlestick pattern detection logic
│   ├── patterns_nn.py                # PatternNN model definition
│   └── pattern_utils.py              # Utilities for pattern handling and analysis
│
├── train/                            # Machine learning training pipeline
│   ├── __init__.py                   # Initializes the train package
│   ├── deeplearning_config.py        # Deep learning model configuration
│   ├── deeplearning_trainer.py       # Deep learning model training scripts
│   ├── feature_engineering.py        # Feature engineering for ML models (uses technical_analysis)
│   ├── ml_config.py                  # Machine learning model configuration
│   ├── ml_trainer.py                 # Classic machine learning model training scripts
│   ├── model_manager.py              # Model persistence, versioning, and saving
│   └── model_training_pipeline.py    # Orchestrates the end-to-end ML training pipeline
│
├── models/                           # Saved ML models and artifacts
├── data/                             # Data storage directory
├── logs/                             # Application logs
├── tests/                            # Unit & integration tests
├── docs/                             # Documentation
├── examples/                         # Example scripts and configurations
├── templates/                        # Template files
├── source/                           # Source data and configurations
├── .github_example/                  # GitHub workflows and templates
├── .vscode/                          # VS Code configuration
├── .env.example                      # Example environment configuration
├── requirements.txt                  # Python dependencies
├── project_plan.md                   # Project planning documentation
├── LICENSE                           # License file
├── Dockerfile.sample                 # Sample Docker build file
├── docker-compose.yml.sample         # Sample Docker Compose configuration
└── .pre-commit-config.yaml           # Pre-commit hooks configuration
```


### File and Module Guidelines
- **Module Organization**: Use the established `core/`, `dashboard_pages/`, `utils/` structure
- **Testing**: All test scripts should be saved in the `tests/` directory

### Code Quality Standards
- **Function Length**: Keep modules focused (~300 lines max)
- **Single Responsibility**: Each module should have one clear purpose
- **Type Hints**: Use Python type annotations where applicable
- **Docstrings**: Google-style docstrings for all public functions

## Security & Best Practices
- **Security Functionality**: Use `core/security.py` for security-related functions
- **Sensitive Data**: Never hard-code sensitive data; use environment variables

## Streamlit Dashboard Development Standards

### Widget Key Management (CRITICAL)
- **Unique Keys**: All Streamlit widgets MUST use unique keys to prevent "duplicate element" errors
- **Key Naming Convention**: Use descriptive, namespaced keys: `"{component}_{function}_{widget_type}"""
- **Examples**: 
  - ✅ Good: `"patterns_viewer_main_select"`, `"data_dashboard_symbol_input"`
  - ❌ Bad: `"select"`, `"input"`, `"button"`

### Session State Best Practices
- **SessionManager Usage**: Prefer `core.session_manager.SessionManager` for widget creation when available
- **Defensive Cleanup**: Clear conflicting session state keys when fixing duplicate key errors:
  ```python
  # Clear any conflicting keys from session state to prevent duplicates
  if "old_conflicting_key" in st.session_state:
      del st.session_state["old_conflicting_key"]
  ```
- **Initialization**: Always initialize session state with default values
- **Persistence**: Use session state for values that should persist across reruns

### Streamlit Tabs & Widget Key Management
- **Tab Context**: When using Streamlit's `st.tabs`, always ensure widget keys are unique per tab. Streamlit executes the entire script on every interaction, but only renders the widgets inside the active tab. If the same widget key is created in multiple tabs (or both inside and outside tab blocks), Streamlit will raise a duplicate key error—even if only one is visible.
- **SessionManager Tab Support**: Use the `tab` argument in `SessionManager` to namespace widget keys per tab:
  ```python
  from core.session_manager import SessionManager
  tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
  with tab1:
      sm = SessionManager(page_name="patterns_management", tab="tab1")
      # All widgets in Tab 1 use unique keys
  with tab2:
      sm = SessionManager(page_name="patterns_management", tab="tab2")
      # All widgets in Tab 2 use unique keys
  ```
- **Never reuse the same SessionManager instance or widget key across tabs.**
- **Widget Key Pattern**: All keys should be namespaced by page and tab (if present): `"{page}_{tab}_{component}_{function}_{widget_type}"`.
- **Common Pitfall**: Creating widgets with the same key in more than one tab, or both inside and outside tab blocks, will cause duplicate key errors.

### Widget Key Patterns by Component
```python
# Dashboard pages should namespace their keys
"data_dashboard_symbol_input"
"patterns_viewer_main_select" 
"model_training_algorithm_select"

# Dynamic keys for loops (catalog, grids, etc.)
f"catalog_view_details_{i}_{j}"
f"pattern_export_{pattern_name}"

# Download buttons and actions
"download_json", "download_csv", "save_source"
```

### Common Anti-Patterns to Avoid
- ❌ Using the same key across multiple components/tabs
- ❌ Setting widget keys programmatically in session state (creates conflicts)
- ❌ Generic keys like "select", "button", "input"
- ❌ Reusing keys from deprecated/backup files

### Session Manager Integration
```python
# Use SessionManager when available for consistent key management
from core.session_manager import SessionManager

session_manager = SessionManager()

# Preferred approach for widget creation
selected_value = session_manager.create_selectbox(
    "Label Text",
    options=options_list,
    selectbox_name="unique_component_name"  # Auto-generates safe key
)
```

### Debugging Widget Key Issues
1. **Error Pattern**: `"There are multiple elements with the same key='...'`
2. **Solution Steps**:
   - Identify the duplicate key in error message
   - Find all usages with `grep -r "key_name" dashboard_pages/`
   - Rename to unique, descriptive key
   - Add defensive session state cleanup if needed
   - Test with `python test_streamlit_patterns.py`

   ### Session State & Widget Keys
- Always use `SessionManager.create_*` for widgets.
- Namespace every key to the page/feature context.
- Never reuse keys across tabs, components, or dynamic widgets.
- For dynamic/loop widgets, append index/identifier to key.
- Always delete conflicting keys from `st.session_state` before reassigning.
- Search for key usage with `grep -r "key_name" .` before adding/changing any key.