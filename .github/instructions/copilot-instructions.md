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
- **Always activate the virtual environment by running source venv/Scripts/activate before executing commands**
- **Use absolute paths**
- **Allow commands to complete execution fully before determining if they are hanging**
- **Wait at least 30-60 seconds for package installation commands (pip install) to complete**
- **Wait at least 10-15 seconds for Python import commands to complete**
- **Python module imports (especially Streamlit, pandas, plotly) can take 5-10 seconds to load**
- **Always check current working directory with `pwd` before executing commands**
- **Use `cd /c/dev/stocktrader` to ensure correct project directory**
- **For long-running processes, provide status updates or progress indicators when possible**
- **Use `which python` and `which pip` to verify virtual environment activation**
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
├── main.py                           # 🚀 NEW: Modular dashboard entry point
├── core/                             # Core dashboard modules
│   ├── dashboard_controller.py       # Main UI orchestration and navigation
│   ├── data_validator.py             # Main data validator for all scripts
│   ├── session_manager.py            # Handles user sessions and state
│   ├── page_loader.py                # Dynamic page discovery and management
│   ├── health_checks.py              # Comprehensive system health monitoring
│   ├── ui_renderer.py                # UI component rendering and presentation layer
│   ├── dashboard_utils.py            # Dashboard utilities
│   ├── technical_indicators.py       # Core technical indicator calculations
│   ├── etrade_candlestick_bot.py     # Trading engine
│   └── risk_manager_v2.py            # Risk management
│
├── dashboard_pages/                  # Individual dashboard pages
├── utils/                            # Utility modules
│   ├── config/                       # Configuration utilities
│   │   └── __init__.py               # Project path and config functions
│   ├── etrade_candlestick_bot.py     # E*TRADE API trading logic
│   ├── etrade_client_factory.py      # E*TRADE client initialization
│   ├── chatgpt.py                    # GPT-4/LLM helpers
│   ├── technicals/
│   │   ├── performance_utils.py      # Pattern detection, dashboard state
│   │   ├── risk_manager.py           # Position sizing & risk controls
│   │   ├── analysis.py               # High-level technical analysis classes 
│   ├── notifier.py                   # Notification system
│   ├── data_downloader.py            # Data download utilities
│   ├── logger.py                     # Logging utilities
│   └── dashboard_utils.py            # Shared dashboard/session state logic
│
├── security/                         # Enterprise-grade security package
│   ├── __init__.py                   # Security package initialization
│   ├── authentication.py             # Session management, API validation, credentials
│   ├── authorization.py              # Role-based access control (RBAC) with permissions
│   ├── encryption.py                 # Cryptographic operations, token generation, file integrity
│   └── utils.py                      # Input sanitization, file validation, path security
│
├── patterns/                         # Pattern recognition modules
│   ├── patterns.py                   # Candlestick pattern detection
│   ├── patterns_nn.py                # PatternNN model definition
│   └── pattern_utils.py              # Pattern utilities
│
├── train/                            # Machine learning training pipeline
│   ├── deeplearning_config.py        # Deep learning configuration
│   ├── deeplearning_trainer.py       # Deep learning training scripts
│   ├── model_training_pipeline.py    # Orchestrates end-to-end ML pipeline
│   ├── model_manager.py              # Model persistence/versioning/saving
│   ├── ml_trainer.py                 # Classic ML training
│   ├── ml_config.py                  # ML configuration
│   └── feature_engineering.py        # Feature engineering (uses technical_analysis)
│
├── models/                           # Saved ML models and artifacts
├── data/                             # Data storage directory
├── logs/                             # Application logs
├── tests/                            # Unit & integration tests
├── .env.example                      # Example environment configuration
├── requirements.txt                  # Python dependencies
├── project_plan.md                   # Project planning documentation
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