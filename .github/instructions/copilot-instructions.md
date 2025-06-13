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
- **Use `cd /c/dev/stocktrader` to ensure correct project directory**
- **Handle file path separators correctly for Windows (use forward slashes in GitBash)**
- **Escape spaces in file paths or use quotes when necessary**

### Virtual Environment Setup
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```
### Running the Dashboard
streamlit run main.py

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
# Streamlit Dashboard Session State & Widget Key Standards

**ALWAYS USE `SessionManager` FOR ALL WIDGET AND SESSION STATE MANAGEMENT.**

---

## Key Guidelines

- **All widget keys must be unique and namespaced by page and tab.**
  - Use `SessionManager(page_name, tab=...)` for widget creation.
  - Never reuse SessionManager or keys across tabs or pages.

- **Key Format:**  
  `"{namespace}_{tab}_{widget_type}_{name}"`
  - _Example:_ `data_analysis_v2_tab1_selectbox_symbol`

- **For tabs:**  
  - Always instantiate `SessionManager` **inside** each tab block with a unique `tab` argument.

- **Initialize state on page load:**  
  - Use an `_init_*_state()` function per page.
  - Use a page-specific flag to clear state only on first load.
---
## Anti-patterns to avoid

- ❌ Using generic, repeated, or global widget keys.
- ❌ Setting keys directly in `st.session_state`.
- ❌ Clearing all of `st.session_state` except for explicit user resets.
---
## Defensive Cleanup

- Clear only conflicting keys, not all state:
  ```python
  if "old_conflicting_key" in st.session_state:
      del st.session_state["old_conflicting_key"]
  ```
- Use SessionManager’s debug tools for troubleshooting key/state issues.

Reference:
See core/session_manager.py for implementation details.