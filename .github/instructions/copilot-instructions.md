---
applyTo: '**'
---
## Environment & Platform Standards
The terminal is bash running on windows 11
- **Python Version**: 3.10.11 or higher
- **Package Manager**: `pip` (Python package installer)
- **linter**: `ruff` (Python linter)


### Operating System & Shell Requirements
- **Target OS**: Windows
- **Default Shell**: Bash (`bash.exe`)
- **All terminal commands MUST be bash-compatible**
- **Avoid emojis in bash commands**

### Terminal Command Standards
- **Avoid emojis in bash commands**
- **Always activate the virtual environment by running source venv/Scripts/activate before executing commands**
- **Use absolute paths** for clarity

#### Project Commands (Bash)
```bash
# Environment setup
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

# Testing
python -m pytest tests/
find core/ -name "*.py" -exec basename {} .py \; | while read module; do python -c "import core.$module; print('$module.py imported successfully')"; done
```

### Project structure
```plaintext
stocktrader/
├── main.py                           # 🚀 NEW: Modular dashboard entry point
├── streamlit_dashboard.py            # ⚠️ LEGACY: Redirects to main.py (deprecated)
├── core/                             # 🆕 Core dashboard modules (COMPLETED ✅)
│   ├── dashboard_controller.py       # Main UI orchestration and navigation
│   ├── data_validator.py             # Main data validator for all scripts
│   ├── session_manager.py            # Handles user sessions and state
│   ├── page_loader.py                # Dynamic page discovery and management
│   ├── health_checks.py              # Comprehensive system health monitoring
│   ├── ui_renderer.py                # ✅ NEW: UI component rendering and presentation layer
│   ├── dashboard_utils.py            # Dashboard utilities
│   ├── technical_indicators.py       # 📈 NEW: Core technical indicator calculations (283 lines)
│   ├── etrade_candlestick_bot.py     # Trading engine
│   └── risk_manager_v2.py            # Risk management
│
├── dashboard_pages/                  # 📊 Individual dashboard pages
│   ├── advanced_ai_trade.py          # 📈 NEW: Advanced AI trading with centralized technical analysis
│   ├── data_dashboard.py             # Data download dashboard
│   ├── data_analysis_v2.py           # Data analysis tools
│   ├── model_training.py             # ML pipeline UI
│   ├── model_visualizer.py           # Model visualization
│   ├── nn_backtest.py                # Neural net backtesting
│   ├── classic_strategy_backtest.py  # Classic strategy backtesting
│   ├── patterns_management.py        # Pattern management UI
│   ├── realtime_dashboard.py         # Latest real-time dashboard
│   └── simple_trade.py               # Simple trading interface
│
├── utils/                            # Utility modules
│   ├── config/                       # 🆕 Configuration utilities
│   │   └── __init__.py               # Project path and config functions
│   ├── etrade_candlestick_bot.py     # E*TRADE API trading logic
│   ├── etrade_client_factory.py      # E*TRADE client initialization
│   ├── chatgpt.py                    # GPT-4/LLM helpers
│   ├── technicals/
│   │   ├── performance_utils.py      # Pattern detection, dashboard state
│   │   ├── risk_manager.py           # Position sizing & risk controls
│   │   ├── analysis.py               # 📈 NEW: High-level technical analysis classes (402 lines)
│   │   ├── indicators.py             # 📈 LEGACY: Backward compatibility (replaced by core module)
│   │   └── technical_analysis.py     # 📈 LEGACY: Replaced by centralized analysis.py
│   ├── notifier.py                   # Notification system
│   ├── data_downloader.py            # Data download utilities
│   ├── logger.py                     # Logging utilities
│   └── dashboard_utils.py            # Shared dashboard/session state logic
│
├── security/                         # Enterprise-grade security package
│   ├── __init__.py                   # Security package initialization
│   ├── authentication.py            # Session management, API validation, credentials
│   ├── authorization.py             # Role-based access control (RBAC) with permissions
│   ├── encryption.py                # Cryptographic operations, token generation, file integrity
│   └── utils.py                      # Input sanitization, file validation, path security
│
├── patterns/                         # Pattern recognition modules
│   ├── __init__.py
│   ├── patterns.py                   # Candlestick pattern detection
│   ├── patterns_nn.py                # PatternNN model definition
│   └── pattern_utils.py              # Pattern utilities
│
├── train/                            # Machine learning training pipeline
│   ├── __init__.py
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
├── source/                           # Source data and configurations
├── .env.example                      # Example environment configuration
├── requirements.txt                  # Python dependencies
├── project_plan.md                   # Project planning documentation
```

### File and Module Guidelines
- **Module Organization**: Use the established `core/`, `dashboard_pages/`, `utils/` structure
- **Data Validation for this project comes from the `core/data_validator` module**
- **Technical Indicators for this project comes from the `core/technical_indicators` module**
- **Testing**: All test scripts should be saved in the `tests/` directory
- **Import System**: Use absolute imports for clarity and maintainability
- **Logging**: Use the `logging` module for all logging needs
- **Configuration**: Use environment variables for configuration (e.g., `STREAMLIT_SERVER_PORT`)

### Code Quality Standards
- **Function Length**: Keep modules focused (~300 lines max)
- **Single Responsibility**: Each module should have one clear purpose
- **Type Hints**: Use Python type annotations where applicable
- **Docstrings**: Google-style docstrings for all public functions
- **Error Messages**: Provide clear, actionable error messages

## Security & Best Practices
- **Security Functionality**: Use `core/security.py` for security-related functions
- **Sensitive Data**: Never hard-code sensitive data; use environment variables


### Path Handling
```bash
# Correct: Windows paths in bash format
projectPath="/c/dev/stocktrader"
corePath="$projectPath/core"
```