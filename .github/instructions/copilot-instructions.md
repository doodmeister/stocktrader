---
applyTo: '**'
---
## Environment & Platform Standards
The terminal is bash running on windows 11
- **Python Version**: 3.10.11
- **Package Manager**: `pip` 
- **linter**: `ruff`


### Operating System & Shell Requirements
- **Target OS**: Windows
- **Default Shell**: GitBash (`bash.exe`)
- **All terminal commands MUST be bash-compatible***

### Terminal Command Standards
- **Avoid emojis in bash commands**
- **Always activate the virtual environment by running source venv/Scripts/activate before executing commands**
- **Use absolute paths**

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
- **Data Validation for this project comes from the `core/data_validator` module**
- **Technical Indicators for this project comes from the `core/technical_indicators` module**
- **Testing**: All test scripts should be saved in the `tests/` directory

### Code Quality Standards
- **Function Length**: Keep modules focused (~300 lines max)
- **Single Responsibility**: Each module should have one clear purpose
- **Type Hints**: Use Python type annotations where applicable
- **Docstrings**: Google-style docstrings for all public functions
- **Linting**: Use `ruff` for linting; run `ruff check .` before committing

## Security & Best Practices
- **Security Functionality**: Use `core/security.py` for security-related functions
- **Sensitive Data**: Never hard-code sensitive data; use environment variables

### Path Handling
```bash
projectPath="/c/dev/stocktrader"
corePath="$projectPath/core"
```