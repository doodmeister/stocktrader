# E*Trade Candlestick Trading Bot & Dashboard Manual

An enterprise-grade trading platform that combines classic technical analysis with machine learning for automated E*Trade trading. Features a **completely modularized Streamlit dashboard** for real-time monitoring, robust risk management, and a comprehensive backtesting & ML pipeline.

## 🚀 NEW: Modular Architecture (COMPLETED ✅)

**The dashboard has been successfully modularized** (May 29, 2025) for better maintainability, performance, and scalability. The original 1800+ line monolithic file has been restructured into **5 focused, maintainable modules**:

- **`main.py`** - Clean entry point with Streamlit configuration
- **`core/dashboard_controller.py`** - Main orchestration and navigation logic
- **`core/page_loader.py`** - Dynamic page discovery and management  
- **`core/health_checks.py`** - Comprehensive system health monitoring
- **`core/ui_renderer.py`** - UI component rendering and presentation layer

**Latest Enhancement (May 29, 2025)**: ✅ **COMPLETED** - Complete UI rendering separation - UI logic has been fully extracted into a dedicated renderer module (`core/ui_renderer.py`), achieving perfect separation of concerns between orchestration and presentation layers.

**Benefits**: 100% functionality preserved, improved maintainability, optimized performance with caching, enhanced developer experience, independent UI component testing, and complete modular architecture.

---

## Table of Contents

1. [Key Features](#key-features)  
2. [System Requirements](#system-requirements)  
3. [Installation](#installation)  
4. [Configuration](#configuration)  
5. [Project Structure](#project-structure)  
6. [Usage](#usage)  
7. [Machine Learning Pipeline](#machine-learning-pipeline)  
8. [Testing](#testing)  
9. [Documentation](#documentation)  
10. [Contributing](#contributing)  
11. [License](#license)  
12. [Support](#support)  
13. [Further Improvement Suggestions](#further-improvement-suggestions)  

---

## Key Features

### 🏗️ Modular Dashboard Architecture (COMPLETED ✅)

- **5 Focused Modules**: Complete separation of concerns with UI rendering layer
- **Enhanced Maintainability**: Each module has a single responsibility and ~300 lines
- **UI/Logic Separation**: Dedicated UI renderer for clean presentation layer
- **Improved Performance**: Optimized caching (30s TTL) and efficient state management
- **Better Testing**: Individual components can be tested independently
- **Health Monitoring**: Comprehensive system health checks with intelligent caching
- **Dynamic Page Discovery**: Automatic detection and categorization of dashboard pages
- **Error Handling**: Improved error isolation and recovery mechanisms

#### Architecture Benefits:
- **Complete Modularity**: Every aspect properly separated (orchestration, UI, health, pages)
- **Single Responsibility**: Each module has one clear purpose
- **Clean Dependencies**: Clear import relationships between modules
- **Independent Testing**: UI components, health checks, and page loading can be tested separately
- **Parallel Development**: Teams can work on different modules simultaneously
- **Code Reuse**: UI renderer and other modules can be reused in different contexts

### 📊 Dashboard Entry Points

```powershell
# 🚀 NEW: Modular entry point (recommended)
streamlit run main.py

# ⚠️ Legacy entry point (shows migration notice)
streamlit run streamlit_dashboard.py
```

#### PowerShell Development Commands
```powershell
# Environment setup
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Test modular architecture (including UI renderer)
python -c "import main; from core import dashboard_controller, page_loader, health_checks, ui_renderer; print('All modules including UI renderer working!')"

# Check individual modules
Get-ChildItem -Path "core\" -Filter "*.py" | ForEach-Object { python -c "import core.$($_.BaseName); print('$($_.Name) imported successfully')" }
```

### Real-Time Trading Dashboard

- Dynamic symbol watchlist  
- Interactive candlestick charts (Plotly)  
- Real-time pattern detection (rule-based & ML)  
- Integrated PatternNN model predictions  
- Risk-managed order execution  

### Advanced Technical Analysis

- Candlestick pattern recognition (Hammer, Doji, Engulfing, etc.)  
- Technical indicators (RSI, MACD, Bollinger Bands)  
- Custom indicator framework  
- ATR-based position sizing

#### Enhanced ModelManager Features

- **SOLID Principles & Modular Architecture** - Clean separation of concerns with dependency injection
- **Enterprise Security Integration** - Leverages dedicated security package for file validation and access control
- **Comprehensive Security** - Path traversal protection, file validation, checksum verification
- **Performance Monitoring** - Operation timing, metrics collection, and performance analytics
- **Thread-Safe Caching** - LRU cache with TTL support for improved performance
- **Configuration Management** - Centralized config with validation and environment support
- **Health Monitoring** - Real-time system status, disk usage, and operational health checks
- **Enhanced Error Handling** - Structured exceptions with graceful degradation
- **Resource Management** - Automatic cleanup, version management, and storage optimization
- **Backward Compatibility** - 100% compatible with existing integrations

### Enterprise Security Framework

- **Modular Security Architecture** - Dedicated security package with specialized modules
- **Authentication & Session Management** - Secure API validation and credential handling
- **Role-Based Access Control (RBAC)** - Granular permissions system with 10+ permission types
- **Cryptographic Operations** - Token generation, file integrity verification, secure hashing
- **Input Sanitization & Validation** - Protection against injection attacks and malicious input
- **Path Security** - File system access protection and path traversal prevention

### Risk Management System

- Position size calculator  
- Dynamic stop-loss and take-profit  
- Portfolio exposure controls  

### Comprehensive Backtesting

- Rule-based and ML strategies  
- Historical OHLCV data simulation  
- Performance metrics: Sharpe Ratio, Max Drawdown, Win Rate  

### Enterprise Integration

- Multi-channel notifications: Email (SMTP), SMS (Twilio), Slack  
- Streamlit caching and async data fetching  
- Containerized deployment (Docker)  

### 🛡️ World-Class Data Validation System (ENHANCED ✅)

**Location**: `core/data_validator.py` (1,300+ lines of enterprise-grade validation)

The StockTrader application features a **world-class data validation system** that provides comprehensive validation for financial data, user inputs, and system parameters. This centralized validation engine ensures data integrity, security, and performance across all application components.

#### 🚀 Core Validation Capabilities

**Symbol Validation with Real-Time API Checking**
- Advanced symbol format validation with customizable patterns
- Real-time Yahoo Finance API verification with intelligent caching
- Thread-safe symbol caching with configurable TTL (4-hour default)
- Batch validation support with rate limiting protection
- Support for complex symbols (ETFs, mutual funds, international stocks)

**Interval-Specific Date Range Validation**
- Yahoo Finance interval limitations enforcement (1m: 7 days, 5m: 60 days, etc.)
- Intelligent date range suggestions for optimal performance
- Future date prevention and weekend/holiday awareness
- Historical data availability checking

**OHLCV Data Integrity Validation**
- Statistical OHLC relationship verification (High ≥ Open/Close/Low, etc.)
- Volume data validation with outlier detection
- Price movement anomaly detection (>50% daily moves)
- Data completeness analysis with gap detection
- Null value percentage monitoring with configurable thresholds

**Financial Parameter Validation**
- Price range validation ($0.01 - $1,000,000)
- Volume limits (0 - 1 trillion shares)
- Quantity validation for position sizing
- Percentage validation with configurable bounds
- Currency amount validation with precision handling

#### 🔒 Security & Performance Features

**Security-Focused Input Sanitization**
- Dangerous character pattern detection and removal
- Path traversal attack prevention (.. detection)
- HTML injection protection with optional allow-listing
- Input length limiting with configurable maximums
- File extension validation for uploads

**Performance Optimization**
- Thread-safe caching with RLock synchronization
- LRU cache eviction with configurable size limits
- API rate limiting to prevent service abuse
- Batch processing for multiple symbol validation
- Intelligent cache warming and expiration

**Advanced Error Handling**
- Structured exception hierarchy for different validation types
- Detailed error messages with actionable suggestions
- Warning system for non-critical issues
- Metadata collection for debugging and analytics
- Graceful degradation when external APIs unavailable

#### 🎯 Integration Points

**Dashboard Integration**
- Used by all dashboard pages for input validation
- Real-time symbol verification in data dashboards
- Date range validation for historical data requests
- File upload validation in model training pages

**Core System Integration**
- Risk manager integration for position validation
- Model training pipeline data validation
- Pattern detection data integrity checks
- API response validation for external data sources

#### 📊 Validation Result Types

```python
# Symbol validation with metadata
result = validator.validate_symbol("AAPL", check_api=True)
# Returns: ValidationResult(is_valid=True, value="AAPL", errors=[], warnings=[], metadata={})

# Date range validation with interval checking
result = validator.validate_dates(start_date, end_date, interval="1m")
# Returns detailed validation with suggested alternatives

# DataFrame validation with statistical analysis
result = validator.validate_dataframe(df, required_cols=['open', 'high', 'low', 'close'])
# Returns: DataFrameValidationResult with comprehensive analysis
```

#### ⚡ Performance Statistics

The validator includes built-in performance monitoring:
- **Validation operations**: Symbol, date, and DataFrame validation counts
- **Cache performance**: Hit/miss ratios for optimization insights
- **API usage**: Call counts and rate limiting statistics
- **Error tracking**: Validation failure patterns and frequencies

#### 🔧 Configuration Options

```python
# Configurable validation parameters
ValidationConfig.SYMBOL_CACHE_TTL = 4 * 60 * 60  # 4 hours
ValidationConfig.MAX_NULL_PERCENTAGE = 0.05      # 5% max nulls
ValidationConfig.API_TIMEOUT = 5                 # 5 second timeout
ValidationConfig.MAX_INVALID_OHLC_PERCENTAGE = 0.05  # 5% max invalid OHLC
```

#### 🆕 Migration from Legacy Validators

The enhanced validator in `core/data_validator.py` supersedes the basic validator in `utils/data_validator.py`, providing:
- **10x more validation rules** (1,300 vs 130 lines)
- **Enterprise security features** not available in the basic version
- **Performance optimization** with intelligent caching and rate limiting
- **Comprehensive error handling** with structured exceptions

**Migration Steps:**

1. **Replace Import Statements**
```python
# OLD (legacy validator)
from utils.data_validator import validate_symbol, validate_date_range

# NEW (enhanced validator)
from core.data_validator import UniversalDataValidator

# Initialize the enhanced validator
validator = UniversalDataValidator()
```

2. **Update Validation Calls**
```python
# OLD: Basic symbol validation
if validate_symbol(symbol):
    # process symbol

# NEW: Enhanced symbol validation with API checking
result = validator.validate_symbol(symbol, check_api=True)
if result.is_valid:
    # process symbol with confidence
    print(f"Validated symbol: {result.value}")
else:
    print(f"Validation errors: {result.errors}")
```

3. **Enhanced Date Range Validation**
```python
# OLD: Basic date validation
start_date, end_date = validate_date_range(start_str, end_str)

# NEW: Interval-aware validation with optimization suggestions
result = validator.validate_dates(start_str, end_str, interval="1m")
if result.is_valid:
    start_date, end_date = result.value
    if result.warnings:
        print(f"Optimization suggestions: {result.warnings}")
```

4. **Update Components Using Local Validators**

Components with local validation logic should migrate to the centralized system:

**Data Dashboard v2** (`dashboard_pages/data_dashboard_v2.py`):
```python
# Replace local validate_symbol function with:
validator = UniversalDataValidator()
result = validator.validate_symbol(symbol, check_api=True)
```

**Realtime Dashboard v3** (`dashboard_pages/realtime_dashboard_v3.py`):
```python
# Replace individual validation functions with unified validator
validator = UniversalDataValidator()
```

#### 🔧 Practical Usage Examples

**1. Symbol Validation in Dashboard Pages**
```python
from core.data_validator import UniversalDataValidator

class DataDashboard:
    def __init__(self):
        self.validator = UniversalDataValidator()
    
    def process_symbol_input(self, symbol):
        # Validate with real-time API checking
        result = self.validator.validate_symbol(symbol, check_api=True)
        
        if not result.is_valid:
            st.error(f"Invalid symbol: {', '.join(result.errors)}")
            return None
            
        if result.warnings:
            st.warning(f"Symbol warnings: {', '.join(result.warnings)}")
            
        # Use validated symbol with confidence
        return result.value
```

**2. DataFrame Validation for Model Training**
```python
def prepare_training_data(self, df):
    # Comprehensive DataFrame validation
    result = self.validator.validate_dataframe(
        df, 
        required_cols=['open', 'high', 'low', 'close', 'volume'],
        check_nulls=True,
        check_ohlc_relationships=True
    )
    
    if not result.is_valid:
        logging.error(f"Data validation failed: {result.errors}")
        return None
        
    # Log data quality metrics
    logging.info(f"Data quality score: {result.metadata.get('quality_score', 'N/A')}")
    logging.info(f"Null percentage: {result.metadata.get('null_percentage', 0):.2%}")
    
    return result.cleaned_data  # Returns cleaned/processed DataFrame
```

**3. Risk Management Integration**
```python
def validate_trade_parameters(self, trade_params):
    # Validate financial parameters
    price_result = self.validator.validate_price(trade_params['price'])
    quantity_result = self.validator.validate_quantity(trade_params['quantity'])
    
    # Collect all validation results
    all_valid = all([price_result.is_valid, quantity_result.is_valid])
    
    if not all_valid:
        errors = []
        errors.extend(price_result.errors)
        errors.extend(quantity_result.errors)
        raise ValidationError(f"Trade validation failed: {errors}")
    
    return {
        'validated_price': price_result.value,
        'validated_quantity': quantity_result.value
    }
```

**4. Batch Symbol Validation for Watchlists**
```python
def validate_watchlist(self, symbols):
    # Efficient batch validation with caching
    validated_symbols = []
    invalid_symbols = []
    
    for symbol in symbols:
        result = self.validator.validate_symbol(symbol, check_api=True)
        if result.is_valid:
            validated_symbols.append(result.value)
        else:
            invalid_symbols.append((symbol, result.errors))
    
    # Report validation statistics
    success_rate = len(validated_symbols) / len(symbols)
    logging.info(f"Watchlist validation: {success_rate:.1%} success rate")
    
    return validated_symbols, invalid_symbols
```

**5. Security-Focused File Upload Validation**
```python
def process_uploaded_file(self, uploaded_file):
    # Validate file security and format
    filename_result = self.validator.sanitize_input(
        uploaded_file.name,
        remove_dangerous=True,
        max_length=255
    )
    
    if not filename_result.is_valid:
        st.error("Invalid filename")
        return None
    
    # Additional file validation can be added here
    # (size limits, content validation, etc.)
    
    return filename_result.value
```

#### 📈 Performance Monitoring

**Cache Performance Analysis**
```python
# Get validation performance statistics
stats = validator.get_performance_stats()

print(f"Symbol cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"API calls made: {stats['api_calls']}")
print(f"Validations performed: {stats['total_validations']}")
print(f"Average validation time: {stats['avg_validation_time']:.3f}s")
```

**Custom Configuration for High-Performance Scenarios**
```python
# Configure for high-frequency validation
from core.data_validator import ValidationConfig

# Extend cache TTL for stable environments
ValidationConfig.SYMBOL_CACHE_TTL = 8 * 60 * 60  # 8 hours

# Increase cache size for large symbol sets
ValidationConfig.CACHE_SIZE = 10000

# Adjust API timeout for slow networks
ValidationConfig.API_TIMEOUT = 10

# Create validator with custom config
validator = UniversalDataValidator()
```

This enhanced validation system ensures **enterprise-grade data integrity** across the entire StockTrader application while providing the flexibility and performance needed for real-time trading operations.
---

## System Requirements

- **Python:** 3.8+  
- **Git**  
- **E*Trade Developer Account** (sandbox and/or production API keys)  
- **Optional:** SMTP server, Twilio account, Slack webhook  
- **Docker** (for containerized deployment)  

---

## Installation

Python 3.10 is the rcommended version for this project. Ensure you have Python and pip installed.

1. **Clone the Repository**  

   ```bash
   git clone https://github.com/doodmeister/stocktrader.git
   cd stocktrader
   ```

2. **Set Up Python Environment**  
   - **Windows:**  

     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```

   - **Linux/macOS:**  

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**  

   ```bash
   pip install -r requirements.txt
   ```

4. **Install TA-Lib (Required for Technical Analysis)**

   ### Option A: Use Precompiled Wheel (Recommended for Windows)

   Download a precompiled `.whl` file for your Python version and Windows architecture. This avoids C build dependencies.

   1. Visit: [Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
   2. Find the matching `.whl` for your Python version and system architecture  
      Example: `TA_Lib‑0.4.0‑cp310‑cp310‑win_amd64.whl` for Python 3.10, 64-bit
   3. Download the appropriate file
   4. Navigate to your download directory and install:

      ```bash
      pip install TA_Lib‑0.4.0‑cp310‑cp310‑win_amd64.whl
      ```

      *(Replace with your downloaded filename)*

   #### Option B: Alternative Installation Methods

   ```bash
   # Via conda (if using Anaconda/Miniconda)
   conda install -c conda-forge ta-lib
   
   # Via pip (may require C++ build tools)
   pip install TA-Lib
   ```

---

## Configuration

1. **Create and Edit `.env`**  

   ```bash
   cp .env.example .env
   ```

   Open `.env` and fill in your E*Trade API keys, account IDs, and any optional notification credentials.

2. **Validate Your Configuration**  

   ```bash
   python -m config.validation
   ```

---

## Project Structure

### 🏗️ Modular Dashboard Architecture (COMPLETED ✅)

```plaintext
stocktrader/
├── main.py                           # 🚀 NEW: Modular dashboard entry point
├── streamlit_dashboard.py            # ⚠️ LEGACY: Redirects to main.py (deprecated)
│
├── core/                             # 🆕 Core dashboard modules (COMPLETED ✅)
│   ├── dashboard_controller.py       # Main UI orchestration and navigation
│   ├── data_validator.py             # Main data validator for all scripts
│   ├── session_manager.py            # Handles user sessions and state
│   ├── page_loader.py                # Dynamic page discovery and management
│   ├── health_checks.py              # Comprehensive system health monitoring
│   ├── ui_renderer.py                # ✅ NEW: UI component rendering and presentation layer
│   ├── dashboard_utils.py            # Dashboard utilities
│   ├── etrade_candlestick_bot.py     # Trading engine
│   └── risk_manager_v2.py            # Risk management
│
├── dashboard_pages/                  # 📊 Individual dashboard pages
│   ├── advanced_ai_trade.py          # Real-time AI based trading
│   ├── data_dashboard.py             # Data download dashboard
│   ├── data_dashboard_v2.py          # Enhanced data dashboard
│   ├── data_analysis_v2.py           # Data analysis tools
│   ├── model_training.py             # ML pipeline UI
│   ├── model_visualizer.py           # Model visualization
│   ├── nn_backtest.py                # Neural net backtesting
│   ├── classic_strategy_backtest.py  # Classic strategy backtesting
│   ├── patterns_management.py        # Pattern management UI
│   ├── realtime_dashboard.py         # Real-time trading dashboard
│   ├── realtime_dashboard_v2.py      # Enhanced real-time dashboard
│   ├── realtime_dashboard_v3.py      # Latest real-time dashboard
│   └── simple_trade.py               # Simple trading interface
│
├── utils/                            # Utility modules
│   ├── config/                       # 🆕 Configuration utilities
│   │   └── __init__.py               # Project path and config functions
│   ├── etrade_candlestick_bot.py     # E*TRADE API trading logic
│   ├── etrade_client_factory.py      # E*TRADE client initialization
│   ├── indicators.py                 # Technical indicators
│   ├── chatgpt.py                    # GPT-4/LLM helpers
│   ├── technicals/
│   │   ├── performance_utils.py      # Pattern detection, dashboard state
│   │   ├── risk_manager.py           # Position sizing & risk controls
│   │   ├── indicators.py             # Stateless technical indicator functions
│   │   └── technical_analysis.py     # TechnicalAnalysis class: scoring, price targets
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

---

## Usage

### 🚀 Launch Dashboard (Modular Architecture)

```powershell
# Recommended: Use the new modular entry point
streamlit run main.py

# Alternative: Legacy entry point (shows migration notice)
streamlit run streamlit_dashboard.py
```

The modular dashboard provides the same functionality as before but with improved:
- **Performance**: Optimized caching and faster load times
- **Maintainability**: Clean, focused modules that are easier to debug
- **Health Monitoring**: Real-time system status with intelligent alerts
- **Error Handling**: Better isolation and recovery from issues

---

## Machine Learning Pipeline

### End-to-End Flow

1. **Download Data**  
   - Fetches OHLCV from E*Trade/Yahoo  
   - Saves CSVs and displays price plot in dashboard  

2. **Train Model**  
   - Automated feature engineering & scaling  
   - Time-series cross-validation  
   - Saves pipeline via `ModelManager`  
   - Displays accuracy, confusion matrix, classification report  

3. **Run Model on Data**  
   - Loads each saved pipeline  
   - Runs `predict()` on new OHLCV  
   - Displays signal chart in dashboard  

4. **Combine with Patterns**  
   - Uses `CandlestickPatterns` to filter ML signals  
   - Overlays final buy/sell signals on candlestick chart  

*This provides a full workflow: raw OHLCV → feature engineering → model training → inference → pattern filtering → final trade signals—all within the Streamlit UI.*

- Pattern Neural Network (PatternNN) for pattern classification  
- Automated data preparation and feature engineering  
- **Production-Grade Model Management** (Enhanced ModelManager)  
- Configurable training parameters
- Real-time inference integration

### Enhanced Model Training Framework

#### ModelTrainer Features

- **Robust Feature Engineering**  
- **Configurable Hyperparameters**  
- **Automatic Time-Series Cross Validation**  
- **Parallelized Training & Metrics Computation**  
- **Persistent Model Saving & Versioning**  

## Testing

- **Unit Tests:**  
pytest tests/

## Documentation

- **Docstrings:** All modules & functions are documented with Google-style docstrings.  

## Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/my-feature`)  
3. Commit your changes (`git commit -m "Add my feature"`)  
4. Push to branch (`git push origin feature/my-feature`)  
5. Open a Pull Request  

Please follow the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and style guidelines.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Support

For issues or questions, please open a GitHub Issue or contact the maintainers at <support@example.com>.

---

## Further Improvement Suggestions

Split the manual into separate docs for user, developer, and API reference.
Add more code samples and usage scenarios for each dashboard page.
Include architecture and data flow diagrams for clarity.
Link to E*Trade API and Streamlit documentation for onboarding.
Add a FAQ and troubleshooting section for common deployment issues.

## Substantive Changes Highlighted

Fixed all import paths to reflect actual module locations (e.g., from patterns.patterns import CandlestickPatterns).
Clarified Docker Compose usage and ensured .env is referenced.
Unified session state initialization instructions for all dashboards.
Updated project structure to match the actual codebase, including all major subfolders and files.
Added troubleshooting section for common errors.
Corrected and expanded usage instructions for all major workflows.
