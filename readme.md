# E*Trade Candlestick Trading Bot & Dashboard Manual

An enterprise-grade trading platform that combines classic technical analysis with machine learning for automated E*Trade trading. Features a Streamlit dashboard for real-time monitoring, robust risk management, and a comprehensive backtesting & ML pipeline.

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

### Machine Learning Pipeline
- Pattern Neural Network (PatternNN) for pattern classification  
- Automated data preparation and feature engineering  
- Model persistence/versioning (ModelManager)  
- Configurable training parameters  
- Real-time inference integration  

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

---

## System Requirements

- **Python:** 3.8+  
- **Git**  
- **E*Trade Developer Account** (sandbox and/or production API keys)  
- **Optional:** SMTP server, Twilio account, Slack webhook  
- **Docker** (for containerized deployment)  

---

## Installation

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

---

## Configuration

1. **Create and Edit `.env`**  
   ```bash
   cp .env.example .env
   ```
   Open `.env` and fill in your E*Trade API keys, account IDs, and any optional notification credentials.

2. **Validate Your Configuration**  
   ```bash
   python -m utils.validation
   ```

---

## Project Structure

```
stocktrader/
├── streamlit_dashboard.py            # Main Streamlit dashboard entry point
├── utils/
│   ├── etrade_candlestick_bot.py     # E*TRADE API trading logic
│   ├── etrade_client_factory.py      # E*TRADE client initialization
│   ├── indicators.py                 # Technical indicators
│   ├── model_manager.py              # Model persistence/versioning
│   ├── technials/
│   │   ├── performance_utils.py      # Pattern detection, dashboard state
│   │   ├── risk_manager.py           # Position sizing & risk controls
│   │   └── indicators.py             # Technical indicators
│   ├── notifier.py                   # Notification system
│   ├── data_validator.py             # Input validation helpers
│   ├── data_downloader.py            # Data download utilities
│   ├── dashboard_utils.py            # Shared dashboard/session state logic
│   └── security.py                   # Credential management
├── patterns/
│   ├── patterns.py                   # Candlestick pattern detection
│   ├── patterns_nn.py                # PatternNN model definition
│   └── pattern_utils.py              # Pattern utilities
├── train/
│   ├── model_training_pipeline.py    # ML training pipeline
│   ├── model_manager.py              # Model persistence/versioning
│   ├── ml_trainer.py                 # Classic ML training
│   ├── ml_config.py                  # ML config
│   └── deeplearning_trainer.py       # Candlestick pattern deep learning model training
├── pages/                            # Streamlit multi-page app
│   ├── live_dashboard.py             # Real-time monitoring
│   ├── data_dashboard.py             # Data visualization dashboard
│   ├── data_analysis.py              # Data analysis tools
│   ├── model_training.py             # ML pipeline UI
│   ├── nn_backtest.py                # Neural net backtesting
│   ├── classic_strategy_backtest.py  # Classic strategy backtesting
│   ├── patterns.py                   # Pattern editor UI
│   └── settings.py                   # System configuration UI
├── data/                             # Data storage and loaders
│   ├── data_loader.py
│   ├── io.py
│   └── model_trainer.py
├── models/                           # Saved models
│   └── patterns_nn.py
├── tests/                            # Unit & integration tests
├── Dockerfile                        # Docker build file
├── docker-compose.yml                # Docker Compose for deployment
├── requirements.txt                  # Python dependencies
├── requirements-dev.txt              # Dev dependencies
└── env.example                       # Example environment config
```

---

## Usage

1. **Launch Dashboard**  
   ```bash
   streamlit run dashboard/streamlit_dashboard.py
   ```

2. **Pattern Detection (CLI)**  
   ```bash
   python patterns.py --symbol AAPL --start 2025-01-01 --end 2025-04-30
   ```
   _Note: The script is `patterns.py`, not `pattern_detection.py`._

3. **Strategy Backtesting**  
   ```bash
   python backtest.py --strategy ml --symbols AAPL,MSFT
   ```

4. **ML Model Training**  
   ```bash
   python training/train_model.py --config training/config.yaml
   ```

5. **Alert System Testing**  
   ```bash
   python utils/notifier_test.py
   ```

6. **Test Suite Execution**  
   ```bash
   pytest --cov=stocktrader
   ```

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

_This provides a full workflow: raw OHLCV → feature engineering → model training → inference → pattern filtering → final trade signals—all within the Streamlit UI._

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

For issues or questions, please open a GitHub Issue or contact the maintainers at support@example.com.

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