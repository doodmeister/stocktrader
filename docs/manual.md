# E*Trade Candlestick Trading Bot & Dashboard

A robust, automated trading platform for E*Trade, featuring technical analysis, machine learning, risk management, and a Streamlit dashboard.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Architecture](#architecture)
6. [Features](#features)
7. [Usage](#usage)
8. [Development](#development)
9. [Troubleshooting](#troubleshooting)
10. [Support](#support)

---

## 1. Quick Start

```bash
git clone https://github.com/<your-org>/etrade-bot.git
cd etrade-bot
docker-compose up -d
```

- Visit [http://localhost:8501](http://localhost:8501) for the dashboard.

---

## 2. Prerequisites

- **Python 3.8+** (`python --version`)
- **Git 2.0+** (`git --version`)
- **64-bit OS** (Windows, Linux, or macOS)
- **8GB RAM** minimum (16GB recommended)
- **Docker & Docker Compose** (optional, for containerized deployment)
- **E*Trade Developer Account** (sandbox and/or production API keys)
- **Optional:** SMTP server (email alerts), Twilio account (SMS), Slack workspace (webhook), NVIDIA GPU (for ML acceleration)

---

## 3. Installation

### 3.1 Clone the Repository

```bash
git clone https://github.com/<your-org>/etrade-bot.git
cd etrade-bot
```

### 3.2 Set Up Python Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
where python
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
which python
```

### 3.3 Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import streamlit; import pandas; import torch"
```

### 3.4 Configure Environment

```bash
cp .env.example .env
python utils/validate_config.py
```

---

## 4. Configuration

Edit the `.env` file with your credentials and settings:

```plaintext
# E*TRADE API (Required)
ETRADE_CONSUMER_KEY=your_consumer_key
ETRADE_CONSUMER_SECRET=your_consumer_secret
ETRADE_OAUTH_TOKEN=your_access_token
ETRADE_OAUTH_TOKEN_SECRET=your_access_token_secret
ETRADE_ACCOUNT_ID=your_account_id
ETRADE_USE_SANDBOX=true  # Set to false for production

# Trading Parameters
MAX_POSITIONS=5
MAX_LOSS_PERCENT=0.02
PROFIT_TARGET_PERCENT=0.03
MAX_DAILY_LOSS=0.05
SYMBOLS=AAPL,MSFT,GOOG,AMZN,TSLA

# Optional: Email Notifications
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@example.com
SMTP_PASS=your_app_password  # Use app-specific password for Gmail

# Optional: SMS Alerts
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1234567890
TWILIO_TO_NUMBER=+1987654321  # Your number to receive alerts

# Optional: Slack Integration
SLACK_WEBHOOK_URL=your_webhook_url
SLACK_CHANNEL=#trading-alerts  # Channel to post alerts to
```

---

## 5. Architecture

```
stocktrader/
├── streamlit_dashboard.py            # Main Streamlit dashboard entry point
├── utils/
│   ├── etrade_candlestick_bot.py     # E*TRADE API trading logic
│   ├── etrade_client_factory.py      # E*TRADE client initialization
│   ├── indicators.py                 # Technical indicators
│   ├── chatgpt.py                    # GPT-4/LLM helpers
│   ├── model_manager.py              # Model persistence/versioning
│   ├── technicals/
│   │   ├── performance_utils.py      # Pattern detection, dashboard state
│   │   ├── risk_manager.py           # Position sizing & risk controls
│   │   └── indicators.py             # Stateless technical indicator functions
│       ├── technical_analysis.py     # TechnicalAnalysis class: scoring, price targets
│       └── risk_manager.py           # Position sizing & risk controls
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
│   ├── deeplearning_config.py        # Classic ML training
│   └── deeplearning_trainer.py       # Deep learning training scripts
│   ├── model_training_pipeline.py    # Orchestrates end-to-end ML pipeline
│   ├── model_manager.py              # Model persistence/versioning/saving
│   ├── ml_trainer.py                 # Classic ML training
│   ├── ml_config.py                  # ML config
│   └── feature_engineering.py        # Feature engineering (uses technical_analysis)
├── pages/                            # Streamlit multi-page app
│   ├── advanced_ai_trade.py          # Real-time AI based trading
│   ├── data_dashboard.py             # Streamlit dashboard for data downloading
│   ├── data_analysis_v2.py           # Data analysis tools
│   ├── model_training.py             # ML pipeline UI
│   ├── model_visualizer.py           # Model visualizer
│   ├── nn_backtest.py                # Neural net backtesting
│   ├── classic_strategy_backtest.py  # Classic strategy backtesting
│   ├── patterns.py                   # Pattern editor UI
├── models/                           # Saved models
├── tests/                            # Unit & integration tests
├── Dockerfile                        # Docker build file
├── docker-compose.yml                # Docker Compose for deployment
├── requirements.txt                  # Python dependencies
├── requirements-dev.txt              # Dev dependencies
└── env.example                       # Example environment config
```

---

## 6. Features

### Pattern Detection

- **Classic Patterns:** Hammer, Doji, Engulfing, etc.
- **Real-time scanning** with confidence scoring (0–100%)
- **Custom pattern definitions**

### ML-Based Recognition

- **Neural networks (LSTM and CNN architectures) for pattern classification
- **Probability scoring with model confidence metrics

### Technical Analysis

Core Indicators: RSI, MACD, Bollinger Bands, Volume, ATR, EMAs
Custom Indicators: Compose strategies with multiple indicators
Backtesting engine with performance metrics (Sharpe, Sortino, drawdown)

### Risk Management

Position Sizing: Kelly criterion, percent-based, fixed-size
Stop Loss Options: Fixed, trailing, ATR-based, time-based
Take Profit: Multiple targets with partial position closing
Daily Loss Limits: Auto-shutdown to prevent excessive losses

### ML Pipeline

- **Data Processing:** OHLCV normalization, feature engineering, validation splitting
- **Model Management:** Automated training, versioning, metrics, persistence

### Dashboard

Live Portfolio Monitoring: Real-time P&L, open positions
Pattern Scanner: Visual display of detected patterns
Strategy Builder: Visual interface for strategy creation
Backtest Visualization: Equity curves, trade history, metrics
---

## 7. Usage

### 7.1 Launch Dashboard

```bash
streamlit run streamlit_dashboard.py
```

---

## 8. Development

### 8.1 Run Tests

```bash
pytest tests/
pytest --cov=. --cov-report=html
```

### 8.2 Docker Deployment

```bash
docker build -t etrade-bot .
docker run -p 8501:8501 -v $(pwd)/data:/app/data etrade-bot
```

---

## 9. Troubleshooting

- **Check `.env` configuration** for typos or missing values.
- **Review Docker logs** for container issues: `docker logs <container_id>`
- **Validate API credentials** (sandbox/production).
- **Enable debug logging**: add `--debug` to CLI commands.
- **Check logs** in the `logs/` directory.
- **Validate dependencies**: `python utils/validate_config.py`

---

## 10. Support

- **Issue Tracking:** Use GitHub Issues for bugs and feature requests.
- **Community:** Join technical discussions and get support.
- **Documentation:** See `/docs` for API reference, guides, and more.

---

Licensed under MIT © 2025

---

## Change Log

### Version 1.0.0 (2025-05-01)

Change Log: Substantive Improvements
Repository and Project Name Consistency:

Updated repository name from "etrade-bot" to "stocktrader" throughout the document to match the actual project folder structure.
Installation Process:

Added a verification step to confirm dependencies are installed correctly.
Added --check-all flag to configuration validation script for more thorough validation.
Configuration:

Added missing environment variables (TWILIO_TO_NUMBER, SLACK_CHANNEL).
Added helpful comments for security (app-specific password for Gmail).
Improved formatting and organization of configuration variables.
Architecture Diagram:

Completely restructured to reflect the actual project organization.
Added missing files and refined descriptions to match functionality.
Added logs directory which was previously missing.
Command Line Parameters:

Added critical missing parameters to CLI commands (e.g., --capital, --output, --threshold).
Added explanations of available strategy options for backtesting.
Fixed notifier test command to use proper module path.
Development Environment:

Added instructions for dev dependencies and pre-commit hooks.
Expanded Docker deployment with proper volume mounting and environment options.
Added Docker Compose command examples.
Troubleshooting:

Reorganized into logical categories with specific verification commands.
Added network and system diagnostic commands.
Included log viewing instructions.

---

## Suggestions for Further Improvement

Add Screenshots of Dashboard: Include annotated screenshots of the Streamlit interface showing key features like pattern detection, backtesting results, and portfolio management to help users understand the visual interface.

Create a Quick Reference Guide: Add a one-page cheat sheet with the most common commands and configurations for quick reference.

Include Sample Strategy Files: Provide example strategy configuration files that users can modify as starting points for custom trading strategies.

Add Integration Diagrams: Create visual flowcharts showing how the different components (API, ML models, notification systems) interact during live trading sessions.

Develop an Environment Setup Script: Create a shell script that automates the entire installation and configuration process to minimize setup issues for new users.
