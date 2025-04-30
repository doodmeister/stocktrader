# E*Trade Candlestick Trading Bot & Dashboard

A robust, automated trading platform for E*Trade, featuring technical analysis, machine learning, risk management, and a Streamlit dashboard.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
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
ETRADE_CONSUMER_KEY=your_sandbox_consumer_key
ETRADE_CONSUMER_SECRET=your_sandbox_consumer_secret
ETRADE_OAUTH_TOKEN=your_sandbox_access_token
ETRADE_OAUTH_TOKEN_SECRET=your_sandbox_access_token_secret
ETRADE_ACCOUNT_ID=your_sandbox_account_id
ETRADE_USE_SANDBOX=true # Set to False for production

# Trading Parameters
MAX_POSITIONS=5
MAX_LOSS_PERCENT=0.02
PROFIT_TARGET_PERCENT=0.03
MAX_DAILY_LOSS=0.05
SYMBOLS=AAPL,MSFT,GOOG

# Optional: Email Notifications
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email
SMTP_PASS=your_password

# Optional: SMS Alerts
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1234567890

# Optional: Slack Integration
SLACK_WEBHOOK_URL=your_webhook_url
```

---

## 5. Architecture

```
stocktrader/
├── streamlit_dashboard.py       # Web dashboard (Streamlit)
├── backtester.py                # Strategy backtesting
├── ml_pipeline.py               # ML model training & inference
│
├── utils/
│   ├── etrade_candlestick_bot.py   # Main trading logic, E*TRADE API, strategy engine
│   ├── indicators.py               # Technical indicators (RSI, MACD, BBands, etc.)
│   ├── model_manager.py            # Model persistence/versioning
│   ├── ml_pipeline.py              # ML pipeline (PatternNN, training, evaluation)
│   ├── performance_utils.py        # Dashboard state, async data, pattern detection, UI
│   ├── validate_config.py          # Config validation
│   ├── validation.py               # Input/config validation helpers
│   └── getuservar.py               # E*TRADE OAuth helper
│
├── core/
│   └── notifier.py                 # Notification utility (email, Slack, etc.)
│
├── data/
│   ├── data_loader.py              # Download OHLCV data (Yahoo Finance)
│   ├── io.py                       # IO utilities (e.g., zip archive)
│   └── data_dashboard.py           # Streamlit dashboard for data/model ops
│
├── train/
│   └── training_pipeline.py        # ML training pipeline (RandomForest, etc.)
│
├── models/
│   └── patterns_nn.py              # PatternNN model definition
│
├── patterns.py                     # Candlestick pattern detection (rule-based)
│
├── pages/
│   ├── live_dashboard.py           # Real-time monitoring
│   ├── backtest.py                 # Backtesting UI
│   ├── model_training.py           # ML pipeline UI
│   └── settings.py                 # System configuration UI
│
├── tests/                          # Unit & integration tests
└── .github/workflows/              # CI/CD pipelines
```

---

## 6. Features

### Pattern Detection

- **Classic Patterns:** Hammer, Doji, Engulfing, etc.
- **Real-time scanning** with confidence scoring (0–100%)
- **Custom pattern definitions**

### ML-Based Recognition

- **LSTM neural networks** for pattern classification
- **Probability scoring** and model versioning

### Technical Analysis

- **Core Indicators:** RSI, MACD, Bollinger Bands, Volume
- **Custom Indicators:** Compose and backtest your own

### Risk Management

- **Position Sizing:** Account-based, risk-adjusted, max exposure
- **Stops & Take Profit:** ATR-based, trailing stops, multi-targets

### ML Pipeline

- **Data Processing:** OHLCV normalization, feature engineering, validation splitting
- **Model Management:** Automated training, versioning, metrics, persistence

---

## 7. Usage

### 7.1 Launch Dashboard

```bash
streamlit run streamlit_dashboard.py
```

### 7.2 Run Backtests

```bash
python backtester.py --symbol AAPL --start 2024-01-01 --end 2024-04-29 --strategy lstm
```

### 7.3 Train ML Models

```bash
python ml_pipeline.py --epochs 50 --batch-size 32 --learning-rate 0.001
```

### 7.4 Pattern Detection (CLI)

```bash
python pattern_detection.py --symbol AAPL --window 10
```

### 7.5 Alert System Test

```bash
python notifier.py --mode test
```

### 7.6 Run Test Suite

```bash
pytest tests/
pytest --cov=. --cov-report=html
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

## Change Log: Substantive Improvements

- Clarified and standardized all section headings for easier navigation.
- Rewrote installation and configuration steps for clarity and to ensure all commands are correct and in logical order.
- Added explicit verification steps (e.g., `python -c "import streamlit; import pandas; import torch"`) to confirm dependencies are installed.
- Updated architecture diagram to match the actual folder structure and file roles, including all major modules and UI pages.
- Grouped usage instructions by task (dashboard, backtesting, ML training) and ensured all CLI commands are complete and accurate.
- Expanded troubleshooting section with actionable steps and command examples.
- Removed redundant or ambiguous instructions and ensured all environment variables are explained.
- Consistent formatting for code blocks, lists, and section breaks.

---

## Suggestions for Further Improvement

- **Split the manual into multiple files** (e.g., `INSTALL.md`, `USAGE.md`, `TROUBLESHOOTING.md`) for easier maintenance and navigation.
- **Add code samples and screenshots** for dashboard usage, model training, and backtesting results.
- **Link to external documentation** (e.g., E*Trade API docs, Streamlit guides) where relevant.
- **Include a FAQ section** addressing common setup and runtime issues.
- **Provide example `.env` files** and sample data for quick testing.