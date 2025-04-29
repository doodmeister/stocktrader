# E*Trade Candlestick Trading Bot & Dashboard

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [System Architecture](#system-architecture)
6. [Core Features](#core-features)
7. [Running the Application](#running-the-application)
8. [Advanced Usage](#advanced-usage)
9. [Development](#development)
10. [Support](#support)

## Overview

The E*Trade Candlestick Trading Bot is an enterprise-grade algorithmic trading platform that combines technical analysis with machine learning for automated stock trading via the E*Trade API. The system features:

- **Real-time Dashboard**: Interactive Plotly-based candlestick charts with live data monitoring
- **Pattern Recognition**: Dual-mode pattern detection (rule-based + ML)
- **Risk Management**: Dynamic position sizing and automated stop-loss
- **Backtesting Engine**: Historical strategy validation with comprehensive metrics
- **Enterprise Integration**: Multi-channel alerting system

## Prerequisites

### Required
- Python 3.8 or higher
- Git
- E*Trade Developer Account
  - Sandbox environment for testing
  - Production API credentials
  - Account with trading permissions
- 64-bit Windows/Linux/macOS
- Minimum 8GB RAM recommended

### Optional
- Docker Desktop (containerization)
- SMTP Server (email alerts)
- Twilio Account (SMS notifications)
- Slack Workspace (webhook integration)
- GPU support for ML training

## Installation

1. **Clone the Repository**
```bash
git clone https://github.com/<your-org>/etrade-bot.git
cd etrade-bot
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment**
```bash
cp .env.example .env
```

## Configuration

### API Credentials
Edit `.env` with your credentials:

```plaintext
# E*TRADE API (Required)
ETRADE_CONSUMER_KEY=your_key
ETRADE_CONSUMER_SECRET=your_secret
ETRADE_OAUTH_TOKEN=your_token
ETRADE_OAUTH_TOKEN_SECRET=your_token_secret
ETRADE_ACCOUNT_ID=your_account_id
ETRADE_SANDBOX=True  # Set to False for production

# Trading Configuration
MAX_POSITIONS=5
MAX_LOSS_PERCENT=0.02
PROFIT_TARGET_PERCENT=0.03
MAX_DAILY_LOSS=0.05
SYMBOLS=AAPL,MSFT,GOOG

# Notification Settings (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email
SMTP_PASS=your_password

TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1234567890

SLACK_WEBHOOK_URL=your_webhook_url
```

## System Architecture

```
etrade-bot/
├── Core Components
│   ├── etrade_candlestick_bot.py    # Main trading engine
│   ├── streamlit_dashboard.py        # Web interface & monitoring
│   ├── backtester.py                # Strategy validation
│   └── ml_pipeline.py               # ML model lifecycle
│
├── Supporting Modules
│   ├── risk_manager.py              # Position & risk control
│   ├── indicators.py                # Technical analysis suite
│   ├── model_manager.py             # Model versioning & inference
│   ├── notifier.py                  # Alert system
│   └── performance_utils.py         # System optimization
│
├── Web Interface
│   └── pages/
│       ├── live_dashboard.py        # Real-time monitoring
│       ├── backtest.py              # Strategy testing
│       ├── model_training.py        # ML pipeline control
│       └── settings.py              # System configuration
│
└── Tests & CI
    ├── tests/                       # Unit & integration tests
    └── .github/workflows/           # CI/CD pipelines
```

## Core Features

### Pattern Detection
- **Classic Patterns**
  - Hammer, Doji, Engulfing patterns
  - Real-time pattern scanning
  - Confidence scoring (0-100%)
  - Custom pattern definitions

- **ML-Based Recognition**
  - LSTM neural networks
  - Pattern classification
  - Probability scoring
  - Model versioning

### Technical Analysis
- **Core Indicators**
  - RSI (configurable periods)
  - MACD (customizable parameters)
  - Bollinger Bands (dynamic bands)
  - Volume analysis

- **Custom Indicators**
  - Indicator composition
  - Custom calculation engine
  - Real-time updates
  - Backtesting support

### Risk Management
- **Position Sizing**
  - Account-based scaling
  - Risk-adjusted positions
  - Maximum exposure limits
  - Position correlation checks

- **Stop Loss & Take Profit**
  - ATR-based stops
  - Trailing stop logic
  - Multiple TP targets
  - Break-even automation

### ML Pipeline
- **Data Processing**
  - OHLCV normalization
  - Feature engineering
  - Training set generation
  - Validation splitting

- **Model Management**
  - Training automation
  - Version control
  - Performance metrics
  - Model persistence

## Running the Application

### Dashboard Launch
```bash
streamlit run streamlit_dashboard.py
```

### Strategy Testing
```bash
python backtester.py --symbol AAPL --start 2024-01-01 --end 2024-04-29 --strategy lstm
```

### ML Training
```bash
python ml_pipeline.py --epochs 50 --batch-size 32 --learning-rate 0.001
```

## Advanced Usage

### Custom Strategies
```python
from backtester import Backtest

class MyStrategy(Backtest):
    def generate_signals(self):
        """
        Custom trading logic implementation
        Returns: DataFrame with signals
        """
        pass
```

### Performance Optimization
- **Caching**
  - `@st.cache_data` for data operations
  - Database result caching
  - Model inference caching

- **Async Operations**
  - Data fetching
  - Order execution
  - Alert dispatching

## Development

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=. --cov-report=html
```

### Docker Deployment
```bash
# Build image
docker build -t etrade-bot .

# Run container
docker run -p 8501:8501 -v $(pwd)/data:/app/data etrade-bot
```

## Support

- **Issue Tracking**: GitHub Issues for bugs & features
- **Community**: Technical discussions & support
- **Documentation**: Additional guides in `/docs`

---
Licensed under MIT © 2025