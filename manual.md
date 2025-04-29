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

The E*Trade Candlestick Trading Bot combines algorithmic trading strategies with machine learning to automate stock trading via the E*Trade API. Key components include:

- **Real-time Dashboard**: Live market data monitoring and trade execution
- **Pattern Recognition**: Both rule-based and ML-powered candlestick pattern detection
- **Risk Management**: Automated position sizing and stop-loss calculation
- **Backtesting**: Historical strategy simulation with performance metrics
- **Notifications**: Multi-channel trade alerts and system status updates

## Prerequisites

### Required
- Python 3.8 or higher
- Git
- E*Trade Developer Account
  - Sandbox environment credentials
  - Production API keys
- 64-bit Windows/Linux/macOS

### Optional
- Docker Desktop (for containerized deployment)
- SMTP Server Access (for email notifications)
- Twilio Account (for SMS notifications)
- Slack Workspace (for chat notifications)

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

Edit `.env` with your credentials:

```plaintext
# E*TRADE API (Required)
ETRADE_CONSUMER_KEY=your_key
ETRADE_CONSUMER_SECRET=your_secret
ETRADE_OAUTH_TOKEN=your_token
ETRADE_OAUTH_TOKEN_SECRET=your_token_secret
ETRADE_ACCOUNT_ID=your_account_id

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
│   ├── etrade_candlestick_bot.py    # Trading engine
│   ├── streamlit_dashboard.py        # Web interface
│   ├── backtester.py                # Strategy testing
│   └── ml_pipeline.py               # Model training
│
├── Supporting Modules
│   ├── risk_manager.py              # Position sizing
│   ├── indicators.py                # Technical analysis
│   ├── model_manager.py             # ML model handling
│   ├── notifier.py                  # Alerts system
│   └── performance_utils.py         # Optimization
│
├── Web Interface
│   └── pages/
│       ├── live_dashboard.py
│       ├── backtest.py
│       ├── model_training.py
│       └── settings.py
│
└── Tests & CI
    ├── tests/
    └── .github/workflows/
```

## Core Features

### Pattern Detection
- Classic candlestick patterns (Hammer, Doji, etc.)
- LSTM-based pattern recognition
- Confidence scoring system

### Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Custom indicator support

### Risk Management
- Position sizing based on account value
- ATR-based stop-loss calculation
- Take-profit automation
- Maximum drawdown controls

### Machine Learning Pipeline
- Data preprocessing
- LSTM model training
- Pattern recognition
- Model persistence

## Running the Application

### Streamlit Dashboard
```bash
streamlit run streamlit_dashboard.py
```

### Backtesting Tool
```bash
python backtester.py --symbol AAPL --start 2024-01-01 --end 2024-04-29 --strategy lstm
```

### Model Training
```bash
python ml_pipeline.py --epochs 50 --batch-size 32 --learning-rate 0.001
```

## Advanced Usage

### Custom Strategy Development
```python
from backtester import Backtest

class MyStrategy(Backtest):
    def generate_signals(self):
        # Your strategy logic here
        pass
```

### Performance Optimization
- Use `@st.cache_data` for expensive operations
- Implement batch processing for data fetching
- Enable async operations where possible

## Development

### Testing
```bash
pytest tests/
```

### Docker Deployment
```bash
docker build -t etrade-bot .
docker run -p 8501:8501 etrade-bot
```

## Support

- GitHub Issues: Bug reports and feature requests
- Discussions: Technical questions and community support
- Documentation: Additional guides in `/docs`

---
Licensed under MIT © 2025