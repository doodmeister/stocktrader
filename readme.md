# E*Trade Candlestick Trading Bot & Dashboard

A Streamlit-based trading dashboard and backtesting framework for E*Trade, with both rule-based candlestick pattern detection and an LSTM-based pattern learner.  Full CI/CD and testing support included.

## 🚀 Features

- **Live Dashboard**
  - Add/remove symbols on the fly  
  - Candlestick plots (Plotly) with pattern annotations  
  - Train & infer a PyTorch LSTM model on live data  
  - Manual Buy/Sell buttons, risk management overlays  
- **Backtester**
  - Offline simulation of strategies on historical OHLCV  
  - Metrics: Sharpe, max drawdown, win rate  
- **Risk Manager**
  - ATR-based stops, position sizing modules  
- **Indicators**
  - RSI, MACD, Bollinger Bands (via `pandas-ta` fallback)  
- **ML Pipeline**
  - Data prep, train/val split, model save/load, evaluation  
- **Notifications**
  - SMTP email, Twilio SMS, Slack webhook  
- **CI/CD & Testing**
  - Pytest suite for patterns, backtester, ML pipeline  
  - GitHub Actions workflows for lint, test, build  

## 📁 Directory Structure

```
.
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── project_plan.md
│
├── etrade_candlestick_bot.py
├── performance_utils.py
├── streamlit_dashboard.py
├── backtester.py
├── risk_manager.py
├── indicators.py
├── model_manager.py
├── notifier.py
├── ml_pipeline.py
│
├── models/
│   └── pattern_nn_v1.pth
│
├── tests/
│   ├── test_patterns.py
│   ├── test_backtester.py
│   └── test_ml_pipeline.py
│
├── pages/
│   ├── live_dashboard.py
│   ├── backtest.py
│   ├── model_training.py
│   └── settings.py
│
└── .github/
    └── workflows/
        ├── ci.yml
        └── deploy.yml
```

## ⚙️ Prerequisites

- Python 3.8+  
- E*Trade developer account (sandbox & production keys)  
- (Optional) Twilio account & Slack webhook for notifications  
- Docker & Git  

## 🛠 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-org>/etrade-bot.git
   cd etrade-bot
   ```
2. **Create and activate a virtualenv**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Copy and populate** `.env.example`
   ```bash
   cp .env.example .env
   # Edit .env with your keys & secrets
   ```

## 📝 Configuration

Populate `.env` with:
```dotenv
ETRADE_CONSUMER_KEY=
ETRADE_CONSUMER_SECRET=
ETRADE_OAUTH_TOKEN=
ETRADE_OAUTH_TOKEN_SECRET=
ETRADE_ACCOUNT_ID=

SMTP_SERVER=
SMTP_PORT=
SMTP_USER=
SMTP_PASS=

TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=

SLACK_WEBHOOK_URL=
```

## ▶️ Usage

### Streamlit Dashboard
```bash
streamlit run streamlit_dashboard.py
```

### Backtesting
```bash
python backtester.py --symbol AAPL --start 2021-01-01 --end 2025-04-28 --strategy rule
```

### ML Pipeline
```bash
python ml_pipeline.py --data-dir data/ohlcv/ --epochs 30 --seq-len 10 --lr 1e-3
```

### Notifications Test
```bash
python notifier.py --mode test
```

### Running Tests
```bash
pytest --cov=.
```

## 🤝 Contributing

Fork, branch, commit, PR. Ensure tests pass and follow code style.

## 📄 License

MIT © Your Name
