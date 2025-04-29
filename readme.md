# E*Trade Candlestick Trading Bot & Dashboard

An enterprise-grade trading platform that combines traditional technical analysis with machine learning for automated E*Trade trading. Features a Streamlit dashboard for real-time monitoring and a comprehensive backtesting framework.

## 🎯 Key Features

- **Real-Time Trading Dashboard**
  - Dynamic symbol watchlist management
  - Interactive candlestick charts with Plotly
  - Real-time pattern detection and annotations
  - Integrated LSTM model predictions
  - Risk-managed order execution interface
  
- **Advanced Technical Analysis**
  - Candlestick pattern recognition engine
  - Technical indicators suite (RSI, MACD, Bollinger Bands)
  - Custom indicator development framework
  - ATR-based position sizing
  
- **Machine Learning Pipeline**
  - Pattern Neural Network (PatternNN) for technical analysis
  - Real-time pattern recognition and classification
  - Automated data preparation and preprocessing
  - Model persistence and versioning with ModelManager
  - Configurable training parameters
  - Real-time inference integration
  
- **Risk Management System**
  - Position size calculator
  - Dynamic stop-loss placement
  - Take-profit optimization
  - Portfolio exposure controls
  
- **Comprehensive Backtesting**
  - Multiple strategy support (Rule-based/ML)
  - Historical OHLCV data simulation
  - Performance metrics:
    - Sharpe Ratio
    - Maximum Drawdown
    - Win Rate
    - Risk-adjusted Returns
  
- **Enterprise Integration**
  - Multi-channel notifications:
    - Email (SMTP)
    - SMS (Twilio)
    - Slack alerts
  - Performance optimization:
    - Streamlit caching
    - Async data fetching
  - Containerized deployment support

## 🔧 System Requirements

- Python 3.8+
- E*Trade Developer Account:
  - Sandbox environment credentials
  - Production API keys
- (Optional) Integration Services:
  - SMTP server access
  - Twilio account
  - Slack workspace with webhook permissions
- Docker (for containerized deployment)
- Git

## 📦 Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/<organization>/etrade-bot.git
   cd etrade-bot
   ```

2. **Environment Setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Unix
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

Required environment variables:
```dotenv
# E*Trade Authentication
ETRADE_CONSUMER_KEY=your_key
ETRADE_CONSUMER_SECRET=your_secret
ETRADE_OAUTH_TOKEN=your_token
ETRADE_OAUTH_TOKEN_SECRET=your_token_secret
ETRADE_ACCOUNT_ID=your_account_id

# Notification Services
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USER=your_email
SMTP_PASS=your_password

TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=your_number

SLACK_WEBHOOK_URL=your_webhook_url
```

## 🚀 Usage

### Trading Dashboard
```bash
streamlit run streamlit_dashboard.py
```

### Pattern Detection
```bash
python pattern_detection.py --symbol AAPL --window 10
```

### Strategy Backtesting
```bash
python backtester.py --symbol AAPL --start 2021-01-01 --end 2025-04-28 --strategy rule
```

### ML Model Training
```bash
python ml_pipeline.py --model pattern_nn --data-dir data/ohlcv/ --epochs 30 --window-size 10 --batch-size 32 --lr 1e-3
```

### Alert System Testing
```bash
python notifier.py --mode test
```

### Test Suite Execution
```bash
pytest --cov=. --cov-report=html
```

## 🧪 Testing

- Comprehensive test suite covering:
  - Pattern detection algorithms
  - Backtesting engine
  - ML pipeline components
  - Risk management calculations
- Automated CI/CD via GitHub Actions
- Code coverage reporting

## 📚 Documentation

Detailed documentation available in `/docs`:
- API Reference
- Configuration Guide
- Strategy Development
- ML Model Training
- Deployment Guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

Please review our contribution guidelines and code of conduct.

## 📄 License

MIT License - Copyright (c) 2025 [Your Organization]

## 🔗 Support

- Issue Tracker: GitHub Issues
- Wiki: Project Documentation
- Contact: support@your-org.com

---
*Note: This project is for educational purposes. Always verify trading strategies thoroughly before deploying with real capital.*

These changes reflect the actual implementation we can see in the repository, particularly:

1. The presence of PatternNN model in the codebase
2. The usage of window-based pattern detection as shown in test_performance_utils.py
3. The integration with ModelManager for model persistence
4. The real-time inference capabilities demonstrated in the backtest.py file

The rest of the readme appears accurate and doesn't need changes based on the current codebase structure.
