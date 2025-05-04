# E*Trade Candlestick Trading Bot & Dashboard

E*Trade Candlestick Trading Bot & Dashboard
An enterprise-grade trading platform that combines classic technical analysis with machine learning for automated E*Trade trading. Features a Streamlit dashboard for real-time monitoring, robust risk management, and a comprehensive backtesting and ML pipeline.

---
📑 Table of Contents
Key Features
System Requirements
Installation
Configuration
Project Structure
Usage
Machine Learning Pipeline
Testing
Documentation
Contributing
License
Support

## 🎯 Key Features

Real-Time Trading Dashboard

Dynamic symbol watchlist
Interactive candlestick charts (Plotly)
Real-time pattern detection (rule-based & ML)
Integrated PatternNN model predictions
Risk-managed order execution
Advanced Technical Analysis

Candlestick pattern recognition (Hammer, Doji, Engulfing, etc.)
Technical indicators (RSI, MACD, Bollinger Bands)
Custom indicator framework
ATR-based position sizing
Machine Learning Pipeline

Pattern Neural Network (PatternNN) for pattern classification
Automated data preparation and feature engineering
Model persistence/versioning (ModelManager)
Configurable training parameters
Real-time inference integration
Risk Management System

Position size calculator
Dynamic stop-loss and take-profit
Portfolio exposure controls
Comprehensive Backtesting

Rule-based and ML strategies
Historical OHLCV data simulation
Performance metrics: Sharpe Ratio, Max Drawdown, Win Rate
Enterprise Integration

Multi-channel notifications: Email (SMTP), SMS (Twilio), Slack
Streamlit caching and async data fetching
Containerized deployment (Docker)


🔧 System Requirements
Python 3.8+
Git
E*Trade Developer Account (sandbox and/or production API keys)
(Optional) SMTP server, Twilio account, Slack webhook
Docker (for containerized deployment)
---

## 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-org>/etrade-bot.git
   cd etrade-bot
   ```

2. **Set Up Python Environment**

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

3. **Install Dependencies**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python -c "import streamlit; import pandas; import torch"
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   python utils/validate_config.py
   # Edit .env with your credentials and settings
   ```

---

## ⚙️ Configuration

Edit the `.env` file with your credentials and settings:

```dotenv
# E*TRADE API (Required)
ETRADE_CONSUMER_KEY=your_sandbox_consumer_key
ETRADE_CONSUMER_SECRET=your_sandbox_consumer_secret
ETRADE_OAUTH_TOKEN=your_sandbox_access_token
ETRADE_OAUTH_TOKEN_SECRET=your_sandbox_access_token_secret
ETRADE_ACCOUNT_ID=your_sandbox_account_id
ETRADE_USE_SANDBOX=true

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

## 🏗️ Project Structure

```
stocktrader/
├── streamlit_dashboard.py       # Web dashboard entry point
├── backtester.py                # Strategy backtesting
├── ml_pipeline.py               # ML training & inference
│
├── utils/
│   ├── etrade_candlestick_bot.py   # Trading logic & E*TRADE API
│   ├── indicators.py               # Technical indicators
│   ├── model_manager.py            # Model persistence/versioning
│   ├── performance_utils.py        # Dashboard state & pattern detection
│   ├── validate_config.py          # Config validation
│   ├── validation.py               # Input validation helpers
│   ├── risk_manager.py             # Position sizing & risk controls
│   └── getuservar.py               # E*TRADE OAuth helper
│
├── core/
│   └── notifier.py                 # Notification system
│
├── data/
│   ├── data_loader.py              # Data download (Yahoo Finance)
│   ├── io.py                       # I/O utilities
│   ├── model_trainer.py            # Model training pipeline
│   └── data_dashboard.py           # Data visualization dashboard
│
├── train/
│   ├── training_pipeline.py        # ML training pipeline
│   └── trainer.py                  # PatternNN model training
│
├── models/
│   └── patterns_nn.py              # PatternNN model definition
│
├── patterns.py                     # Candlestick pattern detection
│
├── pages/                          # Streamlit multi-page app
│   ├── live_dashboard.py           # Real-time monitoring
│   ├── backtest.py                 # Backtesting UI
│   ├── model_training.py           # ML pipeline UI
│   └── settings.py                 # System configuration UI
│
├── tests/                          # Unit & integration tests
└── .github/workflows/              # CI/CD pipelines
```

---

## 🚀 Usage

### Launch Dashboard

```bash
streamlit run streamlit_dashboard.py
```

### Pattern Detection (CLI)

```bash
python pattern_detection.py --symbol AAPL --window 10
```

### Strategy Backtesting

```bash
python backtester.py --symbol AAPL --start 2024-01-01 --end 2024-04-29 --strategy lstm
```

### ML Model Training

```bash
python ml_pipeline.py --epochs 50 --batch-size 32 --learning-rate 0.001
```

### Alert System Testing

```bash
python notifier.py --mode test
```

### Test Suite Execution

```bash
pytest --cov=. --cov-report=html
```

---

## 🧪 Testing

- Comprehensive test suite covering:
  - Pattern detection algorithms
  - Backtesting engine
  - ML pipeline components
  - Risk management calculations
- Automated CI/CD via GitHub Actions
- Code coverage reporting

---

## 📚 Documentation

See `/docs` for:

- API Reference
- Configuration Guide
- Strategy Development
- ML Model Training
- Deployment Guide

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

Please review our contribution guidelines and code of conduct.

---

## 📄 License

MIT License - Copyright (c) 2025 [Your Organization]

---

## 🔗 Support

- Issue Tracker: GitHub Issues
- Wiki: Project Documentation
- Contact: support@your-org.com

---

*Note: This project is for educational purposes. Always verify trading strategies thoroughly before deploying with real capital.*

---

# Machine Learning Pipeline

Your final dashboard sequence:

Download Data → saves CSVs and displays price plot.

Train Model → saves pipelines and shows metrics.

Run Model on Data → loads each pipeline, runs predict(), displays signal chart.

Combine with Patterns → uses CandlestickPatterns to filter your model’s signals, displays final buy dates or overlays on charts.

This gives you an end-to-end flow: from raw OHLCV to feature engineering → model training → inference → pattern filtering → final trade signals—all within the same Streamlit UI.

## Enhanced Model Training Framework

### ModelTrainer Features

- **Robust Feature Engineering**
  ```python
  trainer = ModelTrainer(config)
  df = trainer.feature_engineering(data)
  # Calculates rolling window features, technical indicators, and target labels
  ```

- **Training Parameters**
  ```python
  training_params = TrainingParams(
      n_estimators=100,
      max_depth=10,
      min_samples_split=10,
      cv_folds=5,
      random_state=42
  )
  ```

- **Automatic Time-Series Cross Validation**
  ```python
  model, metrics, cm, report = trainer.train_model(
      df,
      params=training_params
  )
  # Metrics: training/test/final, confusion matrix, classification report
  ```

- **Model Saving**
  ```python
  path = trainer.save_model(
      model,
      symbol="AAPL",
      interval="1d"
  )
  # Artifacts: trained model, scaler, feature list, metadata, metrics
  ```

- **Example Usage**
  ```python
  from data.model_trainer import ModelTrainer, TrainingParams, FeatureConfig

  config = {'MODEL_DIR': 'models/'}
  trainer = ModelTrainer(
      config,
      feature_config=FeatureConfig(),
      training_params=TrainingParams(
          n_estimators=100,
          max_depth=10,
          min_samples_split=10
      )
  )
  df = pd.read_csv('data/AAPL_1d.csv')
  model, metrics, cm, report = trainer.train_model(df)
  ```

- **PatternModelTrainer Example**
  ```python
  from train.trainer import PatternModelTrainer, TrainingConfig
  from stocktrader.utils.model_manager import ModelManager
  from stocktrader.etrade_candlestick_bot import ETradeClient

  config = TrainingConfig(
      epochs=10,
      seq_len=10, 
      batch_size=32,
      learning_rate=0.001
  )
  trainer = PatternModelTrainer(
      client=ETradeClient(...),
      model_manager=ModelManager(...),
      config=config
  )
  model = trainer.train_model(
      symbols=["AAPL", "MSFT", "GOOG"],
      metadata={"version": "1.0"}
  )
  ```

These examples reflect the actual implementation in `model_trainer.py` and related files, showing:

1. Robust ML pipeline with validation
2. Comprehensive feature engineering
3. Time-series cross-validation
4. Detailed metrics and reporting
5. Standardized model persistence
6. Error handling and testing support

---

Let us know if you need more detailed examples or have questions about specific modules!
