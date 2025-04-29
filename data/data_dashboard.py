"""Stock Data Dashboard Module

A Streamlit dashboard for fetching, displaying, managing, and training on historical OHLCV data.
Follows SOLID principles and includes robust error handling.
"""

import logging
from logging.handlers import RotatingFileHandler
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

from stocktrader.data.data_loader import save_to_csv
from stocktrader.core.notifier import Notifier
from stocktrader.utils.validation import sanitize_input
from stocktrader.utils.io import create_zip_archive
from stocktrader.train.training_pipeline import feature_engineering

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("dashboard.log", maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    DEFAULT_SYMBOLS: str = "AAPL,MSFT"
    VALID_INTERVALS: List[str] = ("1d", "1h", "30m", "15m", "5m", "1m")
    MAX_INTRADAY_DAYS: int = 60
    REFRESH_INTERVAL: int = 300
    DATA_DIR: Path = Path("data")
    MODEL_DIR: Path = Path("models")
    CACHE_TTL: int = 3600

class DataDashboard:
    def __init__(self):
        self.config = DashboardConfig()
        self.saved_paths = []
        self.notifier = Notifier()
        self.symbols = []
        self.interval = "1d"
        self.start_date = date.today() - timedelta(days=365)
        self.end_date = date.today()
        self.clean_old = True
        self.auto_refresh = False

        self.config.DATA_DIR.mkdir(exist_ok=True)
        self.config.MODEL_DIR.mkdir(exist_ok=True)

    def _setup_page(self):
        st.set_page_config(page_title="Stock Data Dashboard", layout="centered")
        st.title("📈 Stock OHLCV Data Downloader and Trainer")

    def _render_inputs(self):
        st.info("Note: Intraday intervals are limited to 60 days.")
        symbols_input = st.text_input("Ticker Symbols (comma-separated)", value=self.config.DEFAULT_SYMBOLS)
        self.symbols = [sanitize_input(s.strip()) for s in symbols_input.split(",") if s.strip()]

        self.interval = st.selectbox("Data Interval", options=self.config.VALID_INTERVALS, index=0)

        col1, col2 = st.columns(2)
        with col1:
            self.start_date = st.date_input("Start Date", value=self.start_date)
        with col2:
            self.end_date = st.date_input("End Date", value=self.end_date)

        self.clean_old = st.checkbox("🧹 Clean old CSVs before fetching?", value=True)
        self.auto_refresh = st.checkbox("🔄 Auto-refresh every 5 minutes?", value=False)

    def _handle_user_actions(self):
        if st.button("Fetch Data"):
            self._fetch_data()

        if self.auto_refresh:
            st.caption("🔄 Auto-refresh enabled. Refreshing every 5 minutes...")
            import time
            time.sleep(self.config.REFRESH_INTERVAL)
            st.experimental_rerun()

    def _fetch_data(self):
        if self.clean_old:
            for file in self.config.DATA_DIR.glob("*.csv"):
                file.unlink()

        for symbol in self.symbols:
            if not symbol:
                continue

            st.subheader(f"📊 Data for {symbol}")
            with st.spinner(f"Fetching {symbol}..."):
                try:
                    df = self._download(symbol)
                    if df is not None:
                        path = self._save(df, symbol)
                        self._show(df)
                        self.saved_paths.append(path)
                except Exception as e:
                    logger.error(f"{symbol} fetch error: {e}")
                    st.error(f"Failed to fetch data for {symbol}")

        if len(self.saved_paths) > 1:
            self._offer_zip()

        self._train_models()

    def _download(self, symbol: str) -> Optional[pd.DataFrame]:
        df = yf.download(
            symbol,
            start=self.start_date.strftime("%Y-%m-%d"),
            end=self.end_date.strftime("%Y-%m-%d"),
            interval=self.interval,
            progress=False
        )
        if df.empty:
            return None
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def _save(self, df: pd.DataFrame, symbol: str) -> Path:
        path = self.config.DATA_DIR / f"{symbol}_{self.interval}.csv"
        save_to_csv(df, str(path))
        return path

    def _show(self, df: pd.DataFrame):
        st.dataframe(df.tail(10))
        st.line_chart(df['Close'])
        st.caption(f"✅ Rows: {len(df)} | Date Range: {df.index.min().date()} → {df.index.max().date()}")
        st.caption(f"✅ Latest Close: {df['Close'].iloc[-1]:.2f}")

    def _offer_zip(self):
        zip_data = create_zip_archive(self.saved_paths)
        st.download_button(
            label="📦 Download All as ZIP",
            data=zip_data,
            file_name="stock_data.zip",
            mime="application/zip"
        )

    def _train_models(self):
        if not st.checkbox("📚 Train models after fetching?", value=False):
            return

        st.subheader("📚 Training Models")

        for path in self.saved_paths:
            filename = path.stem
            symbol, interval = filename.split("_")

            with st.spinner(f"Training model for {symbol} [{interval}]..."):
                try:
                    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
                    df = feature_engineering(df)

                    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'volatility', 'sma_5', 'sma_10']
                    X = df[features]
                    y = df['target']

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)

                    train_acc = model.score(X_train, y_train)
                    test_acc = model.score(X_test, y_test)
                    st.success(f"✅ {symbol} | Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")

                    y_pred = model.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay(cm).plot(ax=ax)
                    st.pyplot(fig)

                    model_path = self.config.MODEL_DIR / f"{symbol}_{interval}_model.pkl"
                    joblib.dump(model, model_path)
                    with open(model_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download {symbol} Model (.pkl)",
                            data=f,
                            file_name=model_path.name,
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    logger.error(f"Training failed for {symbol}: {e}")
                    st.error(f"❌ Failed to train model for {symbol}")

    def render_dashboard(self):
        self._setup_page()
        self._render_inputs()
        self._handle_user_actions()

def main():
    try:
        dashboard = DataDashboard()
        dashboard.render_dashboard()
    except Exception as e:
        logger.error(f"Dashboard startup error: {e}")
        st.error("An unexpected error occurred.")

if __name__ == "__main__":
    main()
