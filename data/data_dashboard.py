"""Stock Data Dashboard Module

Streamlit dashboard for stock OHLCV download, model training, and evaluation.
Includes model visualization and dynamic hyperparameter tuning.
"""

import streamlit as st
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from .config import DashboardConfig
from .logger import setup_logger
from .model_trainer import ModelTrainer, TrainingParams
from .data_validator import DataValidator
from stocktrader.utils.validation import sanitize_input
from stocktrader.utils.io import create_zip_archive
from stocktrader.core.notifier import Notifier

logger = setup_logger(__name__)

class DataDashboard:
    def __init__(self):
        self.config = DashboardConfig()
        self._setup_directories()
        self.saved_paths: List[Path] = []
        self.notifier = Notifier()
        self.validator = DataValidator()
        self.model_trainer = ModelTrainer(self.config)
        self._init_state()

    def _init_state(self) -> None:
        self.symbols: List[str] = []
        self.interval = "1d"
        self.start_date = date.today() - timedelta(days=365)
        self.end_date = date.today()
        self.clean_old = True
        self.auto_refresh = False

    def _setup_directories(self) -> None:
        self.config.DATA_DIR.mkdir(exist_ok=True)
        self.config.MODEL_DIR.mkdir(exist_ok=True)

    def _render_inputs(self) -> None:
        st.info("Note: Intraday intervals are limited to 60 days.")
        symbols_input = st.text_input("Ticker Symbols (comma-separated)", value=self.config.DEFAULT_SYMBOLS)
        self.symbols = self.validator.validate_symbols(symbols_input)
        self.interval = st.selectbox("Data Interval", options=self.config.VALID_INTERVALS, index=0)

        col1, col2 = st.columns(2)
        with col1:
            self.start_date = st.date_input("Start Date", value=self.start_date)
        with col2:
            self.end_date = st.date_input("End Date", value=self.end_date)

        if not self.validator.validate_dates(self.start_date, self.end_date):
            st.error("Invalid date range selected")

        self.clean_old = st.checkbox("🧹 Clean old CSVs before fetching?", value=True)
        self.auto_refresh = st.checkbox("🔄 Auto-refresh every 5 minutes?", value=False)

    @st.cache_data(ttl=3600)
    def _download(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(
                symbol,
                start=self.start_date.strftime("%Y-%m-%d"),
                end=self.end_date.strftime("%Y-%m-%d"),
                interval=self.interval,
                progress=False
            )
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return None

    def _handle_user_actions(self) -> None:
        if st.button("Fetch Data"):
            if self.clean_old:
                for file in self.config.DATA_DIR.glob("*.csv"):
                    file.unlink()

            for symbol in self.symbols:
                df = self._download(symbol)
                if df is None:
                    st.error(f"No data for {symbol}")
                    continue

                path = self.config.DATA_DIR / f"{symbol}_{self.interval}.csv"
                df.to_csv(path)
                self.saved_paths.append(path)

                st.subheader(f"📊 {symbol}")
                st.dataframe(df.tail(10))
                st.line_chart(df["Close"])

            if len(self.saved_paths) > 1:
                zip_data = create_zip_archive(self.saved_paths)
                st.download_button("📦 Download All as ZIP", data=zip_data, file_name="stock_data.zip")

        # Model training interface
        if st.checkbox("📚 Train models after fetching?", value=False):
            st.subheader("📚 Model Training")

            n_estimators = st.slider("n_estimators", 10, 300, 100, step=10)
            max_depth = st.slider("max_depth", 1, 20, 10)
            min_samples_split = st.slider("min_samples_split", 2, 20, 10)

            params = TrainingParams(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split
            )

            for path in self.saved_paths:
                symbol, _ = path.stem.split("_", 1)
                df = pd.read_csv(path, index_col="date", parse_dates=True)

                try:
                    model, metrics, cm, report = self.model_trainer.train_model(df, params)
                    model_path = self.model_trainer.save_model(model, symbol, self.interval)

                    st.success(f"✅ {symbol} trained. Model saved to: {model_path.name}")
                    st.json(metrics)

                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay(cm).plot(ax=ax)
                    st.pyplot(fig)

                    with st.expander(f"📄 Classification Report for {symbol}"):
                        st.text(report)

                except Exception as e:
                    st.error(f"Training failed for {symbol}: {e}")

    def render_dashboard(self) -> None:
        try:
            st.set_page_config(page_title="Stock Data Dashboard", layout="centered")
            st.title("📈 Stock OHLCV Data Downloader and Trainer")
            self._render_inputs()
            self._handle_user_actions()
        except Exception as e:
            logger.error(f"Error rendering dashboard: {e}")
            st.error("An unexpected error occurred.")
