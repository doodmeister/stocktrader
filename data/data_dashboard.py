"""Stock Data Dashboard Module

A Streamlit dashboard for fetching, displaying, managing, and training on historical OHLCV data.
Implements SOLID principles with robust error handling and validation.
"""

import streamlit as st
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional
import pandas as pd
import yfinance as yf

from .config import DashboardConfig
from .logger import setup_logger
from .model_trainer import ModelTrainer
from .data_validator import DataValidator
from stocktrader.utils.validation import sanitize_input
from stocktrader.utils.io import create_zip_archive
from stocktrader.core.notifier import Notifier

logger = setup_logger(__name__)

class DataDashboard:
    """Main dashboard class for stock data visualization and model training."""
    
    def __init__(self):
        """Initialize dashboard with configuration and dependencies."""
        self.config = DashboardConfig()
        self._setup_directories()
        self.saved_paths: List[Path] = []
        self.notifier = Notifier()
        self.validator = DataValidator()
        self.model_trainer = ModelTrainer(self.config)
        
        # State initialization
        self._init_state()

    def _init_state(self) -> None:
        """Initialize dashboard state variables."""
        self.symbols: List[str] = []
        self.interval = "1d"
        self.start_date = date.today() - timedelta(days=365)
        self.end_date = date.today()
        self.clean_old = True
        self.auto_refresh = False

    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        try:
            self.config.DATA_DIR.mkdir(exist_ok=True)
            self.config.MODEL_DIR.mkdir(exist_ok=True)
        except PermissionError as e:
            logger.error(f"Permission error creating directories: {e}")
            raise RuntimeError("Unable to create required directories")

    def _render_inputs(self) -> None:
        """Render and handle user input fields."""
        st.info("Note: Intraday intervals are limited to 60 days.")
        
        # Symbol input with validation
        symbols_input = st.text_input(
            "Ticker Symbols (comma-separated)", 
            value=self.config.DEFAULT_SYMBOLS
        )
        self.symbols = self.validator.validate_symbols(symbols_input)
        
        # Interval selection
        self.interval = st.selectbox(
            "Data Interval", 
            options=self.config.VALID_INTERVALS, 
            index=0
        )
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            self.start_date = st.date_input("Start Date", value=self.start_date)
        with col2:
            self.end_date = st.date_input("End Date", value=self.end_date)
            
        if not self.validator.validate_dates(self.start_date, self.end_date):
            st.error("Invalid date range selected")
            
        # Additional options
        self.clean_old = st.checkbox("🧹 Clean old CSVs before fetching?", value=True)
        self.auto_refresh = st.checkbox("🔄 Auto-refresh every 5 minutes?", value=False)

    @st.cache_data(ttl=3600)
    def _download(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Download stock data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with OHLCV data or None if download fails
        """
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

    def render_dashboard(self) -> None:
        """Main method to render the complete dashboard."""
        try:
            st.set_page_config(page_title="Stock Data Dashboard", layout="centered")
            st.title("📈 Stock OHLCV Data Downloader and Trainer")
            
            self._render_inputs()
            self._handle_user_actions()
            
        except Exception as e:
            logger.error(f"Error rendering dashboard: {e}")
            st.error("An unexpected error occurred while rendering the dashboard")
