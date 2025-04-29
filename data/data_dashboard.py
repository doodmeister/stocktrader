"""Stock Data Dashboard Module

A Streamlit dashboard for fetching, displaying, and managing historical OHLCV data.
Integrates with the E*Trade bot's data pipeline for historical price analysis.

Features:
- Fetch and validate stock data from Yahoo Finance
- Interactive data visualization 
- CSV/ZIP export functionality
- Auto-refresh capability for live monitoring
- Data cleanup management
"""

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf
from stocktrader.data.data_loader import save_to_csv
from stocktrader.core.notifier import Notifier
from stocktrader.utils.validation import sanitize_input
from stocktrader.utils.io import create_zip_archive
from stocktrader.train.training_pipeline import run_training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SYMBOLS = "AAPL,MSFT"
DATA_DIR = Path("data")
VALID_INTERVALS = ["1d", "1h", "30m", "15m", "5m", "1m"]
MAX_INTRADAY_DAYS = 60
REFRESH_INTERVAL = 300  # 5 minutes in seconds

class DataDashboard:
    """Manages the stock data dashboard functionality."""
    
    def __init__(self):
        """Initialize dashboard with default settings and ensure data directory exists."""
        self.data_dir = DATA_DIR
        self.data_dir.mkdir(exist_ok=True)
        self.saved_paths = []
        self.notifier = Notifier()
        
    @staticmethod
    def validate_dates(start_date: date, end_date: date, interval: str) -> Tuple[bool, str]:
        """Validate date range and interval compatibility."""
        if start_date > end_date:
            return False, "Start date must be before end date"
        
        if interval != "1d":
            delta = end_date - start_date
            if delta.days > MAX_INTRADAY_DAYS:
                return False, f"Intraday data limited to {MAX_INTRADAY_DAYS} days"
        
        if end_date > date.today():
            return False, "End date cannot be in the future"
            
        return True, ""

    @st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
    def fetch_ohlcv(self, symbol: str, start: str, end: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with caching and error handling."""
        try:
            data = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False
            )
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
                
            return data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def clean_data_directory(self) -> None:
        """Clean old CSV files from data directory."""
        try:
            for file in self.data_dir.glob("*.csv"):
                file.unlink()
            logger.info("Cleaned old CSV files")
        except Exception as e:
            logger.error(f"Error cleaning data directory: {e}")
            raise

    def save_data(self, df: pd.DataFrame, symbol: str, interval: str) -> Optional[Path]:
        """Save DataFrame to CSV and track the path."""
        try:
            path = self.data_dir / f"{symbol}_{interval}.csv"
            save_to_csv(df, str(path))
            self.saved_paths.append(path)
            logger.info(f"Saved data to {path}")
            return path
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return None

    def render_dashboard(self):
        """Render the Streamlit dashboard interface."""
        st.set_page_config(page_title="Stock Data Dashboard", layout="centered")
        st.title("📈 Stock OHLCV Data Downloader")

        # User inputs
        st.info("Note: Intraday intervals are limited to 60 days.")
        symbols_input = st.text_input(
            "Ticker Symbols (comma-separated)",
            value=DEFAULT_SYMBOLS
        )
        interval = st.selectbox("Data Interval", options=VALID_INTERVALS, index=0)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=date.today() - timedelta(days=365)
            )
        with col2:
            end_date = st.date_input("End Date", value=date.today())

        # Options
        clean_old = st.checkbox("🧹 Clean old CSVs before fetching?", value=True)
        auto_refresh = st.checkbox("🔄 Auto-refresh every 5 minutes?", value=False)

        # Process data on button click
        if st.button("Fetch Data"):
            self.process_data_request(
                symbols_input, start_date, end_date, interval, clean_old
            )

        # Handle auto-refresh
        if auto_refresh:
            st.caption("🔄 Auto-refresh enabled. Refreshing every 5 minutes...")
            time.sleep(REFRESH_INTERVAL)
            st.experimental_rerun()

    def process_data_request(
        self,
        symbols_input: str,
        start_date: date,
        end_date: date,
        interval: str,
        clean_old: bool
    ) -> None:
        """Process the data fetching request with validation and error handling."""
        # Input validation
        symbols = [
            sanitize_input(s.strip().upper())
            for s in symbols_input.split(",")
            if s.strip()
        ]
        
        if not symbols:
            st.error("Please enter at least one valid symbol")
            return

        # Validate dates
        valid, message = self.validate_dates(start_date, end_date, interval)
        if not valid:
            st.error(message)
            return

        # Clean old files if requested
        if clean_old:
            self.clean_data_directory()
            st.success("Old CSVs cleaned from data directory.")

        # Process each symbol
        for symbol in symbols:
            self.process_symbol(symbol, start_date, end_date, interval)

        # Create ZIP download for multiple files
        if len(self.saved_paths) > 1:
            self.create_zip_download()
        
        # Add training trigger
        train_models = st.checkbox("📚 Train models after fetching?", value=False)

        if train_models:
            st.subheader("📚 Training Models")
            for path in self.saved_paths:
                 # Extract symbol and interval from filename
                filename = path.stem  # Example: AAPL_1d
                symbol, interval = filename.split("_")
                    
                with st.spinner(f"Training model for {symbol} [{interval}]..."):
                     try:
                        run_training(symbol=symbol, interval=interval)
                        st.success(f"Model trained and saved for {symbol} [{interval}]!")
                    except Exception as e:
                        logger.error(f"Training failed for {symbol}: {e}")
                        st.error(f"Failed to train model for {symbol}")

    def process_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str
    ) -> None:
        """Process individual symbol data fetching and display."""
        if not self.is_valid_symbol(symbol):
            st.warning(f"{symbol} is not a valid stock symbol.")
            return

        st.subheader(f"📊 Data for {symbol}")
        with st.spinner(f"Fetching {symbol}..."):
            df = self.fetch_ohlcv(
                symbol,
                str(start_date),
                str(end_date),
                interval
            )
            
            if df is None:
                st.error(f"No data available for {symbol}")
                return

            self.display_data(df, symbol, interval)

    def create_zip_download(self) -> None:
        """Create and display ZIP download button for multiple files."""
        try:
            zip_data = create_zip_archive(self.saved_paths)
            st.download_button(
                label="📦 Download All as ZIP",
                data=zip_data,
                file_name="stock_data.zip",
                mime="application/zip"
            )
        except Exception as e:
            logger.error(f"Error creating ZIP archive: {e}")
            st.error("Failed to create ZIP archive")

    @staticmethod
    def display_data(df: pd.DataFrame, symbol: str, interval: str) -> None:
        """Display fetched data with visualizations and metadata."""
        st.success(f"Fetched {len(df)} rows for {symbol}")
        
        # Display recent data and chart
        st.dataframe(df.tail(10))
        st.line_chart(df['Close'])
        
        # Show metadata
        st.caption(
            f"✅ Rows: {len(df)} | "
            f"Date Range: {df.index.min().date()} → {df.index.max().date()}"
        )
        st.caption(f"✅ Latest Close: {df['Close'].iloc[-1]:.2f}")

    @staticmethod
    @st.cache_data(ttl=3600)
    def is_valid_symbol(symbol: str) -> bool:
        """Validate stock symbol with caching."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return bool(info and 'regularMarketPrice' in info)
        except Exception as e:
            logger.warning(f"Symbol validation failed for {symbol}: {e}")
            return False

def main():
    """Main entry point for the dashboard."""
    try:
        dashboard = DataDashboard()
        dashboard.render_dashboard()
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()
