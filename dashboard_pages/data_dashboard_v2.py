"""
Stock Data Download Dashboard

A streamlined Streamlit dashboard for downloading, caching, and previewing stock market data.
Supports multiple data sources with robust error handling, input validation, and performance optimization.
"""

# Standard library imports
import functools
import time
import re
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

# Dashboard logger setup
from utils.logger import get_dashboard_logger
logger = get_dashboard_logger(__name__)

# Core utilities
from core.dashboard_utils import (
    setup_page, 
    handle_streamlit_error, 
    safe_streamlit_metric, 
    create_candlestick_chart, 
    validate_ohlc_dataframe, 
    initialize_dashboard_session_state
)

# Import SessionManager to solve button key conflicts and session state issues
from core.session_manager import create_session_manager, show_session_debug_info

# Import UI renderer for home button functionality
from core.ui_renderer import UIRenderer
from core.page_loader import PageLoader

# IO utilities
from utils.io import (
    save_dataframe_with_metadata,
    create_zip_archive,
    clean_directory,
    get_file_info as io_get_file_info,
    export_session_data,
    load_dataframe_with_validation,
    load_metadata
)

# Initialize the page (setup_page returns a logger, but we already have one)
setup_page(
    title="📈 Stock Data Download Dashboard",
    logger_name=__name__,
    sidebar_title="Configuration"
)

# Configuration class
class DashboardConfig:
    def __init__(self):
        self.DATA_DIR = Path("data")
        self.MODEL_DIR = Path("models")
        self.DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
        self.VALID_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "1d"]

# Data validator class
class DataValidator:
    def __init__(self):
        self._symbol_pattern = re.compile(r'^[A-Z0-9.-]{1,10}$')
    
    def validate_symbol(self, symbol: str) -> str:
        if not symbol:
            raise ValueError("Symbol cannot be empty")
            
        symbol = symbol.strip().upper()
        
        if not self._symbol_pattern.match(symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        if len(symbol) > 10:
            raise ValueError("Symbol too long (max 10 characters)")
            
        if not any(c.isalpha() for c in symbol):
            raise ValueError("Symbol must contain at least one letter")
        
        return symbol
    
    def validate_symbols(self, symbols_input: str) -> List[str]:
        if not symbols_input or not symbols_input.strip():
            return []
            
        raw_symbols = [s.strip() for s in symbols_input.split(',')]
        valid_symbols = []
        
        for symbol in raw_symbols:
            if symbol:
                try:
                    validated = self.validate_symbol(symbol)
                    valid_symbols.append(validated)
                except ValueError as e:
                    logger.warning(f"Invalid symbol {symbol}: {e}")
                    continue
        
        if not valid_symbols:
            raise ValueError("No valid symbols found in input")
            
        return valid_symbols
    
    def validate_dates(self, start_date: date, end_date: date, interval: str = "1d") -> bool:
        """Validate date range with interval-specific limitations"""
        # Basic date validation
        if not (start_date <= end_date and end_date <= date.today()):
            return False
        
        # Calculate the number of days in the range
        days_diff = (end_date - start_date).days
        
        # Yahoo Finance interval-specific limitations
        interval_limits = {
            "1m": 7,      # 7 days max for 1-minute data
            "5m": 60,     # 60 days max for 5-minute data
            "15m": 60,    # 60 days max for 15-minute data
            "30m": 60,    # 60 days max for 30-minute data
            "1h": 730,    # ~2 years max for hourly data
            "1d": 36500   # ~100 years (essentially unlimited) for daily data
        }
        
        max_days = interval_limits.get(interval, 730)
        
        if days_diff > max_days:
            logger.warning(f"Date range too large for {interval} interval: {days_diff} days (max: {max_days})")
            return False
            
        return True
    
    def get_max_date_range_message(self, interval: str) -> str:
        """Get user-friendly message about max date range for interval"""
        interval_messages = {
            "1m": "📅 1-minute data: Maximum 7 days",
            "5m": "📅 5-minute data: Maximum 60 days", 
            "15m": "📅 15-minute data: Maximum 60 days",
            "30m": "📅 30-minute data: Maximum 60 days",
            "1h": "📅 Hourly data: Maximum ~2 years",
            "1d": "📅 Daily data: No practical limit"
        }
        return interval_messages.get(interval, "📅 Check data provider limitations")

# Utility functions
def sanitize_filename(filename: str) -> str:
    if not filename:
        return "unnamed_file"
    
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[^\w\s._-]', '', filename)
    filename = re.sub(r'\s+', '_', filename)
    filename = re.sub(r'_+', '_', filename)
    return filename.strip('_').strip()

def download_stock_data(symbols: List[str], start_date: date, end_date: date, 
                       interval: str) -> Dict[str, pd.DataFrame]:
    """Download stock data using yfinance"""
    result = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            if not df.empty:
                df = df.reset_index()
                df.columns = [col.lower() for col in df.columns]
                if 'date' in df.columns:
                    df.rename(columns={'date': 'timestamp'}, inplace=True)
                result[symbol] = df
        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
    return result

def get_file_info(file_path: Path) -> Dict:
    """Get file information using io utility"""
    try:
        return io_get_file_info(file_path)
    except Exception:
        return {'size_mb': 0, 'modified': datetime.now()}

# Exception handling decorator
def handle_streamlit_exception(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            st.error(f"❌ Operation failed: {str(e)}")
            return None
    return wrapper

# Constants
MAX_CONCURRENT_DOWNLOADS = 5
DEFAULT_CHART_HEIGHT = 400

class DataDashboard:
    """Production-grade Streamlit dashboard for stock data management."""

    def __init__(self):
        self.config = DashboardConfig()
        self.validator = DataValidator()
        
        # Initialize SessionManager for conflict-free button handling
        self.session_manager = create_session_manager("data_dashboard_v2")
        
        self._setup_directories()
        self._init_instance_variables()
        self._init_session_state()
        
        self.start_time = time.time()
        
        logger.info("DataDashboard initialized")

    def _init_instance_variables(self) -> None:
        """Initialize instance variables"""
        self.saved_paths: List[Path] = []
        self.symbols: List[str] = []
        self.start_date: date = date.today() - timedelta(days=365)
        self.end_date: date = date.today()
        self.interval: str = "1d"
        self.clean_old: bool = True
        self.max_workers: int = min(MAX_CONCURRENT_DOWNLOADS, len(self.config.DEFAULT_SYMBOLS))

    def _init_session_state(self) -> None:
        """Initialize session state"""
        initialize_dashboard_session_state()
        
        dashboard_defaults = {
            "data_fetched": False,
            "last_fetch_time": None,
            "fetch_count": 0,
            "saved_paths": [],
            "download_progress": 0.0,
            "current_operation": "",
            "cache_hits": 0,
            "total_records": 0,
            "show_export": False,
            "show_analysis": False
        }
        
        for key, default_value in dashboard_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        if st.session_state.get('saved_paths'):
            self.saved_paths = [Path(p) for p in st.session_state['saved_paths'] 
                             if Path(p).exists()]

    def _setup_directories(self) -> None:
        """Create required directories"""
        directories = [
            self.config.DATA_DIR,
            self.config.MODEL_DIR,
            self.config.DATA_DIR / "exports"        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.critical(f"Failed to create directory {directory}: {e}")
                st.error(f"🚨 Cannot access required directory: {directory}")
                st.stop()

    def _render_performance_metrics(self) -> None:
        """Display performance metrics in sidebar"""
        with st.sidebar:
            if not self.session_manager.create_checkbox(
                "Show Performance Metrics", 
                "show_performance_metrics",
                value=False
            ):
                return
            
        with st.sidebar.expander("📊 Session Metrics", expanded=False):
            uptime = time.time() - self.start_time
            
            safe_streamlit_metric("Uptime", f"{uptime:.1f}s")
            safe_streamlit_metric("Downloads", str(st.session_state.get("fetch_count", 0)))
            safe_streamlit_metric("Cache Hits", str(st.session_state.get("cache_hits", 0)))
            safe_streamlit_metric("Total Records", str(st.session_state.get("total_records", 0)))

    def _render_inputs(self) -> None:
        """Render input controls"""
        st.subheader("📋 Data Configuration")
        
        interval_limits = {
            "1m": "7 days", "5m": "60 days", "15m": "60 days",
            "30m": "60 days", "1h": "2 years", "1d": "No limit"
        }
        
        current_limit = interval_limits.get(self.interval, "Check provider")
        st.info(
            f"💡 **Current interval ({self.interval})** - "
            f"Historical limit: {current_limit}. "
            f"Intraday data may have restrictions."
        )
        self._render_symbol_input()
        self._render_interval_selection()
        self._render_date_inputs()
        self._render_options()

    def _render_symbol_input(self) -> None:
        """Render symbol input with validation"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symbols_input = self.session_manager.create_text_input(
                "🎯 Ticker Symbols",
                value=", ".join(self.config.DEFAULT_SYMBOLS),
                text_input_name="symbol_input",
                help="Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)",
                placeholder="AAPL, MSFT, GOOGL"
            )
        with col2:
            validate_clicked = self.session_manager.create_button(
                "🔄 Validate", 
                "validate_symbols",
                help="Check symbol validity"            )
        
        # Process symbols automatically and show validation feedback
        previous_symbols = self.symbols.copy()
        try:
            self.symbols = self.validator.validate_symbols(symbols_input)
            if self.symbols != previous_symbols:
                st.session_state["data_fetched"] = False
                logger.info(f"Symbols updated: {previous_symbols} → {self.symbols}")
                
        except ValueError as e:
            st.error(f"❌ Invalid symbols: {e}")
            self.symbols = []
        
        # Provide immediate validation feedback if validate button clicked
        if validate_clicked:
            self._validate_symbols_realtime(symbols_input)

        self._display_symbol_status()

    def _validate_symbols_realtime(self, symbols_input: str) -> None:
        """Perform real-time symbol validation"""
        if not symbols_input.strip():
            st.warning("⚠️ Please enter at least one symbol")
            return
        
        with st.spinner("🔍 Validating symbols..."):
            try:
                raw_symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
                if not raw_symbols:
                    st.warning("⚠️ No valid symbols found in input")
                    return
                
                valid_symbols = []
                invalid_symbols = []
                
                for symbol in raw_symbols:
                    try:
                        validated = self.validator.validate_symbol(symbol)
                        valid_symbols.append(validated)
                    except Exception as e:
                        invalid_symbols.append(f"{symbol} ({str(e)})")
                
                if valid_symbols:
                    st.success(f"✅ **Valid symbols:** {', '.join(valid_symbols)}")
                    
                if invalid_symbols:
                    st.error(f"❌ **Invalid symbols:** {', '.join(invalid_symbols)}")
                    
            except Exception as e:
                logger.error(f"Symbol validation error: {e}")
                st.error(f"❌ Validation failed: {str(e)}")

    def _render_interval_selection(self) -> None:
        """Render interval selection with date range validation"""
        intervals_with_desc = {
            "1m": "1 Minute (High frequency, limited history)",
            "5m": "5 Minutes (Intraday trading)",
            "15m": "15 Minutes (Short-term analysis)",
            "30m": "30 Minutes (Swing trading)",
            "1h": "1 Hour (Medium-term analysis)",
            "1d": "1 Day (Long-term analysis)"        }
        
        current_index = 0
        if self.interval in self.config.VALID_INTERVALS:
            current_index = self.config.VALID_INTERVALS.index(self.interval)
        
        previous_interval = self.interval
        self.interval = self.session_manager.create_selectbox(
            "📊 Data Interval",
            options=self.config.VALID_INTERVALS,
            index=current_index,
            selectbox_name="data_interval",            format_func=lambda x: intervals_with_desc.get(x, x),
            help="Choose the frequency of data points"
        )
        
        # Check if interval changed and warn about date range compatibility
        if previous_interval != self.interval:
            st.session_state["data_fetched"] = False
            # Check if current date range is compatible with new interval
            if not self.validator.validate_dates(self.start_date, self.end_date, self.interval):
                days_span = (self.end_date - self.start_date).days
                interval_limits = {
                    "1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730, "1d": 36500
                }
                max_days = interval_limits.get(self.interval, 730)
                st.warning(f"⚠️ Current date range ({days_span} days) is too large for **{self.interval}** interval (max: {max_days} days)")
                st.info("🔧 Please adjust your date range below or the download will fail.")

    def _render_date_inputs(self) -> None:
        """Render date inputs with presets"""
        st.subheader("📅 Date Range")
        
        # Quick preset buttons
        col1, col2, col3, col4 = st.columns(4)
        today = date.today()
        
        with col1:
            if self.session_manager.create_button("📅 1 Week", "preset_week_btn", help="Last 7 days"):
                st.session_state["preset_start_date"] = today - timedelta(days=7)
                st.session_state["preset_end_date"] = today
                st.session_state["data_fetched"] = False
                st.success("✅ Date range set to last 7 days")
                st.rerun()
            
        with col2:
            if self.session_manager.create_button("📅 1 Month", "preset_month_btn", help="Last 30 days"):
                st.session_state["preset_start_date"] = today - timedelta(days=30)
                st.session_state["preset_end_date"] = today
                st.session_state["data_fetched"] = False
                st.success("✅ Date range set to last 30 days")
                st.rerun()
            
        with col3:
            if self.session_manager.create_button("📅 3 Months", "preset_3month_btn", help="Last 90 days"):
                st.session_state["preset_start_date"] = today - timedelta(days=90)
                st.session_state["preset_end_date"] = today
                st.session_state["data_fetched"] = False
                st.success("✅ Date range set to last 90 days")
                st.rerun()
            
        with col4:
            if self.session_manager.create_button("📅 1 Year", "preset_year_btn", help="Last 365 days"):
                st.session_state["preset_start_date"] = today - timedelta(days=365)
                st.session_state["preset_end_date"] = today
                st.session_state["data_fetched"] = False
                st.success("✅ Date range set to last 365 days")
                st.rerun()
        
        # Use session state values if available, otherwise use instance values
        current_start = st.session_state.get("preset_start_date", self.start_date)
        current_end = st.session_state.get("preset_end_date", self.end_date)
        
        # Manual date selection
        col1, col2 = st.columns(2)
        with col1:
            new_start = self.session_manager.create_date_input(
                "Start Date",
                value=current_start,
                max_value=today,
                help="Beginning of data range",
                date_input_name="start_date_input"
            )
            
        with col2:
            new_end = self.session_manager.create_date_input(
                "End Date",
                value=current_end,
                max_value=today,
                help="End of data range",
                date_input_name="end_date_input"
            )

        # Update dates and clear preset values
        if new_start != self.start_date or new_end != self.end_date:
            self.start_date = new_start
            self.end_date = new_end
            st.session_state["data_fetched"] = False
            # Clear preset values once they're applied
            if "preset_start_date" in st.session_state:
                del st.session_state["preset_start_date"]
            if "preset_end_date" in st.session_state:
                del st.session_state["preset_end_date"]
          # Validation with interval-specific limitations
        basic_valid = self.start_date <= self.end_date and self.end_date <= date.today()
        interval_valid = self.validator.validate_dates(self.start_date, self.end_date, self.interval)
        
        if not basic_valid:
            st.error("❌ Invalid date range: Start date must be ≤ End date and not in the future")
        elif not interval_valid:
            days_span = (self.end_date - self.start_date).days
            interval_limits = {
                "1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730, "1d": 36500
            }
            max_days = interval_limits.get(self.interval, 730)
            
            # Show warning with automatic adjustment option
            st.warning(f"⚠️ Date range too large for **{self.interval}** interval: {days_span} days (max: {max_days} days)")
            st.info(self.validator.get_max_date_range_message(self.interval))
            
            # Offer automatic adjustment
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🔧 Auto-adjust range", help=f"Automatically adjust to last {max_days} days"):
                    new_start = today - timedelta(days=max_days)
                    st.session_state["preset_start_date"] = new_start
                    st.session_state["preset_end_date"] = today
                    st.session_state["data_fetched"] = False
                    st.success(f"✅ Date range adjusted to last {max_days} days for {self.interval} interval")
                    st.rerun()
            with col2:
                st.caption("💡 Consider using a daily (1d) interval for longer time periods")
        else:
            days_span = (self.end_date - self.start_date).days
            st.success(f"✅ Valid range: {days_span} days ({self.start_date} to {self.end_date})")
            
            # Show helpful interval info for valid ranges
            if self.interval in ["1m", "5m", "15m", "30m"]:
                st.info(self.validator.get_max_date_range_message(self.interval))

    def _render_options(self) -> None:
        """Render additional options"""
        st.subheader("⚙️ Options")
        col1, col2 = st.columns(2)
        
        with col1:
            self.clean_old = self.session_manager.create_checkbox(
                "🧹 Clean old files",
                "clean_old_files",
                value=self.clean_old,
                help="Remove existing CSV files before downloading new data"
            )
            
        with col2:
            self.max_workers = self.session_manager.create_slider(
                "🚀 Concurrent Downloads",
                min_value=1,
                max_value=10,
                value=self.max_workers,
                slider_name="max_workers",
                help="Number of parallel download threads"
            )

    @handle_streamlit_exception
    def _clean_existing_files(self) -> int:
        """Clean existing CSV files using io utility"""
        if not self.symbols:
            return 0
        
        try:
            # Build patterns for symbols
            patterns = []
            for symbol in self.symbols:
                safe_symbol = sanitize_filename(symbol)
                patterns.append(f"{safe_symbol}_*.csv")
                patterns.append(f"{safe_symbol}_*.meta.json")
            
            total_removed = 0
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, pattern in enumerate(patterns):
                try:
                    removed_count, _ = clean_directory(
                        directory=self.config.DATA_DIR,
                        pattern=pattern,
                        dry_run=False
                    )
                    total_removed += removed_count
                except Exception as e:
                    logger.error(f"Error cleaning files with pattern {pattern}: {e}")
                
                progress = (i + 1) / len(patterns)
                progress_bar.progress(progress)
                status_text.text(f"Cleaning files... {i + 1}/{len(patterns)}")
            
            progress_bar.empty()
            status_text.empty()
            
            if total_removed > 0:
                st.success(f"🧹 Cleaned {total_removed} existing file(s)")
            
            st.session_state["data_fetched"] = False
            return total_removed
            
        except Exception as e:
            logger.error(f"Error during file cleanup: {e}")
            return 0

    def _download_with_progress(
        self, 
        symbols: List[str], 
        start_date: date, 
        end_date: date, 
        interval: str
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Download data with progress tracking"""
        if not symbols:
            return None
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            total_symbols = len(symbols)
            completed_symbols = 0
            results = {}
            errors = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_symbol = {
                    executor.submit(
                        download_stock_data,
                        [symbol],
                        start_date,
                        end_date,
                        interval
                    ): symbol for symbol in symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    
                    try:
                        data = future.result(timeout=60)
                        if data and symbol in data:
                            results[symbol] = data[symbol]
                            logger.info(f"Successfully downloaded data for {symbol}")
                        else:
                            errors.append(f"No data returned for {symbol}")
                            
                    except Exception as e:
                        error_msg = f"Failed to download {symbol}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                    
                    completed_symbols += 1
                    progress = completed_symbols / total_symbols
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Downloading... {completed_symbols}/{total_symbols} "
                        f"({len(results)} successful, {len(errors)} errors)"
                    )
            
            progress_bar.empty()
            status_text.empty()
            
            if results:
                st.success(f"✅ Successfully downloaded {len(results)}/{total_symbols} symbols")
                
            if errors:
                st.warning(f"⚠️ {len(errors)} download(s) failed")
                with st.expander("Error Details", expanded=False):
                    for error in errors:
                        st.text(error)
            
            return results if results else None
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            logger.exception(f"Critical error during download: {e}")
            return None

    def _save_data_with_metadata(
        self, 
        data_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[List[Path], int]:
        """Save data with metadata using io utility"""
        if not data_dict:
            return [], 0
        
        saved_paths = []
        successful_saves = 0
        total_records = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (symbol, df) in enumerate(data_dict.items()):
            try:
                safe_symbol = sanitize_filename(symbol)
                file_path = self.config.DATA_DIR / f"{safe_symbol}_{self.interval}.csv"
                
                # Create metadata
                metadata = {
                    'symbol': symbol,
                    'interval': self.interval,
                    'start_date': self.start_date.isoformat(),
                    'end_date': self.end_date.isoformat(),
                    'records': len(df),
                    'created_at': datetime.now().isoformat(),
                    'dashboard_version': '2.0'
                }
                
                # Save using io utility
                success, message, backup_path = save_dataframe_with_metadata(
                    df=df,
                    file_path=file_path,
                    metadata=metadata,
                    create_backup=True
                )
                
                if success:
                    saved_paths.append(file_path)
                    successful_saves += 1
                    total_records += len(df)
                    logger.info(f"Saved {len(df)} records for {symbol}: {message}")
                else:
                    logger.error(f"Failed to save {symbol}: {message}")
                    st.error(f"⚠️ Failed to save {symbol}: {message}")
                
            except Exception as e:
                logger.error(f"Failed to save data for {symbol}: {e}")
                st.error(f"⚠️ Failed to save {symbol}: {e}")
            
            progress = (i + 1) / len(data_dict)
            progress_bar.progress(progress)
            status_text.text(f"Saving... {i + 1}/{len(data_dict)}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Update session state
        st.session_state['saved_paths'] = [str(p) for p in saved_paths]
        st.session_state['total_records'] = total_records
        
        return saved_paths, successful_saves    @handle_streamlit_exception
    def _fetch_and_display_data(self) -> int:
        """Main data fetching workflow"""
        if not self.symbols:
            st.error("❌ Please enter at least one valid stock symbol")
            return 0
            
        if not self.validator.validate_dates(self.start_date, self.end_date, self.interval):
            # Show specific error message for interval-related issues
            days_span = (self.end_date - self.start_date).days
            interval_limits = {
                "1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730, "1d": 36500
            }
            max_days = interval_limits.get(self.interval, 730)
            
            if days_span > max_days:
                st.error(f"❌ Date range too large for {self.interval} interval: {days_span} days (max: {max_days} days)")
                st.info(f"💡 {self.validator.get_max_date_range_message(self.interval)}")
            else:
                st.error("❌ Invalid date range selected")
            return 0
        
        st.info("🚀 Starting data download process...")
        
        try:
            # Clean existing files if requested
            if self.clean_old:
                with st.spinner("🧹 Cleaning existing files..."):
                    cleaned_count = self._clean_existing_files()
                    if cleaned_count and cleaned_count > 0:
                        time.sleep(1)
            
            # Download data
            with st.spinner("📥 Downloading market data..."):
                st.session_state["current_operation"] = "Downloading"
                raw_data = self._download_with_progress(
                    self.symbols, self.start_date, self.end_date, self.interval
                )
                
                if not raw_data:
                    st.error("❌ No data was successfully downloaded")
                    return 0
            
            # Save data
            with st.spinner("💾 Saving data files..."):
                st.session_state["current_operation"] = "Saving"
                saved_paths, successful_saves = self._save_data_with_metadata(raw_data)
                self.saved_paths = saved_paths
            
            # Update session state
            if successful_saves > 0:
                st.session_state.update({
                    "data_fetched": True,
                    "last_fetch_time": datetime.now(),
                    "fetch_count": st.session_state.get("fetch_count", 0) + 1,
                    "current_operation": "Complete"
                })
                
                total_records = sum(len(df) for df in raw_data.values())
                st.success(f"🎉 Downloaded {successful_saves} symbols ({total_records:,} total records)")
        
            # Display data
            for symbol, df in raw_data.items():
                self._display_symbol_data(symbol, df)
            
            return successful_saves
            
        except Exception as e:
            logger.exception(f"Unexpected error in data fetch workflow: {e}")
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            return 0
        finally:
            st.session_state["current_operation"] = ""

    def _display_symbol_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Display data for a single symbol"""
        st.subheader(f"📊 {symbol}")
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            safe_streamlit_metric("Records", f"{len(df):,}")
        with col2:
            safe_streamlit_metric("Columns", str(len(df.columns)))
        with col3:
            if 'timestamp' in df.columns:
                date_range_str = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
                safe_streamlit_metric("Date Range", date_range_str)
        with col4:
            if 'close' in df.columns:
                safe_streamlit_metric("Latest Close", f"${df['close'].iloc[-1]:.2f}")
    
        # Data preview
        with st.expander(f"📋 Data Preview - {symbol}", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

    def _display_cached_data(self) -> None:
        """Display cached data using io utilities"""
        if not self.saved_paths:
            return
        
        st.info("📋 Displaying cached data from previous session")
        
        for path in self.saved_paths:
            if not path.exists():
                continue
            
            symbol = path.stem.split('_')[0]
            
            try:
                # Load data using io utility
                df, load_message = load_dataframe_with_validation(
                    file_path=path, 
                    parse_dates=["timestamp"]
                )
                
                if df is None:
                    st.error(f"❌ Failed to load {symbol}: {load_message}")
                    continue
                
                # Load metadata using io utility
                metadata, meta_message = load_metadata(path.with_suffix('.meta.json'))
                
                if metadata:
                    st.caption(
                        f"📊 {symbol}: {metadata.get('records', 'Unknown')} records "
                        f"from {metadata.get('start_date', 'Unknown')} to {metadata.get('end_date', 'Unknown')}"
                    )
                
                st.session_state["cache_hits"] = st.session_state.get("cache_hits", 0) + 1
                self._display_symbol_data(symbol, df)
                
            except Exception as e:
                st.error(f"❌ Failed to load {symbol}: {e}")

    def _display_symbol_status(self) -> None:
        """Display symbol status"""
        if not self.symbols:
            st.caption("⚪ No symbols selected")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if self.symbols:
                st.success(f"**Active symbols:** {', '.join(self.symbols)}")
            
            st.caption(f"📊 {len(self.symbols)} symbol(s) selected")
            
            # Estimate data points
            days_span = (self.end_date - self.start_date).days
            interval_multipliers = {
                "1m": days_span * 1440,
                "5m": days_span * 288,   
                "15m": days_span * 96,
                "30m": days_span * 48,
                "1h": days_span * 24,
                "1d": days_span
            }
            
            estimated_points = interval_multipliers.get(self.interval, days_span) * len(self.symbols)
            st.caption(f"📈 Estimated data points: ~{estimated_points:,}")
        
        with col2:
            if self.symbols:
                # Size estimate
                avg_bytes_per_record = 100
                estimated_size_bytes = estimated_points * avg_bytes_per_record
                
                if estimated_size_bytes < 1024:
                    size_display = f"{estimated_size_bytes} B"
                elif estimated_size_bytes < 1024 * 1024:
                    size_display = f"{estimated_size_bytes / 1024:.1f} KB"
                else:
                    size_display = f"{estimated_size_bytes / (1024 * 1024):.1f} MB"
                
                st.metric("Est. Download Size", size_display)
                
                # Time estimate
                est_time_seconds = len(self.symbols) * 2
                if est_time_seconds < 60:
                    time_display = f"{est_time_seconds}s"
                else:
                    time_display = f"{est_time_seconds // 60}m {est_time_seconds % 60}s"
                
                st.metric("Est. Download Time", time_display)
    
    def _show_export_interface(self):
        """Export interface using io utilities"""
        st.subheader("📁 Export Downloaded Data")
        if self.session_manager.create_button("❌ Close Export", "close_export"):
            st.session_state["show_export"] = False
            st.rerun()
            return
        
        # Get all CSV files in the data directory
        all_csv_files = list(self.config.DATA_DIR.glob("*.csv"))
        
        if not all_csv_files:
            st.warning("⚠️ No data files available for export")
            return
        
        # Show available files
        with st.expander("📋 Available Files", expanded=False):
            st.write(f"**Found {len(all_csv_files)} CSV files in data directory:**")
            
            # Group files by symbol for better display
            symbol_groups = {}
            for file_path in all_csv_files:
                try:
                    symbol = file_path.stem.split('_')[0]
                    if symbol not in symbol_groups:
                        symbol_groups[symbol] = []
                    symbol_groups[symbol].append(file_path)
                except IndexError:
                    # Handle files that don't follow expected naming pattern
                    symbol_groups.setdefault('Other', []).append(file_path)
            
            for symbol, files in symbol_groups.items():
                st.write(f"**{symbol}:**")
                for file_path in files:
                    file_info = get_file_info(file_path)
                    meta_path = file_path.with_suffix('.meta.json')
                    meta_indicator = " 📋" if meta_path.exists() else ""                    # Get file modification time
                    try:
                        mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        timestamp_str = mod_time.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        timestamp_str = "Unknown"
                    st.write(f"  • {file_path.name} ({file_info.get('size_mb', 'Unknown')} MB) - {timestamp_str}{meta_indicator}")
        
        # File selection options
        st.subheader("📂 Select Files to Export")
        
        export_option = self.session_manager.create_radio(
            "Choose which files to export:",
            options=["All files", "Current session only", "Select specific files"],
            radio_name="export_option"
        )
        
        files_to_export = []
        if export_option == "All files":
            files_to_export = all_csv_files
            st.info(f"ℹ️ Will export all {len(files_to_export)} CSV files")
            
        elif export_option == "Current session only":
            files_to_export = [p for p in self.saved_paths if p.exists()]
            if files_to_export:
                st.info(f"ℹ️ Will export {len(files_to_export)} files from current session")
            else:
                st.warning("⚠️ No files from current session available")
                return
                
        elif export_option == "Select specific files":
            st.write("**Select files to include in export:**")
            
            selected_files = []
            # Group by symbol for organized selection
            for symbol, files in symbol_groups.items():
                with st.expander(f"{symbol} ({len(files)} files)", expanded=True):
                    for file_path in files:
                        file_info = get_file_info(file_path)
                        if self.session_manager.create_checkbox(
                            f"{file_path.name} ({file_info.get('size_mb', 'Unknown')} MB)",
                            f"select_{file_path.stem}"
                        ):
                            selected_files.append(file_path)
            
            files_to_export = selected_files
            
            if not files_to_export:
                st.warning("⚠️ No files selected for export")
                return
            else:
                st.info(f"ℹ️ Selected {len(files_to_export)} files for export")
        
        # Export format and options (same as before)
        export_format = self.session_manager.create_selectbox(
            "Choose export format:",
            options=["CSV (Individual Files)", "CSV (Combined)", "JSON"],
            selectbox_name="export_format_select"
        )
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            include_metadata = self.session_manager.create_checkbox("Include metadata", "include_metadata", value=True)
        with col2:
            compress_files = self.session_manager.create_checkbox("Compress output", "compress_files", value=False)
        
        # Format descriptions
        format_descriptions = {
            "CSV (Individual Files)": "📦 ZIP archive with separate CSV files for each symbol",
            "CSV (Combined)": "📄 Single CSV file with all symbols combined",
            "JSON": "🔗 JSON format suitable for APIs and web applications"
        }
        
        st.info(f"ℹ️ {format_descriptions.get(export_format, 'Selected export format')}")
          # Export button
        if self.session_manager.create_button("📥 Create Export", "create_export", type="primary", help="Create and download export package"):
            try:
                with st.spinner("Creating export package..."):
                    # Temporarily update saved_paths for export methods
                    original_saved_paths = self.saved_paths
                    self.saved_paths = files_to_export
                    
                    export_path = self._create_export_package(
                        export_format, include_metadata, compress_files
                    )
                    
                    # Restore original saved_paths
                    self.saved_paths = original_saved_paths
                    
                    if export_path and export_path.exists():
                        file_info = get_file_info(export_path)
                        
                        st.success(f"✅ Export created: {export_path.name}")
                        
                        # Download button
                        with open(export_path, 'rb') as f:
                            file_data = f.read()
                        
                        st.download_button(
                            label=f"📁 Download {export_path.name} ({file_info.get('size_mb', 'Unknown')} MB)",
                            data=file_data,
                            file_name=export_path.name,
                            mime=self._get_mime_type(export_format),
                            key="download_export"
                        )
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("File Size", f"{file_info.get('size_mb', 'Unknown')} MB")
                        with col2:
                            st.metric("Files Exported", str(len(files_to_export)))
                        with col3:
                            st.metric("Format", export_format.split(' ')[0])
                            
                    else:
                        st.error("❌ Failed to create export file")
                        
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
                logger.error(f"Export error: {e}")

    def _create_export_package(
        self, 
        export_format: str, 
        include_metadata: bool, 
        compress_files: bool
    ) -> Optional[Path]:
        """Create export package using io utilities"""
        
        if not self.saved_paths:
            raise ValueError("No data files to export")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.config.DATA_DIR / "exports"
        export_dir.mkdir(exist_ok=True)
        
        try:
            if export_format == "CSV (Individual Files)":
                return self._export_individual_csv(export_dir, timestamp, include_metadata)
            elif export_format == "CSV (Combined)":
                return self._export_combined_csv(export_dir, timestamp, compress_files)
            elif export_format == "JSON":
                return self._export_json(export_dir, timestamp, compress_files)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            logger.error(f"Export package creation failed: {e}")
            raise

    def _export_individual_csv(
        self, export_dir: Path, timestamp: str, include_metadata: bool
    ) -> Path:
        """Export individual CSV files in ZIP using io utility"""
        
        # Collect files to include
        files_to_zip = []
        
        for file_path in self.saved_paths:
            if file_path.exists():
                files_to_zip.append(file_path)
                
                if include_metadata:
                    meta_path = file_path.with_suffix('.meta.json')
                    if meta_path.exists():
                        files_to_zip.append(meta_path)
        
        if not files_to_zip:
            raise ValueError("No files found to export")
        
        # Create ZIP using io utility
        zip_data = create_zip_archive(files_to_zip)
        
        # Save ZIP file
        zip_path = export_dir / f"stock_data_csv_{timestamp}.zip"
        with open(zip_path, 'wb') as f:
            f.write(zip_data)
        
        logger.info(f"Created ZIP export with {len(files_to_zip)} files: {zip_path}")
        return zip_path

    def _export_combined_csv(self, export_dir: Path, timestamp: str, compress_files: bool) -> Path:
        """Export combined CSV file using io utility"""
        combined_data = []
        
        for file_path in self.saved_paths:
            if file_path.exists():
                try:
                    df, message = load_dataframe_with_validation(
                        file_path=file_path, 
                        parse_dates=["timestamp"]
                    )
                    
                    if df is not None:
                        symbol = file_path.stem.split('_')[0]
                        df['symbol'] = symbol
                        combined_data.append(df)
                    else:
                        logger.warning(f"Failed to load {file_path}: {message}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
        
        if not combined_data:
            raise ValueError("No valid data files found")
        
        # Combine all data
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df = combined_df.sort_values(['symbol', 'timestamp'])
        
        # Save using io utility
        filename = f"combined_stock_data_{timestamp}.csv"
        if compress_files:
            filename += ".gz"
        
        export_path = export_dir / filename
        
        # Use pandas compression for CSV
        if compress_files:
            combined_df.to_csv(export_path, index=False, compression='gzip')
        else:
            combined_df.to_csv(export_path, index=False)
        
        logger.info(f"Created combined CSV export: {export_path}")
        return export_path

    def _export_json(self, export_dir: Path, timestamp: str, compress_files: bool) -> Path:
        """Export as JSON using io utility"""
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'symbols': [],
                'total_records': 0,
                'dashboard_version': '2.0'
            },
            'data': {}
        }
        
        total_records = 0
        
        for file_path in self.saved_paths:
            if file_path.exists():
                try:
                    df, message = load_dataframe_with_validation(
                        file_path=file_path, 
                        parse_dates=["timestamp"]
                    )
                    
                    if df is not None:
                        symbol = file_path.stem.split('_')[0]
                        
                        symbol_data = df.to_dict('records')
                        
                        # Convert timestamps to ISO format
                        for record in symbol_data:
                            if 'timestamp' in record and pd.notna(record['timestamp']):
                                record['timestamp'] = pd.to_datetime(record['timestamp']).isoformat()
                        
                        export_data['data'][symbol] = symbol_data
                        export_data['export_info']['symbols'].append(symbol)
                        total_records += len(symbol_data)
                    else:
                        logger.warning(f"Failed to process {file_path} for JSON export: {message}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process {file_path} for JSON export: {e}")
                    continue
        
        export_data['export_info']['total_records'] = total_records
        
        # Save using io utility
        filename = f"stock_data_{timestamp}.json"
        export_path = export_dir / filename
        
        if compress_files:
            # For JSON compression, we'll handle it manually
            import gzip
            filename += ".gz"
            export_path = export_dir / filename
            
            json_content = json.dumps(export_data, indent=2, default=str)
            with gzip.open(export_path, 'wt', encoding='utf-8') as f:
                f.write(json_content)
        else:
            # Use io utility for regular JSON
            from utils.io import save_dataframe
                  # Convert to temporary dataframe for saving, or handle manually
            json_content = json.dumps(export_data, indent=2, default=str)
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(json_content)
        
        logger.info(f"Created JSON export: {export_path}")
        return export_path
    
    def _get_mime_type(self, export_format: str) -> str:
        """Get MIME type for export format"""
        mime_types = {
            "CSV (Individual Files)": "application/zip",
            "CSV (Combined)": "text/csv",
            "JSON": "application/json"        }
        return mime_types.get(export_format, "application/octet-stream")

    def run(self) -> None:
        """Main dashboard application"""
        # Render header with home button for navigation consistency
        try:
            page_loader = PageLoader(logger)
            pages_config = page_loader.load_pages_configuration()
            ui_renderer = UIRenderer()
            ui_renderer.render_header(pages_config, None)
        except Exception as e:
            logger.warning(f"Could not render navigation header: {e}")
            # Fallback home button
            if st.button("🏠 Home", help="Return to main dashboard"):
                st.session_state.current_page = 'home'
                st.session_state.page_history = ['home']
                st.rerun()
        
        st.markdown(
            "**Enterprise-grade data acquisition** • "
            "Download, validate, and analyze market data with advanced features"
        )
        
        self._render_performance_metrics()
        
        # Main tabs
        tabs = st.tabs(["📥 Data Download", "🔍 Analysis Hub", "⚙️ Settings"])
        
        with tabs[0]:  # Data Download Tab
            self._render_inputs()
              # Action buttons
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                fetch_clicked = self.session_manager.create_button(
                    "🚀 Download Data",
                    "download_data",
                    type="primary",
                    use_container_width=True,
                    help="Start the data download process"
                )
            
            with col2:
                if self.session_manager.create_button("🧹 Clear Cache", "clear_cache", help="Clear all cached data"):
                    try:
                        # Clear session state
                        for key in ["data_fetched", "saved_paths", "cache_hits", "total_records"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        # Clean files using io utility
                        cleaned_count, _ = clean_directory(
                            directory=self.config.DATA_DIR,
                            pattern="*.csv",
                            dry_run=False
                        )
                        meta_cleaned, _ = clean_directory(
                            directory=self.config.DATA_DIR,
                            pattern="*.meta.json",
                            dry_run=False                        )
                        
                        total_cleaned = cleaned_count + meta_cleaned
                        st.success(f"✅ Cache cleared - {total_cleaned} files removed")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Failed to clear cache: {e}")
            
            with col3:
                if self.session_manager.create_button("🔄 Refresh", "refresh_view", help="Refresh the current view"):
                    st.rerun()
                st.info("💡 Use refresh to update data display after changes")

            st.divider()
        
            # Main content
            if fetch_clicked:
                successful_downloads = self._fetch_and_display_data()
                # if successful_downloads > 0:
                #     st.balloons()
            else:
                if st.session_state.get("data_fetched") and self.saved_paths:
                    self._display_cached_data()
                else:
                    st.info(
                        "👋 **Welcome!** Select your symbols and date range above, "
                        "then click **Download Data** to get started."
                    )
        
        with tabs[1]:  # Analysis Hub            st.header("🔍 Data Analysis Hub")
            st.info(
                "🚀 **Advanced Analysis** • "
                "Visit the **Data Analysis** page for pattern recognition, "
                "technical indicators, and **Strategy Backtesting**."
            )
            
            if self.saved_paths:
                col1, col2 = st.columns(2)
                with col1:
                    if self.session_manager.create_button("📁 Export Data", "export_data", type="primary", help="Export downloaded data"):
                        st.session_state["show_export"] = True
                        st.rerun()
                
                with col2:
                    st.markdown("**🔗 Quick Links**")
                    st.markdown("- 📊 [Data Analysis Page](data_analysis_v2)")
                    st.markdown("- 🤖 [Model Training Page](model_training)")
                    st.markdown("- 📈 [Strategy Testing](backtesting)")
                
                if st.session_state.get("show_export", False):
                    self._show_export_interface()
                    
            else:
                st.info("No data available for analysis. Download some data first!")
                
            # Quick overview of analysis features
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **📊 Technical Analysis**
                - Candlestick Patterns
                - Technical Indicators
                - Price Action Analysis
                """)
            
            with col2:
                st.markdown("""
                **🤖 Machine Learning**
                - Pattern Neural Networks
                - Feature Engineering
                - Model Training & Validation
                """)
            
            with col3:
                st.markdown("""                **📈 Strategy Testing**
                - Historical Backtesting
                - Performance Metrics
                - Risk Analysis
                """)
        
        with tabs[2]:  # Settings
            st.header("⚙️ Dashboard Settings")
            st.subheader("💾 Data Management")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🗑️ Delete All Data")
                st.warning("⚠️ This will permanently delete all downloaded data files")
                
                confirm_delete = self.session_manager.create_checkbox(
                    "I understand this will delete all data", 
                    "confirm_delete_all"
                )
                if confirm_delete:
                    if self.session_manager.create_button("🗑️ Delete Now", "delete_all_data", type="secondary", help="Permanently delete all data"):
                        try:
                            # Clean using io utility
                            csv_cleaned, _ = clean_directory(
                                directory=self.config.DATA_DIR,
                                pattern="*.csv",
                                dry_run=False
                            )
                            meta_cleaned, _ = clean_directory(
                                directory=self.config.DATA_DIR,
                                pattern="*.meta.json",
                                dry_run=False
                            )
                            
                            total_deleted = csv_cleaned + meta_cleaned
                            st.success(f"✅ Deleted {total_deleted} files")
                            st.session_state["data_fetched"] = False
                            st.session_state["saved_paths"] = []
                            self.saved_paths.clear()
                            
                            # Clear the confirmation checkbox
                            st.session_state["confirm_delete"] = False
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Failed to delete data: {e}")
                else:
                    st.info("Check the box above to enable deletion")
            
            with col2:
                if self.session_manager.create_button("📊 Export Session Info", "export_session_info", help="Download session information"):
                    try:
                        # Use io utility for session export
                        session_json = export_session_data(
                            symbols=self.symbols,
                            date_range={
                                'start': self.start_date.isoformat(),
                                'end': self.end_date.isoformat()
                            },
                            interval=self.interval,
                            session_stats={
                                'fetch_count': st.session_state.get('fetch_count', 0),
                                'cache_hits': st.session_state.get('cache_hits', 0),
                                'total_records': st.session_state.get('total_records', 0)
                            }
                        )
                        
                        st.download_button(
                            "📥 Download Session Info",
                            session_json,
                            file_name=f"session_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"❌ Failed to export session info: {e}")

if __name__ == "__main__":
    try:
        dashboard = DataDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Dashboard failed to initialize: {str(e)}")
        st.exception(e)
        logger.exception("Dashboard initialization failed")