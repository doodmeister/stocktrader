"""
Stock Data Download Dashboard

A Streamlit-based dashboard for downloading, caching, and previewing stock market data.
Supports multiple data sources (E*Trade, Yahoo Finance) with robust error handling,
input validation, and performance optimization.

Key Features:
- Multi-symbol data download with configurable intervals
- Automatic data validation and cleaning
- Session state management and caching
- Real-time progress tracking and notifications
- Integration with the broader trading platform architecture
"""

import asyncio
import functools
import hashlib
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.logger import setup_logger
from utils.config.config import DashboardConfig
from utils.data_validator import DataValidator
from utils.notifier import Notifier
from utils.data_downloader import download_stock_data, clear_cache
from utils.security import sanitize_filename, validate_file_path
from core.dashboard_utils import initialize_dashboard_session_state

# Configure logging
logger = setup_logger(__name__)

# Constants
MAX_CONCURRENT_DOWNLOADS = 5
CACHE_EXPIRY_HOURS = 24
DEFAULT_CHART_HEIGHT = 400
PROGRESS_UPDATE_INTERVAL = 0.1


class DataDownloadError(Exception):
    """Custom exception for data download errors."""
    pass


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


def handle_streamlit_exception(func):
    """
    Decorator for robust exception handling in Streamlit methods.
    
    Logs exceptions with full context and displays user-friendly error messages.
    Prevents application crashes while maintaining debugging information.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except (DataDownloadError, DataValidationError) as e:
            logger.warning(f"Expected error in {func.__name__}: {e}")
            st.warning(str(e))
            return None
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}: {e}")
            st.error(f"An unexpected error occurred: {e}")
            # In production, you might want to send this to a monitoring service
            return None
    return wrapper


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_data(file_path: str, file_hash: str) -> Optional[pd.DataFrame]:
    """
    Cache data loading with file hash verification for integrity.
    
    Args:
        file_path: Path to the CSV file
        file_hash: SHA256 hash of the file for cache invalidation
        
    Returns:
        Cached DataFrame or None if cache miss/invalid
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return None
            
        # Verify file integrity
        with open(path, 'rb') as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
        
        if current_hash != file_hash:
            logger.debug(f"Cache invalidated for {file_path} due to file changes")
            return None
            
        df = pd.read_csv(path, parse_dates=["timestamp"])
        logger.debug(f"Loaded cached data for {path.stem}: {len(df)} records")
        return df
        
    except Exception as e:
        logger.warning(f"Failed to load cached data from {file_path}: {e}")
        return None


class DataDashboard:
    """
    Production-grade Streamlit dashboard for stock data management.
    
    This class provides a comprehensive interface for downloading, validating,
    and previewing stock market data with enterprise-level features:
    
    - Asynchronous data fetching with progress tracking
    - Intelligent caching and session management
    - Robust error handling and recovery
    - Security-focused input validation
    - Performance optimization for large datasets
    - Integration with notification systems
    """

    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        validator: Optional[DataValidator] = None,
        notifier: Optional[Notifier] = None
    ):
        """
        Initialize the dashboard with dependency injection for testability.
        
        Args:
            config: Dashboard configuration (uses default if None)
            validator: Data validator instance (creates default if None)
            notifier: Notification service (creates default if None)
        """
        self.config = config or DashboardConfig()
        self.validator = validator or DataValidator()
        self.notifier = notifier or Notifier()
        
        # Initialize core state
        self._setup_directories()
        self._init_instance_variables()
        self._init_session_state()
        
        # Performance tracking
        self.start_time = time.time()
        self.operation_metrics = {}
        
        logger.info(
            f"DataDashboard initialized - "
            f"Data dir: {self.config.DATA_DIR}, "
            f"Session symbols: {len(self.symbols)}"
        )

    def _init_instance_variables(self) -> None:
        """Initialize instance variables with secure defaults."""
        self.saved_paths: List[Path] = []
        self.symbols: List[str] = []
        self.start_date: date = date.today() - timedelta(days=365)
        self.end_date: date = date.today()
        self.interval: str = "1d"
        self.clean_old: bool = True
        self.max_workers: int = min(MAX_CONCURRENT_DOWNLOADS, len(self.config.DEFAULT_SYMBOLS))

    def _init_session_state(self) -> None:
        """Initialize session state with secure defaults and validation."""
        defaults = {
            "data_fetched": False,
            "last_fetch_time": None,
            "fetch_count": 0,
            "saved_paths": [],
            "download_progress": 0.0,
            "current_operation": "",
            "error_count": 0,
            "last_error": None,
            "cache_hits": 0,
            "total_records": 0
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # Restore saved paths if available
        if st.session_state.get('saved_paths'):
            self.saved_paths = [Path(p) for p in st.session_state['saved_paths'] 
                             if Path(p).exists()]
            logger.debug(f"Restored {len(self.saved_paths)} valid file paths from session")

    def _setup_directories(self) -> None:
        """Create required directories with proper error handling."""
        directories = [
            self.config.DATA_DIR,
            self.config.MODEL_DIR,
            self.config.DATA_DIR / "cache",
            self.config.DATA_DIR / "temp"
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                # Verify write permissions
                test_file = directory / ".write_test"
                test_file.touch()
                test_file.unlink()
                
            except (OSError, PermissionError) as e:
                logger.critical(f"Failed to create/access directory {directory}: {e}")
                st.error(f"🚨 Critical: Cannot access required directory: {directory}")
                st.stop()
        
        logger.info(f"Verified access to {len(directories)} required directories")

    def _render_performance_metrics(self) -> None:
        """Display performance and session metrics in sidebar."""
        if not st.sidebar.checkbox("Show Performance Metrics", value=False):
            return
            
        with st.sidebar.expander("📊 Session Metrics", expanded=False):
            uptime = time.time() - self.start_time
            
            metrics_data = {
                "Uptime": f"{uptime:.1f}s",
                "Downloads": st.session_state.get("fetch_count", 0),
                "Cache Hits": st.session_state.get("cache_hits", 0),
                "Total Records": st.session_state.get("total_records", 0),
                "Errors": st.session_state.get("error_count", 0)
            }
            
            for label, value in metrics_data.items():
                st.metric(label, value)

    def _render_inputs(self) -> None:
        """Render input controls with enhanced validation and UX."""
        st.subheader("📋 Data Configuration")
        
        # Enhanced info box with dynamic content
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
        """Render symbol input with real-time validation."""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symbols_input = st.text_input(
                "🎯 Ticker Symbols",
                value=", ".join(self.config.DEFAULT_SYMBOLS),
                help="Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)",
                placeholder="AAPL, MSFT, GOOGL"
            )
        
        with col2:
            if st.button("🔄 Validate", help="Check symbol validity"):
                self._validate_symbols_realtime(symbols_input)
        
        # Process symbols with validation
        previous_symbols = self.symbols.copy()
        try:
            self.symbols = self.validator.validate_symbols(symbols_input)
            if self.symbols != previous_symbols:
                st.session_state["data_fetched"] = False
                logger.info(f"Symbols updated: {previous_symbols} → {self.symbols}")
                
        except ValueError as e:
            st.error(f"❌ Invalid symbols: {e}")
            self.symbols = []

        self._display_symbol_status()

    def _validate_symbols_realtime(self, symbols_input: str) -> None:
        """Perform real-time symbol validation with user feedback."""
        if not symbols_input.strip():
            st.warning("Please enter at least one symbol")
            return
            
        try:
            symbols = self.validator.validate_symbols(symbols_input)
            st.success(f"✅ Valid symbols: {', '.join(symbols)}")
        except ValueError as e:
            st.error(f"❌ Validation failed: {e}")

    def _display_symbol_status(self) -> None:
        """Display enhanced symbol status with metadata."""
        if not self.symbols:
            st.warning("⚠️ No valid symbols selected")
            return
            
        # Display symbols with estimated data size
        with st.container():
            st.success(f"✅ **Selected Symbols:** {', '.join(self.symbols)}")
            
            # Estimate data size and download time
            days_requested = (self.end_date - self.start_date).days
            estimated_records = len(self.symbols) * days_requested
            estimated_size_mb = estimated_records * 0.1  # Rough estimate
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Symbols", len(self.symbols))
            col2.metric("Est. Records", f"{estimated_records:,}")
            col3.metric("Est. Size", f"{estimated_size_mb:.1f} MB")

    def _render_interval_selection(self) -> None:
        """Render interval selection with contextual information."""
        intervals_with_desc = {
            "1m": "1 Minute (High frequency, limited history)",
            "5m": "5 Minutes (Intraday trading)",
            "15m": "15 Minutes (Short-term analysis)",
            "30m": "30 Minutes (Swing trading)",
            "1h": "1 Hour (Medium-term analysis)",
            "1d": "1 Day (Long-term analysis)"
        }
        
        # Find current selection index
        current_index = 0
        if self.interval in self.config.VALID_INTERVALS:
            current_index = self.config.VALID_INTERVALS.index(self.interval)
        
        self.interval = st.selectbox(
            "📊 Data Interval",
            options=self.config.VALID_INTERVALS,
            index=current_index,
            format_func=lambda x: intervals_with_desc.get(x, x),
            help="Choose the frequency of data points"
        )

    def _render_date_inputs(self) -> None:
        """Render date inputs with intelligent validation and presets."""
        st.subheader("📅 Date Range")
        
        # Quick preset buttons
        col1, col2, col3, col4 = st.columns(4)
        today = date.today()
        
        if col1.button("📅 1 Week", help="Last 7 days"):
            self.start_date = today - timedelta(days=7)
            self.end_date = today
            
        if col2.button("📅 1 Month", help="Last 30 days"):
            self.start_date = today - timedelta(days=30)
            self.end_date = today
            
        if col3.button("📅 3 Months", help="Last 90 days"):
            self.start_date = today - timedelta(days=90)
            self.end_date = today
            
        if col4.button("📅 1 Year", help="Last 365 days"):
            self.start_date = today - timedelta(days=365)
            self.end_date = today

        # Manual date selection
        col1, col2 = st.columns(2)
        with col1:
            new_start = st.date_input(
                "Start Date",
                value=self.start_date,
                max_value=today,
                help="Beginning of data range"
            )
            
        with col2:
            new_end = st.date_input(
                "End Date",
                value=self.end_date,
                max_value=today,
                help="End of data range"
            )

        # Update dates and validate
        if new_start != self.start_date or new_end != self.end_date:
            self.start_date = new_start
            self.end_date = new_end
            st.session_state["data_fetched"] = False

        # Validation with detailed feedback
        if not self.validator.validate_dates(self.start_date, self.end_date):
            st.error("❌ Invalid date range: Start date must be ≤ End date")
        else:
            days_span = (self.end_date - self.start_date).days
            st.info(f"📊 Range: {days_span} days ({self.start_date} to {self.end_date})")

    def _render_options(self) -> None:
        """Render additional options and advanced settings."""
        st.subheader("⚙️ Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.clean_old = st.checkbox(
                "🧹 Clean old files",
                value=self.clean_old,
                help="Remove existing CSV files before downloading new data"
            )
            
        with col2:
            self.max_workers = st.slider(
                "🚀 Concurrent Downloads",
                min_value=1,
                max_value=10,
                value=self.max_workers,
                help="Number of parallel download threads"
            )

        # Advanced options in expander
        with st.expander("🔧 Advanced Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                cache_enabled = st.checkbox(
                    "💾 Enable Caching",
                    value=True,
                    help="Cache downloaded data for faster subsequent loads"
                )
                
            with col2:
                validate_data = st.checkbox(
                    "✅ Validate Data Quality",
                    value=True,
                    help="Perform data quality checks after download"
                )
            
            # Store advanced options in session state
            st.session_state.update({
                "cache_enabled": cache_enabled,
                "validate_data": validate_data
            })

    @handle_streamlit_exception
    def _clean_existing_files(self) -> int:
        """
        Clean existing CSV files with detailed progress tracking.
        
        Returns:
            Number of files successfully removed
        """
        if not self.symbols:
            return 0
            
        files_to_remove = []
        for symbol in self.symbols:
            pattern = f"{sanitize_filename(symbol)}*"
            files_to_remove.extend(self.config.DATA_DIR.glob(pattern))
        
        if not files_to_remove:
            logger.info("No existing files to clean")
            return 0

        # Progress tracking for file removal
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        removed_count = 0
        total_files = len(files_to_remove)
        
        for i, file_path in enumerate(files_to_remove):
            try:
                if validate_file_path(file_path, self.config.DATA_DIR):
                    file_path.unlink()
                    removed_count += 1
                    logger.debug(f"Removed file: {file_path}")
                else:
                    logger.warning(f"Skipped suspicious file path: {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")
                
            # Update progress
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Cleaning files... {i + 1}/{total_files}")
        
        progress_bar.empty()
        status_text.empty()
        
        if removed_count > 0:
            st.success(f"🧹 Cleaned {removed_count} existing file(s)")
            logger.info(f"Successfully removed {removed_count}/{total_files} files")
            
        st.session_state["data_fetched"] = False
        return removed_count

    def _download_with_progress(
        self, 
        symbols: List[str], 
        start_date: date, 
        end_date: date, 
        interval: str
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Download data with real-time progress tracking and error recovery.
        
        Args:
            symbols: List of stock symbols to download
            start_date: Start date for data range
            end_date: End date for data range
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            Dictionary mapping symbols to DataFrames, or None on failure
        """
        if not symbols:
            raise DataDownloadError("No symbols provided for download")
            
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            total_symbols = len(symbols)
            completed_symbols = 0
            results = {}
            errors = []
            
            # Use ThreadPoolExecutor for concurrent downloads
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all download tasks
                future_to_symbol = {
                    executor.submit(
                        download_stock_data,
                        [symbol],
                        start_date,
                        end_date,
                        interval,
                        self.notifier
                    ): symbol for symbol in symbols
                }
                
                # Process completed downloads
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    
                    try:
                        data = future.result(timeout=60)  # 60 second timeout per symbol
                        if data and symbol in data:
                            results[symbol] = data[symbol]
                            logger.info(f"Successfully downloaded data for {symbol}")
                        else:
                            errors.append(f"No data returned for {symbol}")
                            
                    except Exception as e:
                        error_msg = f"Failed to download {symbol}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                    
                    # Update progress
                    completed_symbols += 1
                    progress = completed_symbols / total_symbols
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Downloading... {completed_symbols}/{total_symbols} "
                        f"({len(results)} successful, {len(errors)} errors)"
                    )
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Report results
            if results:
                st.success(
                    f"✅ Successfully downloaded {len(results)}/{total_symbols} symbols"
                )
                
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
            raise DataDownloadError(f"Download failed: {str(e)}")

    def _validate_and_process_data(
        self, 
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Validate and process downloaded data with quality checks.
        
        Args:
            data_dict: Raw data dictionary from download
            
        Returns:
            Processed and validated data dictionary
        """
        if not data_dict:
            raise DataValidationError("No data to validate")
            
        processed_data = {}
        validation_errors = []
        
        for symbol, df in data_dict.items():
            try:
                # Basic data validation
                if df.empty:
                    validation_errors.append(f"{symbol}: Empty dataset")
                    continue
                
                # Standardize column names and index
                df = self._standardize_dataframe(df)
                
                # Data quality checks
                quality_issues = self._check_data_quality(df, symbol)
                if quality_issues:
                    validation_errors.extend(quality_issues)
                
                # Additional processing if validation enabled
                if st.session_state.get("validate_data", True):
                    df = self._enhance_dataframe(df, symbol)
                
                processed_data[symbol] = df
                logger.debug(f"Validated data for {symbol}: {len(df)} records")
                
            except Exception as e:
                error_msg = f"{symbol}: Processing failed - {str(e)}"
                validation_errors.append(error_msg)
                logger.error(error_msg)
        
        # Report validation results
        if validation_errors:
            st.warning(f"⚠️ Data validation found {len(validation_errors)} issue(s)")
            with st.expander("Validation Issues", expanded=False):
                for error in validation_errors:
                    st.text(f"• {error}")
        
        if not processed_data:
            raise DataValidationError("No valid data after processing")
            
        return processed_data

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame format for consistency."""
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(lvl) for lvl in col if lvl]) for col in df.columns.values]
        
        # Ensure timestamp column exists
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        elif "date" in df.columns:
            df.rename(columns={"date": "timestamp"}, inplace=True)
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Standardize OHLCV column names
        column_mapping = {
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        }
        
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        return df

    def _check_data_quality(self, df: pd.DataFrame, symbol: str) -> List[str]:
        """Perform comprehensive data quality checks."""
        issues = []
        
        # Check for required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"{symbol}: Missing columns: {', '.join(missing_cols)}")
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            issues.append(f"{symbol}: Null values found in {null_counts[null_counts > 0].to_dict()}")
        
        # Check for invalid OHLC relationships
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).sum()
            
            if invalid_ohlc > 0:
                issues.append(f"{symbol}: {invalid_ohlc} invalid OHLC relationships")
        
        # Check for extreme values
        if 'close' in df.columns:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()  # More than 50% change
            if extreme_changes > 0:
                issues.append(f"{symbol}: {extreme_changes} extreme price changes (>50%)")
        
        return issues

    def _enhance_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add computed columns and enhancements to the DataFrame."""
        try:
            # Add basic technical indicators if OHLCV data is available
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Calculate daily returns
                df['returns'] = df['close'].pct_change()
                
                # Calculate simple moving averages
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                
                # Calculate volatility
                df['volatility'] = df['returns'].rolling(window=20).std()
                
                logger.debug(f"Enhanced data for {symbol} with technical indicators")
            
        except Exception as e:
            logger.warning(f"Failed to enhance data for {symbol}: {e}")
        
        return df

    @handle_streamlit_exception
    def _save_data_with_metadata(
        self, 
        data_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[List[Path], int]:
        """
        Save data with metadata and integrity checks.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame mappings
            
        Returns:
            Tuple of (saved file paths, number of successful saves)
        """
        if not data_dict:
            raise DataValidationError("No data to save")
        
        saved_paths = []
        successful_saves = 0
        total_records = 0
        
        # Progress tracking for saves
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (symbol, df) in enumerate(data_dict.items()):
            try:
                # Generate secure filename
                safe_symbol = sanitize_filename(symbol)
                file_path = self.config.DATA_DIR / f"{safe_symbol}_{self.interval}.csv"
                
                # Backup existing file if it exists
                if file_path.exists():
                    backup_path = file_path.with_suffix('.csv.bak')
                    file_path.replace(backup_path)
                
                # Save with atomic write (write to temp, then move)
                temp_path = file_path.with_suffix('.tmp')
                df.to_csv(temp_path, index=False)
                temp_path.replace(file_path)
                
                # Create metadata file
                metadata = {
                    'symbol': symbol,
                    'interval': self.interval,
                    'start_date': self.start_date.isoformat(),
                    'end_date': self.end_date.isoformat(),
                    'records': len(df),
                    'created_at': datetime.now().isoformat(),
                    'file_size': file_path.stat().st_size
                }
                
                metadata_path = file_path.with_suffix('.meta.json')
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                saved_paths.append(file_path)
                successful_saves += 1
                total_records += len(df)
                
                logger.info(f"Saved {len(df)} records for {symbol} to {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save data for {symbol}: {e}")
                st.error(f"⚠️ Failed to save {symbol}: {e}")
            
            # Update progress
            progress = (i + 1) / len(data_dict)
            progress_bar.progress(progress)
            status_text.text(f"Saving... {i + 1}/{len(data_dict)}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Update session state
        st.session_state['saved_paths'] = [str(p) for p in saved_paths]
        st.session_state['total_records'] = total_records
        
        return saved_paths, successful_saves

    @handle_streamlit_exception
    def _fetch_and_display_data(self) -> int:
        """
        Main data fetching workflow with comprehensive error handling.
        
        Returns:
            Number of successfully processed symbols
        """
        # Input validation
        if not self.symbols:
            st.error("❌ Please enter at least one valid stock symbol")
            return 0
            
        if not self.validator.validate_dates(self.start_date, self.end_date):
            st.error("❌ Invalid date range selected")
            return 0
        
        # Pre-flight checks
        st.info("🚀 Starting data download process...")
        
        try:
            # Step 1: Clean existing files if requested
            if self.clean_old:
                with st.spinner("🧹 Cleaning existing files..."):
                    cleaned_count = self._clean_existing_files()
                    if cleaned_count > 0:
                        time.sleep(1)  # Brief pause for user feedback
            
            # Step 2: Download data with progress tracking
            with st.spinner("📥 Downloading market data..."):
                st.session_state["current_operation"] = "Downloading"
                raw_data = self._download_with_progress(
                    self.symbols, self.start_date, self.end_date, self.interval
                )
                
                if not raw_data:
                    st.error("❌ No data was successfully downloaded")
                    return 0
            
            # Step 3: Validate and process data
            with st.spinner("✅ Validating data quality..."):
                st.session_state["current_operation"] = "Validating"
                processed_data = self._validate_and_process_data(raw_data)
            
            # Step 4: Save data with metadata
            with st.spinner("💾 Saving data files..."):
                st.session_state["current_operation"] = "Saving"
                saved_paths, successful_saves = self._save_data_with_metadata(processed_data)
                self.saved_paths = saved_paths
            
            # Step 5: Update session state and display results
            if successful_saves > 0:
                st.session_state.update({
                    "data_fetched": True,
                    "last_fetch_time": datetime.now(),
                    "fetch_count": st.session_state.get("fetch_count", 0) + 1,
                    "current_operation": "Complete"
                })
                
                # Success message with metrics
                total_records = sum(len(df) for df in processed_data.values())
                st.success(
                    f"🎉 **Download Complete!** "
                    f"Downloaded {successful_saves} symbols "
                    f"({total_records:,} total records)"
                )
                
                # Send notification if configured
                try:
                    self.notifier.send_order_notification({
                        'symbol': 'DATA_DOWNLOAD',
                        'side': 'INFO',
                        'message': f"Downloaded {successful_saves} symbols with {total_records} records",
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Failed to send notification: {e}")
            
            # Step 6: Display downloaded data
            for symbol, df in processed_data.items():
                self._display_symbol_data(symbol, df)
            
            return successful_saves
            
        except (DataDownloadError, DataValidationError) as e:
            st.error(f"❌ {str(e)}")
            return 0
        except Exception as e:
            logger.exception(f"Unexpected error in data fetch workflow: {e}")
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            return 0
        finally:
            st.session_state["current_operation"] = ""

    def _create_enhanced_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create enhanced Plotly chart with technical indicators and styling."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price Chart', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Main price chart
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add moving averages if available
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='red', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Volume chart if available
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['volume'],
                    name='Volume',
                    marker_color='rgba(158,202,225,0.8)',
                    hovertemplate='<b>Volume</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Volume: %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{symbol} - Market Data Analysis',
                x=0.5,
                font=dict(size=20)
            ),
            height=DEFAULT_CHART_HEIGHT + 150,
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridcolor='lightgray',
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Price ($)",
            showgrid=True,
            gridcolor='lightgray',
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Volume",
            showgrid=True,
            gridcolor='lightgray',
            row=2, col=1
        )
        
        return fig

    def _display_symbol_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Display comprehensive symbol data with enhanced visualizations."""
        st.subheader(f"📊 {symbol} - Market Data")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Chart", "📋 Data Table", "📊 Statistics", "🔍 Quality Report"
        ])
        
        with tab1:  # Enhanced Chart
            try:
                chart = self._create_enhanced_chart(df, symbol)
                st.plotly_chart(chart, use_container_width=True)
            except Exception as e:
                logger.error(f"Failed to create chart for {symbol}: {e}")
                # Fallback to simple line chart
                try:
                    st.line_chart(df.set_index('timestamp')['close'])
                except Exception:
                    st.error(f"Unable to display chart for {symbol}")
        
        with tab2:  # Data Table
            # Show recent data with better formatting
            display_df = df.tail(20).copy()
            
            # Format numeric columns
            numeric_cols = display_df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if col in ['open', 'high', 'low', 'close']:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
                elif col == 'volume':
                    display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}")
                elif col in ['returns', 'volatility']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Data summary
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", f"{len(df):,}")
            col2.metric("Date Range", f"{(df['timestamp'].max() - df['timestamp'].min()).days} days")
            
            if 'close' in df.columns:
                col3.metric("Price Range", f"${df['close'].min():.2f} - ${df['close'].max():.2f}")
                col4.metric("Latest Price", f"${df['close'].iloc[-1]:.2f}")
        
        with tab3:  # Statistics
            if 'close' in df.columns:
                # Price statistics
                st.subheader("📈 Price Statistics")
                price_stats = df['close'].describe()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Price", f"${price_stats['mean']:.2f}")
                    st.metric("Median Price", f"${price_stats['50%']:.2f}")
                    st.metric("Standard Deviation", f"${price_stats['std']:.2f}")
                
                with col2:
                    st.metric("Min Price", f"${price_stats['min']:.2f}")
                    st.metric("Max Price", f"${price_stats['max']:.2f}")
                    st.metric("Price Range", f"${price_stats['max'] - price_stats['min']:.2f}")
                
                # Returns analysis if available
                if 'returns' in df.columns:
                    st.subheader("📊 Returns Analysis")
                    returns_stats = df['returns'].describe()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Return", f"{returns_stats['mean']:.4f}")
                    col2.metric("Volatility", f"{returns_stats['std']:.4f}")
                    col3.metric("Sharpe Ratio", f"{returns_stats['mean'] / returns_stats['std']:.2f}")
        
        with tab4:  # Quality Report
            st.subheader("🔍 Data Quality Report")
            
            # Missing data analysis
            missing_data = df.isnull().sum()
            if missing_data.any():
                st.warning("⚠️ Missing Data Detected")
                st.dataframe(missing_data[missing_data > 0], use_container_width=True)
            else:
                st.success("✅ No missing data found")
            
            # Data completeness
            expected_records = (self.end_date - self.start_date).days
            actual_records = len(df)
            completeness = actual_records / expected_records * 100 if expected_records > 0 else 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Records", f"{expected_records:,}")
            col2.metric("Actual Records", f"{actual_records:,}")
            col3.metric("Completeness", f"{completeness:.1f}%")
            
            # Data type information
            st.subheader("📋 Data Types")
            dtype_info = pd.DataFrame({
                'Column': df.columns,
                'Type': [str(dtype) for dtype in df.dtypes],
                'Non-Null Count': [df[col].count() for col in df.columns]
            })
            st.dataframe(dtype_info, use_container_width=True, hide_index=True)

    def _display_cached_data(self) -> None:
        """Display previously cached data with performance optimization."""
        if not self.saved_paths:
            return
            
        st.info("📋 Displaying cached data from previous session")
        
        for path in self.saved_paths:
            if not path.exists():
                continue
                
            symbol = path.stem.split('_')[0]
            
            # Calculate file hash for cache validation
            with open(path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Try to load from cache first
            df = get_cached_data(str(path), file_hash)
            
            if df is None:
                # Cache miss - load directly
                try:
                    df = pd.read_csv(path, parse_dates=["timestamp"])
                    logger.debug(f"Cache miss for {symbol} - loaded from disk")
                except Exception as e:
                    logger.error(f"Failed to load cached data for {symbol}: {e}")
                    continue
            else:
                st.session_state["cache_hits"] = st.session_state.get("cache_hits", 0) + 1
                logger.debug(f"Cache hit for {symbol}")
            
            self._display_symbol_data(symbol, df)

    def run(self) -> None:
        """Main dashboard application with enhanced UI and error handling."""
        # Header with branding
        st.title("📈 Stock Data Download Dashboard")
        st.markdown(
            "**Enterprise-grade data acquisition** • "
            "Download, validate, and analyze market data with advanced features"
        )
        
        # Performance metrics in sidebar
        self._render_performance_metrics()
        
        # Main content tabs
        tabs = st.tabs(["📥 Data Download", "🔍 Analysis Hub", "⚙️ Settings"])
        
        with tabs[0]:  # Main Data Download Tab
            self._render_inputs()
            
            # Action buttons with enhanced styling
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                fetch_clicked = st.button(
                    "🚀 Download Data",
                    type="primary",
                    use_container_width=True,
                    help="Start the data download process"
                )
            
            with col2:
                if st.button("🧹 Clear Cache", help="Clear all cached data"):
                    try:
                        clear_cache()
                        st.success("✅ Cache cleared")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to clear cache: {e}")
            
            with col3:
                if st.button("🔄 Refresh", help="Refresh the current view"):
                    st.experimental_rerun()
            
            st.divider()
            
            # Main content area
            if fetch_clicked:
                successful_downloads = self._fetch_and_display_data()
                if successful_downloads > 0:
                    st.balloons()  # Celebration for successful downloads
            else:
                # Display cached data if available
                if st.session_state.get("data_fetched") and self.saved_paths:
                    self._display_cached_data()
                else:
                    # Welcome message for new users
                    st.info(
                        "👋 **Welcome!** Select your symbols and date range above, "
                        "then click **Download Data** to get started."
                    )
        
        with tabs[1]:  # Analysis Hub
            st.header("🔍 Data Analysis Hub")
            st.info(
                "🚀 **Advanced Analysis** • "
                "Visit the **Model Training** page for machine learning features, "
                "**Pattern Recognition**, and **Strategy Backtesting**."
            )
            
            # Quick links to other features
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **🤖 Machine Learning**
                - Pattern Neural Networks
                - Feature Engineering
                - Model Training & Validation
                """)
            
            with col2:
                st.markdown("""
                **📊 Technical Analysis**
                - Candlestick Patterns
                - Technical Indicators
                - Price Action Analysis
                """)
            
            with col3:
                st.markdown("""
                **📈 Strategy Testing**
                - Historical Backtesting
                - Performance Metrics
                - Risk Analysis
                """)
        
        with tabs[2]:  # Settings
            st.header("⚙️ Dashboard Settings")
            
            # Data management settings
            st.subheader("💾 Data Management")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Delete All Data", help="Remove all downloaded data files"):
                    if st.checkbox("⚠️ I understand this will delete all data", key="confirm_delete"):
                        try:
                            # Implement safe data deletion
                            deleted_count = 0
                            for file in self.config.DATA_DIR.glob("*.csv"):
                                if validate_file_path(file, self.config.DATA_DIR):
                                    file.unlink()
                                    deleted_count += 1
                            
                            st.success(f"✅ Deleted {deleted_count} data files")
                            st.session_state["data_fetched"] = False
                            self.saved_paths.clear()
                            
                        except Exception as e:
                            st.error(f"❌ Failed to delete data: {e}")
            
            with col2:
                if st.button("📊 Export Session Info", help="Download session information"):
                    try:
                        session_info = {
                            'timestamp': datetime.now().isoformat(),
                            'symbols': self.symbols,
                            'date_range': {
                                'start': self.start_date.isoformat(),
                                'end': self.end_date.isoformat()
                            },
                            'interval': self.interval,
                            'session_stats': {
                                'fetch_count': st.session_state.get('fetch_count', 0),
                                'cache_hits': st.session_state.get('cache_hits', 0),
                                'total_records': st.session_state.get('total_records', 0)
                            }
                        }
                        
                        import json
                        session_json = json.dumps(session_info, indent=2)
                        st.download_button(
                            "📥 Download Session Info",
                            session_json,
                            file_name=f"session_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"❌ Failed to export session info: {e}")


def main():
    """Main entry point with enhanced error handling and monitoring."""
    try:
        # Initialize session state and dashboard
        initialize_dashboard_session_state()
        
        # Create and run dashboard
        dashboard = DataDashboard()
        dashboard.run()
        
    except Exception as e:
        logger.critical(f"Critical error in main application: {e}", exc_info=True)
        st.error(
            "🚨 **Critical Error** • The application encountered a serious error. "
            "Please check the logs and restart the application."
        )
        
        # Display error details for debugging (only in development)
        if st.checkbox("Show Debug Information", value=False):
            st.exception(e)


if __name__ == "__main__" or st._is_running_with_streamlit:
    main()