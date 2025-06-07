"""
Stock Data Dashboard - Production-grade implementation with robust export functionality.

This module provides a comprehensive Streamlit dashboard for downloading, validating,
and exporting stock market data with production-level error handling and performance.
"""

import functools
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.dashboard_utils import (
    DashboardStateManager,
    handle_streamlit_error,
    safe_streamlit_metric,
    setup_page
)
from core.data_validator import (
    DataValidator,
    get_global_validator,
    validate_symbols
)
from core.session_manager import create_session_manager, show_session_debug_info
from utils.config.config import DashboardConfig
from utils.data_downloader import download_stock_data
from utils.logger import get_dashboard_logger
from utils.notifier import Notifier
from utils.io import (
    create_zip_archive,
    save_dataframe_with_metadata,
    get_file_info,
    clean_directory,
    export_session_data,
    validate_file_path
)

# Module-level logger
logger = get_dashboard_logger(__name__)

# Initialize page configuration
setup_page(
    title="📈 Stock Data Download Dashboard",
    logger_name=__name__,
    sidebar_title="Dashboard Controls"
)


def handle_streamlit_exception(method):
    """
    Production-grade exception handler decorator for Streamlit callbacks.
    
    Args:
        method: The method to wrap with exception handling
        
    Returns:
        Wrapped method with comprehensive error handling
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            handle_streamlit_error(e, f"{method.__name__}")
            return None
    return wrapper


class DataDashboardConfig:
    """Configuration constants for the Data Dashboard."""
    
    # File and export settings
    MAX_FILE_SIZE_MB = 100
    MAX_EXPORT_FILES = 5
    EXPORT_CLEANUP_COUNT = 5
    
    # Performance settings
    MAX_CONCURRENT_DOWNLOADS = 3
    DOWNLOAD_TIMEOUT_SECONDS = 30
    
    # UI settings
    MAX_DISPLAYED_FILES = 3
    CHART_HEIGHT = 400
    
    # Data validation
    MAX_SYMBOL_LENGTH = 10
    MAX_SYMBOLS_COUNT = 20
    MIN_DATE_RANGE_DAYS = 1
    MAX_DATE_RANGE_DAYS = 3650  # ~10 years


class ExportManager:
    """Handles all export operations using utils.io functionality."""
    
    def __init__(self, config: DashboardConfig, logger_instance):
        self.config = config
        self.logger = logger_instance
        self.export_dir = self.config.DATA_DIR / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_files_for_export(self, file_paths: List[Path]) -> Tuple[List[Path], int]:
        """
        Validate files before export operation using utils.io.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Tuple of (valid_paths, total_size_bytes)
        """
        valid_paths = []
        total_size = 0
        
        for path in file_paths:
            # Use utils.io for file validation
            if not validate_file_path(path, self.config.DATA_DIR):
                self.logger.warning(f"File path security check failed: {path}")
                continue
            
            file_info = get_file_info(path)
            
            if not file_info["exists"]:
                self.logger.warning(f"File not found: {path}")
                continue
            
            if not file_info["is_file"]:
                self.logger.warning(f"Not a file: {path}")
                continue
            
            if file_info["size"] == 0:
                self.logger.warning(f"Empty file skipped: {path}")
                continue
            
            if file_info["size_mb"] > DataDashboardConfig.MAX_FILE_SIZE_MB:
                self.logger.warning(f"File too large: {path} ({file_info['size_mb']:.1f}MB)")
                continue
            
            valid_paths.append(path)
            total_size += file_info["size"]
        
        return valid_paths, total_size
    
    def create_zip_export(self, file_paths: List[Path], timestamp: str) -> Optional[Path]:
        """
        Create ZIP archive using utils.io functionality.
        
        Args:
            file_paths: List of valid file paths to archive
            timestamp: Timestamp string for filename
            
        Returns:
            Path to created ZIP file or None if failed
        """
        zip_path = self.export_dir / f"stockdata_export_{timestamp}.zip"
        
        try:
            # Use utils.io to create ZIP archive
            zip_data = create_zip_archive(file_paths)
            
            if not zip_data:
                self.logger.error("No data returned from ZIP archive creation")
                return None
            
            # Write ZIP file as binary
            try:
                zip_path.write_bytes(zip_data)
            except Exception as e:
                self.logger.error(f"Failed to write ZIP file: {e}")
                return None
            
            # Validate ZIP creation
            zip_info = get_file_info(zip_path)
            if not zip_info["exists"] or zip_info["size"] == 0:
                self.logger.error("ZIP file was not created or is empty")
                return None
            
            self.logger.info(f"Successfully created ZIP with {len(file_paths)} files: {zip_path}")
            return zip_path
            
        except Exception as e:
            self.logger.error(f"ZIP creation failed: {e}", exc_info=True)
            # Clean up failed ZIP file
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
            return None
    
    def cleanup_old_exports(self, keep_count: int = DataDashboardConfig.EXPORT_CLEANUP_COUNT) -> int:
        """
        Clean up old export files using utils.io functionality.
        
        Args:
            keep_count: Number of recent exports to keep
            
        Returns:
            Number of files cleaned up
        """
        try:
            # Get all ZIP export files
            zip_files = sorted(
                self.export_dir.glob("stockdata_export_*.zip"),
                key=lambda x: get_file_info(x)["modified"],
                reverse=True
            )
            
            if len(zip_files) <= keep_count:
                return 0
            
            # Remove old files
            files_to_remove = zip_files[keep_count:]
            removed_count = 0
            
            for old_zip in files_to_remove:
                try:
                    old_zip.unlink()
                    removed_count += 1
                    self.logger.info(f"Cleaned up old export: {old_zip.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {old_zip.name}: {e}")
            
            return removed_count
            
        except Exception as e:
            self.logger.warning(f"Export cleanup failed: {e}")
            return 0
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


class DataDownloadManager:
    """Manages data download operations with concurrent processing and robust error handling."""
    
    def __init__(self, config: DashboardConfig, notifier: Notifier, logger_instance):
        self.config = config
        self.notifier = notifier
        self.logger = logger_instance
    
    def download_data_concurrent(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        interval: str
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        Download data for multiple symbols with concurrent processing.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval
            
        Returns:
            Tuple of (successful_downloads, error_messages)
        """
        if not symbols:
            return {}, ["No symbols provided"]
        
        # Setup progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        errors = []
        max_workers = min(DataDashboardConfig.MAX_CONCURRENT_DOWNLOADS, len(symbols))
        
        try:
            status_text.text(f"Downloading {len(symbols)} symbols with {max_workers} workers...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit download tasks
                future_to_symbol = {
                    executor.submit(
                        self._download_single_symbol,
                        symbol, start_date, end_date, interval
                    ): symbol for symbol in symbols
                }
                
                # Process completed downloads
                completed = 0
                for future in as_completed(future_to_symbol, timeout=DataDashboardConfig.DOWNLOAD_TIMEOUT_SECONDS):
                    symbol = future_to_symbol[future]
                    completed += 1
                    
                    progress = completed / len(symbols)
                    progress_bar.progress(progress)
                    status_text.text(f"Downloaded {completed}/{len(symbols)}: {symbol}")
                    
                    try:
                        df = future.result()
                        if df is not None and not df.empty:
                            results[symbol] = df
                        else:
                            errors.append(f"{symbol}: No data returned")
                    except Exception as e:
                        error_msg = f"{symbol}: {str(e)}"
                        errors.append(error_msg)
                        self.logger.error(f"Download failed for {symbol}: {e}")
        
        except Exception as e:
            self.logger.error(f"Concurrent download failed: {e}", exc_info=True)
            errors.append(f"Download system error: {str(e)}")
        
        finally:
            progress_bar.empty()
            status_text.empty()
        
        return results, errors
    
    def _download_single_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        Download data for a single symbol with error handling.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame or None if failed
        """
        try:
            data = download_stock_data(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                notifier=self.notifier
            )
            
            if data and symbol in data:
                df = data[symbol]
                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join([str(lvl) for lvl in col if lvl]) for col in df.columns.values]
                return df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Single symbol download failed for {symbol}: {e}")
            raise


class DataDashboard:
    """
    Production-grade Streamlit dashboard for stock data download and management.
    
    Features:
    - Robust data validation and download
    - Concurrent data processing
    - Comprehensive export functionality using utils.io
    - Production-level error handling
    - Resource management and cleanup
    """
    
    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        validator: Optional[DataValidator] = None,
        notifier: Optional[Notifier] = None
    ):
        """
        Initialize dashboard with dependency injection for testability.
        
        Args:
            config: Dashboard configuration instance
            validator: Data validator instance
            notifier: Notification handler instance
        """
        self.config = config or DashboardConfig()
        self.validator = validator or get_global_validator()
        self.notifier = notifier or Notifier()
        
        # Initialize managers
        self.session_manager = create_session_manager("data_dashboard")
        self.state_manager = DashboardStateManager()
        self.export_manager = ExportManager(self.config, logger)
        self.download_manager = DataDownloadManager(self.config, self.notifier, logger)
        
        # Setup required directories
        self._setup_directories()
        
        # Initialize state variables
        self._initialize_dashboard_state()
        
        logger.info("DataDashboard initialized successfully")
    
    def _setup_directories(self) -> None:
        """Create required directories with proper error handling."""
        try:
            self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directories ready: DATA={self.config.DATA_DIR}, MODEL={self.config.MODEL_DIR}")
        except OSError as e:
            logger.critical(f"Failed to create directory {e.filename}: {e.strerror}")
            st.error(f"Critical Error: Could not create required directory ({e.filename})")
            raise
    
    def _initialize_dashboard_state(self) -> None:
        """Initialize dashboard state with default values."""
        # Use centralized state initialization
        self.state_manager.initialize_session_state()
        
        # Dashboard-specific defaults
        defaults = {
            "data_fetched": False,
            "last_fetch_time": None,
            "fetch_count": 0,
            "total_records": 0,
            "saved_paths": [],
            "page_load_time": datetime.now()
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Initialize instance variables
        self.saved_paths = [Path(p) for p in st.session_state.get('saved_paths', [])]
        self.symbols = []
        self.start_date = date.today() - timedelta(days=365)
        self.end_date = date.today()
        self.interval = "1d"
        self.clean_old = True
        
        logger.debug("Dashboard state initialized")
    
    def run(self) -> None:
        """Main dashboard execution method."""
        try:
            # Render main interface
            self._render_inputs()
            
            # Download button
            if st.button("🚀 Download Data", type="primary", use_container_width=True):
                if self._validate_inputs():
                    self._fetch_and_display_data()
                else:
                    st.error("❌ Please check your inputs before downloading")
            
            # Sidebar export interface
            with st.sidebar:
                st.divider()
                self._render_export_interface()
                
                # Session export option
                self._render_session_export()
                
                # Debug information (if enabled)
                if st.checkbox("🔧 Debug Info", key="show_debug"):
                    show_session_debug_info()
                    
        except Exception as e:
            handle_streamlit_error(e, "running dashboard")
    
    def _validate_inputs(self) -> bool:
        """
        Validate all user inputs before processing.
        
        Returns:
            True if all inputs are valid
        """
        if not self.symbols:
            st.error("❌ Please enter at least one valid stock symbol")
            return False
        
        if len(self.symbols) > DataDashboardConfig.MAX_SYMBOLS_COUNT:
            st.error(f"❌ Too many symbols (max: {DataDashboardConfig.MAX_SYMBOLS_COUNT})")
            return False
        
        if self.start_date >= self.end_date:
            st.error("❌ Start date must be before end date")
            return False
        
        date_range_days = (self.end_date - self.start_date).days
        if date_range_days < DataDashboardConfig.MIN_DATE_RANGE_DAYS:
            st.error("❌ Date range too small")
            return False
        
        if date_range_days > DataDashboardConfig.MAX_DATE_RANGE_DAYS:
            st.error(f"❌ Date range too large (max: {DataDashboardConfig.MAX_DATE_RANGE_DAYS} days)")
            return False
        
        return True
    
    def _render_inputs(self) -> None:
        """Render all input controls with validation."""
        st.subheader("📊 Data Selection")
        st.info("💡 Intraday intervals (1m, 5m) typically limit history to ~60 days")
        
        self._render_symbol_inputs()
        self._render_interval_selection()
        self._render_date_inputs()
        self._render_options()
        
        st.divider()
    
    def _render_symbol_inputs(self) -> None:
        """Render symbol input controls with real-time validation."""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symbols_input = self.session_manager.create_text_input(
                "🎯 Stock Symbols",
                value=", ".join(self.config.DEFAULT_SYMBOLS),
                help="Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)",
                placeholder=f"e.g., {', '.join(self.config.DEFAULT_SYMBOLS[:3])}",
                text_input_name="symbol_input"
            )
        
        with col2:
            if self.session_manager.create_button("🔍 Validate", "validate_symbols"):
                self._validate_symbols_realtime(symbols_input)
        
        # Process symbol input
        if symbols_input and symbols_input.strip():
            new_symbols = self._parse_and_validate_symbols(symbols_input)
            if new_symbols != self.symbols:
                self.symbols = new_symbols
                st.session_state["data_fetched"] = False
                logger.info(f"Symbols updated: {self.symbols}")
        else:
            self.symbols = []
        
        self._display_symbol_status()
    
    def _parse_and_validate_symbols(self, symbols_input: str) -> List[str]:
        """
        Parse and validate symbol input string.
        
        Args:
            symbols_input: Comma-separated symbol string
            
        Returns:
            List of validated symbols
        """
        symbols = []
        for symbol in symbols_input.split(','):
            symbol = symbol.strip().upper()
            if symbol and len(symbol) <= DataDashboardConfig.MAX_SYMBOL_LENGTH:
                # Basic symbol validation
                if re.match(r'^[A-Z][A-Z0-9.-]*$', symbol):
                    symbols.append(symbol)
        
        return symbols
    
    def _validate_symbols_realtime(self, symbols_input: str) -> None:
        """Perform real-time symbol validation with user feedback."""
        if not symbols_input.strip():
            st.warning("⚠️ Please enter at least one symbol")
            return
        
        with st.spinner("🔍 Validating symbols..."):
            try:
                result = validate_symbols(symbols_input)
                if result.is_valid:
                    st.success(f"✅ Valid symbols: {symbols_input}")
                else:
                    st.error(f"❌ Validation errors: {', '.join(result.errors)}")
            except Exception as e:
                logger.error(f"Symbol validation error: {e}")
                st.error(f"❌ Validation failed: {str(e)}")
    
    def _display_symbol_status(self) -> None:
        """Display current symbol status and estimates."""
        if not self.symbols:
            st.caption("⚪ No symbols selected")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"**Active symbols:** {', '.join(self.symbols)}")
            st.caption(f"📊 {len(self.symbols)} symbol(s) selected")
            
            # Estimate data points
            days_span = (self.end_date - self.start_date).days
            estimated_points = self._estimate_data_points(days_span, self.interval, len(self.symbols))
            st.caption(f"📈 Estimated data points: ~{estimated_points:,}")
        
        with col2:
            # Size estimate
            estimated_size = self._estimate_download_size(estimated_points)
            st.metric("Est. Size", estimated_size)
    
    def _estimate_data_points(self, days: int, interval: str, symbol_count: int) -> int:
        """Estimate total data points based on parameters."""
        multipliers = {
            "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
            "1h": 24, "1d": 1, "1wk": 1/7, "1mo": 1/30
        }
        points_per_day = multipliers.get(interval, 1)
        return int(days * points_per_day * symbol_count)
    
    def _estimate_download_size(self, data_points: int) -> str:
        """Estimate download size in human-readable format."""
        bytes_per_record = 100  # Estimated bytes per OHLCV record
        total_bytes = data_points * bytes_per_record
        return ExportManager.format_file_size(total_bytes)
    
    def _render_interval_selection(self) -> None:
        """Render interval selection control."""
        interval_selected = self.session_manager.create_selectbox(
            "📅 Data Interval",
            options=self.config.VALID_INTERVALS,
            index=0,
            help="Data frequency: daily='1d', hourly='1h', etc.",
            selectbox_name="interval_select"
        )
        
        if interval_selected and interval_selected != self.interval:
            self.interval = interval_selected
            st.session_state["data_fetched"] = False
            logger.info(f"Interval changed to: {self.interval}")
    
    def _render_date_inputs(self) -> None:
        """Render date selection controls with presets and validation."""
        st.subheader("📅 Date Range")
        
        # Date preset buttons
        self._render_date_presets()
        
        # Manual date selection
        col1, col2 = st.columns(2)
        
        with col1:
            new_start = self.session_manager.create_date_input(
                "Start Date",
                value=self.start_date,
                max_value=date.today(),
                help="Start date for historical data",
                date_input_name="start_date_input"
            )
        
        with col2:
            new_end = self.session_manager.create_date_input(
                "End Date", 
                value=self.end_date,
                max_value=date.today(),
                help="End date for historical data",
                date_input_name="end_date_input"
            )
        
        # Update dates if changed
        if new_start and new_start != self.start_date:
            self.start_date = new_start
            st.session_state["data_fetched"] = False
        
        if new_end and new_end != self.end_date:
            self.end_date = new_end
            st.session_state["data_fetched"] = False
        
        # Display date range validation
        self._display_date_validation()
    
    def _render_date_presets(self) -> None:
        """Render quick date preset buttons."""
        col1, col2, col3, col4 = st.columns(4)
        
        presets = [
            ("1 Week", 7, col1),
            ("1 Month", 30, col2), 
            ("3 Months", 90, col3),
            ("1 Year", 365, col4)
        ]
        
        today = date.today()
        
        for label, days, col in presets:
            with col:
                if self.session_manager.create_button(
                    f"📅 {label}",
                    f"preset_{days}days",
                    help=f"Set range to last {days} days"
                ):
                    self.start_date = today - timedelta(days=days)
                    self.end_date = today
                    st.session_state["data_fetched"] = False
                    st.success(f"✅ Date range set to {label}")
    
    def _display_date_validation(self) -> None:
        """Display date range validation status."""
        if self.start_date >= self.end_date:
            st.error("❌ Start date must be before end date")
        else:
            days_span = (self.end_date - self.start_date).days
            st.success(f"✅ Valid range: {days_span} days ({self.start_date} to {self.end_date})")
    
    def _render_options(self) -> None:
        """Render additional options."""
        col1, _ = st.columns(2)
        with col1:
            self.clean_old = self.session_manager.create_checkbox(
                "🗑️ Clean old files before download",
                "clean_old_csvs",
                value=self.clean_old,
                help="Remove existing CSV files for selected symbols before downloading new data"
            )
    
    @handle_streamlit_exception
    def _fetch_and_display_data(self) -> int:
        """
        Main data fetching and display logic with comprehensive error handling.
        
        Returns:
            Number of successfully processed symbols
        """
        if not self._validate_inputs():
            return 0
        
        with st.spinner("📥 Downloading data..."):
            try:
                logger.info(f"Starting download: symbols={self.symbols}, range={self.start_date} to {self.end_date}")
                
                # Clean old files if requested
                if self.clean_old:
                    self._clean_existing_files()
                
                # Download data concurrently
                raw_data, errors = self.download_manager.download_data_concurrent(
                    self.symbols, self.start_date, self.end_date, self.interval
                )
                
                if not raw_data:
                    st.error("❌ No data was successfully downloaded")
                    if errors:
                        with st.expander("Error Details"):
                            for error in errors:
                                st.text(error)
                    return 0
                
                # Process and save data
                successful = self._process_and_save_data(raw_data)
                
                # Update session state
                self._update_session_state_after_fetch(successful, raw_data)
                
                # Display results
                if successful > 0:
                    self._display_download_results(successful, raw_data)
                    for symbol, df in raw_data.items():
                        self._display_symbol_data(symbol, df)
                
                return successful
                
            except Exception as e:
                logger.error(f"Data fetch error: {e}", exc_info=True)
                st.error(f"❌ Download failed: {str(e)}")
                return 0
    
    def _process_and_save_data(self, raw_data: Dict[str, pd.DataFrame]) -> int:
        """
        Process and save downloaded data using utils.io functionality.
        
        Args:
            raw_data: Dictionary of symbol -> DataFrame
            
        Returns:
            Number of successfully saved files
        """
        self.saved_paths = []
        successful = 0
        
        for symbol, df in raw_data.items():
            if df is None or df.empty:
                st.warning(f"⚠️ {symbol}: No data received")
                continue
            
            try:
                # Process DataFrame for saving
                processed_df = self._prepare_dataframe_for_saving(df)
                
                # Save data with metadata using utils.io
                if self._save_data_with_metadata(symbol, processed_df):
                    successful += 1
                    st.success(f"✅ {symbol}: {len(processed_df)} records saved")
                else:
                    st.error(f"❌ {symbol}: Failed to save")
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                st.error(f"❌ {symbol}: Processing error - {str(e)}")
        
        return successful
    
    def _prepare_dataframe_for_saving(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for safe file saving.
        
        Args:
            df: Raw DataFrame from download
            
        Returns:
            Processed DataFrame ready for saving
        """
        df_copy = df.copy()
        
        # Ensure timestamp column exists
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy = df_copy.reset_index()
            df_copy.rename(columns={df_copy.columns[0]: 'timestamp'}, inplace=True)
        elif 'Date' in df_copy.columns:
            df_copy.rename(columns={'Date': 'timestamp'}, inplace=True)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df_copy.columns:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        
        return df_copy
    
    def _save_data_with_metadata(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Save DataFrame with metadata using utils.io functionality.
        utils.io.save_dataframe_with_metadata is expected to handle hash calculation.
        
        Args:
            symbol: Stock symbol
            df: DataFrame to save
            
        Returns:
            True if save was successful
        """
        try:
            file_path = self.config.DATA_DIR / f"{symbol}_{self.interval}.csv"
            
            # Create metadata - file_hash will be populated by the save_dataframe_with_metadata utility
            metadata = {
                'symbol': symbol,
                'interval': self.interval,
                'start_date': str(self.start_date),
                'end_date': str(self.end_date),
                'download_timestamp': datetime.now().isoformat(),
                'record_count': len(df),
                'columns': list(df.columns),
                'file_hash': None  # To be calculated and filled by save_dataframe_with_metadata
            }
            
            # Save using utils.io.
            # save_dataframe_with_metadata is now responsible for saving the DataFrame,
            # calculating its hash, adding the hash to the metadata, and saving the metadata JSON.
            success, message, backup_path = save_dataframe_with_metadata(
                df, file_path, metadata, create_backup=True
            )
            
            if success:
                self.saved_paths.append(file_path)
                # The log message assumes the utility now handles the hash.
                logger.info(f"Data for {symbol} saved to {file_path} with metadata (hash included by save utility).")
                return True
            else:
                logger.error(f"Failed to save {symbol}: {message}")
                return False
                
        except Exception as e:
            handle_streamlit_error(e, f"saving data for {symbol}")
            return False
    
    def _update_session_state_after_fetch(self, successful: int, raw_data: Dict[str, pd.DataFrame]) -> None:
        """Update session state after successful data fetch."""
        total_records = sum(len(df) for df in raw_data.values() if df is not None)
        
        st.session_state.update({
            "data_fetched": True,
            "last_fetch_time": datetime.now(),
            "fetch_count": st.session_state.get("fetch_count", 0) + 1,
            "total_records": st.session_state.get("total_records", 0) + total_records,
            "saved_paths": [str(p) for p in self.saved_paths]
        })
    
    def _display_download_results(self, successful: int, raw_data: Dict[str, pd.DataFrame]) -> None:
        """Display download results summary."""
        total_records = sum(len(df) for df in raw_data.values() if df is not None)
        st.success(f"🎉 Successfully downloaded {successful} symbol(s) • {total_records:,} total records")
    
    @handle_streamlit_exception
    def _clean_existing_files(self) -> None:
        """Clean existing CSV files for selected symbols using utils.io."""
        removed_count = 0
        removed_files = []
        
        for symbol in self.symbols:
            pattern = f"{symbol}*"
            try:
                # Use utils.io clean_directory for each symbol pattern
                count, files = clean_directory(self.config.DATA_DIR, pattern, dry_run=False)
                removed_count += count
                removed_files.extend(files)
            except Exception as e:
                logger.error(f"Failed to clean files for {symbol}: {e}")
        
        if removed_count > 0:
            st.info(f"🗑️ Cleaned {removed_count} old file(s)")
            logger.info(f"Cleaned {removed_count} old data files: {removed_files}")
    
    def _display_symbol_data(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Display comprehensive symbol data with charts and metrics.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with stock data
        """
        st.subheader(f"📊 Data Preview: {symbol}")
        tab1, tab2, tab3 = st.tabs(["📋 Recent Data", "📈 Chart", "📊 Metrics"])
        
        with tab1:
            self._display_data_table(df)
        
        with tab2:
            self._display_price_chart(symbol, df)
        
        with tab3:
            self._display_data_metrics(df)
    
    def _display_data_table(self, df: pd.DataFrame) -> None:
        """Display recent data table with metadata."""
        if "timestamp" in df.columns:
            df_display = df.copy()
            df_display["timestamp"] = pd.to_datetime(df_display["timestamp"])
            st.dataframe(df_display.tail(10), use_container_width=True)
            
            # Display metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                safe_streamlit_metric("Total Records", f"{len(df):,}")
            with col2:
                safe_streamlit_metric("Start Date", str(df_display['timestamp'].min().date()))
            with col3:
                safe_streamlit_metric("End Date", str(df_display['timestamp'].max().date()))
        else:
            st.dataframe(df.tail(10), use_container_width=True)
    
    def _display_price_chart(self, symbol: str, df: pd.DataFrame) -> None:
        """Display interactive price chart with enhanced scaling and larger patterns."""
        try:
            # Create figure with larger default size
            fig = go.Figure()
            
            # Determine x-axis data
            x_data = df['timestamp'] if 'timestamp' in df.columns else df.index
            
            # Enhanced chart configuration for better pattern visibility
            chart_config = {
                'height': DataDashboardConfig.CHART_HEIGHT * 1.5,  # 50% larger
                'line_width': 3,  # Thicker lines
                'candlestick_width': 0.8,  # Wider candlesticks
                'volume_height_ratio': 0.3  # For volume subplot if needed
            }
            
            # Create appropriate chart type with enhanced visibility
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Enhanced candlestick chart
                fig.add_trace(go.Candlestick(
                    x=x_data,
                    open=df['open'],
                    high=df['high'], 
                    low=df['low'],
                    close=df['close'],
                    name=symbol,
                    increasing_line_color='#26A69A',  # Custom green
                    decreasing_line_color='#EF5350',  # Custom red
                    increasing_line_width=2,
                    decreasing_line_width=2,
                    increasing_fillcolor='rgba(38, 166, 154, 0.3)',
                    decreasing_fillcolor='rgba(239, 83, 80, 0.3)'
                ))
                
                # Add volume bars if available (subplot for better pattern visibility)
                if 'volume' in df.columns:
                    # Create subplot with volume
                    from plotly.subplots import make_subplots
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3],  # Price chart gets 70%, volume gets 30%
                        subplot_titles=(f'{symbol} Price', 'Volume')
                    )
                    
                    # Add price candlestick to top subplot
                    fig.add_trace(go.Candlestick(
                        x=x_data,
                        open=df['open'],
                        high=df['high'], 
                        low=df['low'],
                        close=df['close'],
                        name=symbol,
                        increasing_line_color='#26A69A',
                        decreasing_line_color='#EF5350',
                        increasing_line_width=2,
                        decreasing_line_width=2,
                        increasing_fillcolor='rgba(38, 166, 154, 0.3)',
                        decreasing_fillcolor='rgba(239, 83, 80, 0.3)'
                    ), row=1, col=1)
                    
                    # Add volume bars to bottom subplot
                    volume_colors = ['#26A69A' if close >= open else '#EF5350' 
                                   for close, open in zip(df['close'], df['open'])]
                    
                    fig.add_trace(go.Bar(
                        x=x_data,
                        y=df['volume'],
                        name='Volume',
                        marker_color=volume_colors,
                        opacity=0.7
                    ), row=2, col=1)
                    
            elif 'close' in df.columns:
                # Enhanced line chart for close price only
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=df['close'],
                    mode='lines',
                    name=f"{symbol} Close",
                    line=dict(
                        width=chart_config['line_width'],
                        color='#1f77b4'
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Date: %{x}<br>' +
                                'Price: $%{y:.2f}<extra></extra>'
                ))
                
                # Add moving averages for better pattern recognition
                if len(df) >= 20:
                    # 20-day moving average
                    ma20 = df['close'].rolling(window=20).mean()
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=ma20,
                        mode='lines',
                        name='MA20',
                        line=dict(width=2, color='orange', dash='dash'),
                        opacity=0.8
                    ))
                
                if len(df) >= 50:
                    # 50-day moving average
                    ma50 = df['close'].rolling(window=50).mean()
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=ma50,
                        mode='lines',
                        name='MA50',
                        line=dict(width=2, color='red', dash='dot'),
                        opacity=0.8
                    ))
            else:
                st.warning("⚠️ No suitable price data for charting")
                return
            
            # Enhanced layout configuration for better pattern visibility
            layout_config = {
                'title': {
                    'text': f"{symbol} Price Chart - Enhanced View",
                    'font': {'size': 20, 'family': 'Arial, sans-serif'},
                    'x': 0.5
                },
                'height': int(chart_config['height']),
                'showlegend': True,
                'legend': {
                    'orientation': "h",
                    'yanchor': "bottom",
                    'y': 1.02,
                    'xanchor': "right",
                    'x': 1
                },
                'hovermode': 'x unified',
                'template': 'plotly_white',  # Clean white background
                'margin': dict(l=60, r=60, t=80, b=60),
                'font': {'size': 12}
            }
            
            # Configure axes for better scale visibility
            if 'volume' in df.columns and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Multi-subplot layout
                layout_config.update({
                    'xaxis': {
                        'title': 'Date',
                        'gridcolor': 'lightgray',
                        'gridwidth': 1,
                        'showgrid': True
                    },
                    'yaxis': {
                        'title': 'Price ($)',
                        'gridcolor': 'lightgray', 
                        'gridwidth': 1,
                        'showgrid': True,
                        'side': 'left'
                    },
                    'xaxis2': {
                        'title': 'Date',
                        'gridcolor': 'lightgray',
                        'gridwidth': 1,
                        'showgrid': True
                    },
                    'yaxis2': {
                        'title': 'Volume',
                        'gridcolor': 'lightgray',
                        'gridwidth': 1,
                        'showgrid': True,
                        'side': 'left'
                    }
                })
            else:
                # Single chart layout
                layout_config.update({
                    'xaxis': {
                        'title': 'Date',
                        'gridcolor': 'lightgray',
                        'gridwidth': 1,
                        'showgrid': True,
                        'tickfont': {'size': 11}
                    },
                    'yaxis': {
                        'title': 'Price ($)',
                        'gridcolor': 'lightgray',
                        'gridwidth': 1, 
                        'showgrid': True,
                        'tickfont': {'size': 11},
                        'side': 'left'
                    }
                })
            
            # Apply layout configuration
            fig.update_layout(**layout_config)
            
            # Add range selector for better navigation of patterns
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label="7d", step="day", stepmode="backward"),
                            dict(count=30, label="30d", step="day", stepmode="backward"),
                            dict(count=90, label="3m", step="day", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=False),  # Disable range slider for cleaner look
                    type="date"
                )
            )
            
            # Display the enhanced chart
            st.plotly_chart(
                fig, 
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'{symbol}_chart',
                        'height': int(chart_config['height']),
                        'width': 1200,
                        'scale': 2  # Higher resolution export
                    }
                }
            )
            
            # Add chart analysis summary
            self._display_chart_analysis(symbol, df)
            
        except Exception as e:
            handle_streamlit_error(e, f"creating enhanced chart for {symbol}")
            # Fallback to simple line chart with enhanced styling
            if 'close' in df.columns:
                st.line_chart(
                    df['close'], 
                    height=int(DataDashboardConfig.CHART_HEIGHT * 1.2)
                )
            else:
                st.error("❌ Chart generation failed")
    
    def _display_data_metrics(self, df: pd.DataFrame) -> None:
        """Display data metrics in the Metrics tab."""
        try:
            if "close" in df.columns:
                col1, col2, col3 = st.columns(3)
                with col1:
                    safe_streamlit_metric("Mean Price", f"${df['close'].mean():.2f}")
                with col2:
                    safe_streamlit_metric("Median Price", f"${df['close'].median():.2f}")
                with col3:
                    safe_streamlit_metric("Std Dev", f"${df['close'].std():.2f}")
            else:
                st.write("⚠️ No price data available for metrics")
        except Exception as e:
            logger.error(f"Metrics display failed: {e}", exc_info=True)

    def _display_chart_analysis(self, symbol: str, df: pd.DataFrame) -> None:
        """Display quick chart analysis to highlight patterns."""
        if 'close' not in df.columns or len(df) < 5:
            return
        
        try:
            with st.expander(f"📊 {symbol} Pattern Analysis", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Price trend analysis
                    recent_close = df['close'].iloc[-1]
                    week_ago_close = df['close'].iloc[-min(7, len(df))]
                    trend_pct = ((recent_close - week_ago_close) / week_ago_close) * 100
                    
                    trend_emoji = "📈" if trend_pct > 0 else "📉" if trend_pct < 0 else "➡️"
                    st.metric(
                        f"{trend_emoji} 7-Day Trend", 
                        f"{trend_pct:+.2f}%",
                        delta=f"${recent_close - week_ago_close:+.2f}"
                    )
                
                with col2:
                    # Volatility indicator
                    if len(df) >= 20:
                        volatility = df['close'].pct_change().std() * 100
                        vol_level = "High" if volatility > 3 else "Medium" if volatility > 1 else "Low"
                        st.metric(
                            "📊 Volatility",
                            f"{vol_level}",
                            delta=f"{volatility:.2f}%"
                        )
                
                with col3:
                    # Volume analysis (if available)
                    if 'volume' in df.columns and len(df) >= 10:
                        recent_vol = df['volume'].iloc[-5:].mean()
                        avg_vol = df['volume'].mean()
                        vol_ratio = recent_vol / avg_vol
                        
                        vol_status = "Above Avg" if vol_ratio > 1.2 else "Below Avg" if vol_ratio < 0.8 else "Normal"
                        st.metric(
                            "📦 Volume",
                            vol_status,
                            delta=f"{vol_ratio:.2f}x avg"
                        )
                
                # Pattern detection hints
                if len(df) >= 20:
                    st.caption("💡 **Pattern Hints:** Look for trend lines, support/resistance levels, and volume confirmation in the enhanced chart above.")
                
        except Exception as e:
            logger.error(f"Chart analysis failed for {symbol}: {e}")
    
    def _render_export_interface(self) -> None:
        """Render export interface in sidebar using utils.io functionality."""
        st.subheader("📁 Export Data")
        
        if not self.saved_paths:
            st.warning("⚠️ No data files available")
            st.info("💡 Download data first to enable exports")
            return
        
        # Validate and display files using utils.io
        valid_paths, total_size = self.export_manager.validate_files_for_export(self.saved_paths)
        
        if not valid_paths:
            st.error("❌ No valid files found for export")
            return
        
        # Display file summary
        self._display_export_file_summary(valid_paths, total_size)
        
        # Export format selection
        export_format = st.selectbox(
            "Export Format:",
            options=["ZIP Archive", "Individual CSV Files"],
            key="export_format_select",
            help="ZIP packages all files together, Individual allows separate downloads"
        )
        
        # Export button
        if st.button("📥 Create Export", key="create_export_btn", use_container_width=True):
            self._handle_export_request(export_format, valid_paths)
    
    def _display_export_file_summary(self, valid_paths: List[Path], total_size: int) -> None:
        """Display summary of files available for export using utils.io file info."""
        st.write(f"**{len(valid_paths)} file(s) ready:**")
        
        display_count = min(DataDashboardConfig.MAX_DISPLAYED_FILES, len(valid_paths))
        for path in valid_paths[:display_count]:
            file_info = get_file_info(path)
            if file_info["exists"]:
                size_str = ExportManager.format_file_size(file_info["size"])
                st.caption(f"• {file_info['name']} ({size_str})")
            else:
                st.caption(f"• {path.name} (unavailable)")
        
        if len(valid_paths) > display_count:
            st.caption(f"... and {len(valid_paths) - display_count} more")
        
        if total_size > 0:
            total_size_str = ExportManager.format_file_size(total_size)
            st.caption(f"📊 Total size: {total_size_str}")
    
    def _handle_export_request(self, export_format: str, valid_paths: List[Path]) -> None:
        """Handle export request using utils.io functionality."""
        try:
            with st.spinner("Creating export..."):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if export_format == "ZIP Archive":
                    self._create_and_offer_zip_export(valid_paths, timestamp)
                else:
                    self._create_individual_file_exports(valid_paths)
                    
        except Exception as e:
            logger.error(f"Export request failed: {e}", exc_info=True)
            st.error(f"❌ Export failed: {str(e)}")
            self._show_export_fallback_guidance()
    
    def _create_and_offer_zip_export(self, valid_paths: List[Path], timestamp: str) -> None:
        """Create ZIP export using utils.io and offer for download."""
        zip_path = self.export_manager.create_zip_export(valid_paths, timestamp)
        
        if not zip_path:
            st.error("❌ Failed to create ZIP archive")
            return
        
        try:
            # Read ZIP file for download
            with open(zip_path, 'rb') as f:
                zip_data = f.read()
            
            st.download_button(
                label=f"📁 Download {zip_path.name}",
                data=zip_data,
                file_name=zip_path.name,
                mime="application/zip",
                key="zip_download_btn",
                use_container_width=True
            )
            
            # Show success info using utils.io file info
            zip_info = get_file_info(zip_path)
            zip_size = ExportManager.format_file_size(zip_info["size"])
            st.success(f"✅ ZIP archive ready: {zip_size}")
            
            # Cleanup old exports
            cleaned = self.export_manager.cleanup_old_exports()
            if cleaned > 0:
                st.info(f"🗑️ Cleaned up {cleaned} old export(s)")
                
        except Exception as e:
            logger.error(f"ZIP download preparation failed: {e}")
            st.error("❌ Could not prepare ZIP for download")
    
    def _create_individual_file_exports(self, valid_paths: List[Path]) -> None:
        """Create individual file download buttons using utils.io."""
        st.write("**📁 Download Individual Files:**")
        
        for i, path in enumerate(valid_paths):
            try:
                # Get file info using utils.io
                file_info = get_file_info(path)
                
                if not file_info["exists"]:
                    st.error(f"❌ File not accessible: {path.name}")
                    continue
                
                with open(path, 'rb') as f:
                    file_data = f.read()
                
                file_size = ExportManager.format_file_size(file_info["size"])
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.download_button(
                        label=f"📄 {file_info['name']}",
                        data=file_data,
                        file_name=file_info['name'],
                        mime="text/csv",
                        key=f"csv_download_{i}",
                        use_container_width=True
                    )
                with col2:
                    st.caption(file_size)
                    
            except Exception as e:
                logger.error(f"Failed to prepare {path.name}: {e}")
                st.error(f"❌ Could not prepare {path.name}")
        
        st.success(f"✅ {len(valid_paths)} file(s) ready for download")
    
    def _render_session_export(self) -> None:
        """Render session export functionality using utils.io."""
        st.divider()
        st.subheader("📋 Session Export")
        
        if st.button("📊 Export Session Info", key="export_session_btn", use_container_width=True):
            try:
                # Prepare session data
                date_range = {
                    "start": str(self.start_date),
                    "end": str(self.end_date)
                }
                
                session_stats = {
                    "total_records": st.session_state.get("total_records", 0),
                    "fetch_count": st.session_state.get("fetch_count", 0),
                    "last_fetch_time": str(st.session_state.get("last_fetch_time", "")),
                    "saved_files_count": len(self.saved_paths)
                }
                
                # Use utils.io to export session data
                session_json = export_session_data(
                    self.symbols, date_range, self.interval, session_stats
                )
                
                # Offer download
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"session_export_{timestamp}.json"
                
                st.download_button(
                    label=f"📄 Download {filename}",
                    data=session_json,
                    file_name=filename,
                    mime="application/json",
                    key="session_json_download",
                    use_container_width=True
                )
                
                st.success("✅ Session export ready")
                
            except Exception as e:
                logger.error(f"Session export failed: {e}")
                st.error(f"❌ Session export failed: {str(e)}")
    
    def _show_export_fallback_guidance(self) -> None:
        """Show fallback guidance when export fails."""
        st.info("💡 **Manual Export Option:**")
        st.code(f"Data directory: {self.config.DATA_DIR}")
        
        if self.saved_paths:
            st.write("**Available files:**")
            for path in self.saved_paths[:5]:
                file_info = get_file_info(path)
                if file_info["exists"]:
                    st.text(f"• {file_info['name']}")


# Entry point when run as standalone script
if __name__ == "__main__":
    dashboard = DataDashboard()
    dashboard.run()
