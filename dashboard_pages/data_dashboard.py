from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import re
import functools
import time

import pandas as pd
import streamlit as st
import plotly.graph_objects as go  # Add missing import

from utils.config.config import DashboardConfig  # Correct location for DashboardConfig
from core.data_validator import DataValidator  # For type hinting
from core.session_manager import create_session_manager, show_session_debug_info
from utils.notifier import Notifier
from utils.data_downloader import (
    download_stock_data,
    clear_cache
)
from core.dashboard_utils import (
    initialize_dashboard_session_state,
    setup_page,
    handle_streamlit_error,
    safe_streamlit_metric,
    create_candlestick_chart,
    validate_ohlc_dataframe,    safe_file_write,
    show_success_with_actions,
    DashboardStateManager
)
from core.data_validator import (
    get_global_validator,
    validate_symbols,
    validate_dates,
    batch_validate_dataframe,
    FinancialData
)

# Dashboard logger setup
from utils.logger import get_dashboard_logger
logger = get_dashboard_logger(__name__)

# Initialize the page at module level (like data_dashboard_v2.py)
setup_page(
    title="📈 Stock Data Download Dashboard", 
    logger_name=__name__,
    sidebar_title="Dashboard Controls"
)

def handle_streamlit_exception(method):
    """
    Decorator to catch exceptions in Streamlit callbacks and methods.
    Uses the centralized error handler from dashboard_utils.
    """
    @functools.wraps(method)  # functools is now imported
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            handle_streamlit_error(e, f"{method.__name__}")
            return None
    return wrapper

class DataDashboard:
    """
    Streamlit dashboard for downloading and previewing stock data.
    Uses the modular data validation system from core.data_validator.
    """

    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        validator: Optional[DataValidator] = None,
        notifier: Optional[Notifier] = None
    ):
        """Initialize the dashboard with configuration and dependencies."""
        self.config: DashboardConfig = config or DashboardConfig()
        self.validator: DataValidator = validator or get_global_validator()
        self.notifier: Notifier = notifier or Notifier()
        
        # Always use create_session_manager from core.session_manager
        self.session_manager = create_session_manager("data_dashboard")
        
        # Use dashboard state manager from utils
        self.state_manager = DashboardStateManager()
        
        # Set unique page load time if not already set
        if 'page_load_time' not in st.session_state:
            st.session_state['page_load_time'] = datetime.now()

        self._setup_directories()
        
        self.saved_paths: List[Path] = []
        self.symbols: List[str] = []
        self.start_date: date = date.today() - timedelta(days=365)
        self.end_date: date = date.today()
        self.interval: str = "1d"
        self.clean_old: bool = True

        logger.info("Initialized DataDashboard with default state variables.")
        self._init_session_state()
        logger.debug("Session state initialized with defaults.")
        
        if 'saved_paths' in st.session_state:
            self.saved_paths = [Path(p) for p in st.session_state['saved_paths']]
            logger.info(f"Restored {len(self.saved_paths)} saved data file paths from previous session.")

    def _init_session_state(self):
        """Ensure required session state variables exist with default values."""
        # Use centralized session state initialization
        self.state_manager.initialize_session_state()
        
        # Add dashboard-specific defaults
        defaults = {
            "data_fetched": False,
            "last_fetch_time": None,
            "fetch_count": 0,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    def _setup_directories(self):
        """Create necessary directories for data and models if they don't exist."""
        try:
            self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.critical(f"Failed to create directory {e.filename}: {e.strerror}", exc_info=True)
            st.error(f"Fatal Error: Could not create required directory ({e.filename}).")
            raise
        logger.info(f"Data directory: {self.config.DATA_DIR}, Model directory: {self.config.MODEL_DIR}")

    def _render_inputs(self):
        """Render the input controls for symbols, dates, and options."""
        st.subheader("Data Selection")
        st.info("Note: Intraday intervals (e.g. '1m', '5m') typically limit history to ~60 days.")        # Symbol input with validation (updated to match v2 approach)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symbols_input = self.session_manager.create_text_input(
                "🎯 Ticker Symbols",
                value=", ".join(self.config.DEFAULT_SYMBOLS),
                help="Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)",
                placeholder="AAPL, MSFT, GOOGL",
                text_input_name="symbol_input"
            )
        
        with col2:
            validate_clicked = self.session_manager.create_button(
                "🔄 Validate", 
                "validate_symbols",
                help="Check symbol validity"
            )
        
        if validate_clicked:
            self._validate_symbols_realtime(symbols_input)
        
        # Process symbols (matching v2 logic exactly)
        previous_symbols = self.symbols.copy()
        try:
            # Use the global validate_symbols function for robust validation
            result = validate_symbols(symbols_input)
            if result.is_valid:
                self.symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
                if self.symbols != previous_symbols:
                    st.session_state["data_fetched"] = False
                    logger.info(f"Symbols updated: {previous_symbols} → {self.symbols}")
            else:
                st.error(f"❌ Invalid symbols: {result.errors}")
                self.symbols = []
        except Exception as e:
            st.error(f"❌ Invalid symbols: {e}")
            self.symbols = []

        self._display_symbol_status()        # Interval selection
        self.interval = self.session_manager.create_selectbox(
            "Data Interval",
            options=self.config.VALID_INTERVALS,
            index=0,
            help="Choose frequency of data points (daily='1d', hourly='1h', etc.)",
            selectbox_name="interval_select"
        )

        # Date range selection with validation
        self._render_date_inputs()        # Options: clean old data files
        col1, _ = st.columns(2)
        with col1:
            self.clean_old = self.session_manager.create_checkbox(
                "Clean old CSVs before fetching?",
                "clean_old_csvs",
                value=self.clean_old,
                help="Delete previously saved CSV files for these symbols before downloading new data."
            )
        st.divider()

    def _validate_symbols_realtime(self, symbols_input: str) -> None:
        """Perform real-time symbol validation using the global validator."""
        if not symbols_input.strip():
            st.warning("⚠️ Please enter at least one symbol")
            return
        
        with st.spinner("🔍 Validating symbols..."):
            try:
                result = validate_symbols(symbols_input)
                if result.is_valid:
                    st.success(f"✅ **Valid symbols:** {symbols_input}")
                else:
                    st.error(f"❌ Invalid symbols: {result.errors}")
            except Exception as e:
                logger.error(f"Symbol validation error: {e}")
                st.error(f"❌ Validation failed: {str(e)}")

    def _display_symbol_status(self) -> None:
        """Display symbol status (renamed from _show_symbol_status for consistency)"""
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

    def _render_date_inputs(self):
        """Render start and end date inputs with validation and presets."""
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
                help="Start date for historical data",
                date_input_name="start_date_input"
            )
            
        with col2:
            new_end = self.session_manager.create_date_input(
                "End Date",
                value=current_end,
                max_value=today,
                help="End date for historical data",
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

        # Use the global validate_dates function
        result = validate_dates(self.start_date, self.end_date)
        if not result.is_valid:
            st.error(f"Invalid date range: {result.errors}")
        else:
            days_span = (self.end_date - self.start_date).days
            st.success(f"✅ Valid range: {days_span} days ({self.start_date} to {self.end_date})")

    @handle_streamlit_exception
    def _clean_existing_files(self):
        """Remove existing CSV files for the selected symbols (called before fetching new data if clean_old=True)."""
        removed = 0
        for symbol in self.symbols:
            pattern = f"{symbol}*"
            for file in self.config.DATA_DIR.glob(pattern):
                try:
                    file.unlink()
                    removed += 1
                except Exception as e:
                    logger.error(f"Failed to delete file {file}: {e}")
        if removed > 0:
            logger.info(f"Removed {removed} old data file(s) before fetching new data.")
        st.session_state["data_fetched"] = False

    def _download(self, symbols: List[str], start_date: date, end_date: date, interval: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Download OHLCV data for given symbols with concurrent downloads and progress tracking.
        """
        if not symbols:
            logger.warning("No symbols provided to download.")
            return None
            
        # Show progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_symbols = len(symbols)
        completed = 0
        results = {}
        errors = []
        
        try:
            # Import here to avoid issues if not available
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_workers = min(3, total_symbols)  # Conservative concurrent limit
            
            status_text.text(f"Downloading {total_symbols} symbols with {max_workers} workers...")
            
            def download_single_symbol(symbol):
                """Download data for a single symbol"""
                try:
                    from utils.data_downloader import download_stock_data as single_download
                    data = single_download(
                        symbols=[symbol],
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval,
                        notifier=self.notifier
                    )
                    return symbol, data.get(symbol) if data else None
                except Exception as e:
                    logger.error(f"Failed to download {symbol}: {e}")
                    return symbol, None
            
            # Use ThreadPoolExecutor for concurrent downloads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all download tasks
                future_to_symbol = {
                    executor.submit(download_single_symbol, symbol): symbol 
                    for symbol in symbols
                }
                
                # Process completed downloads
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    completed += 1
                    progress = completed / total_symbols
                    progress_bar.progress(progress)
                    status_text.text(f"Downloaded {completed}/{total_symbols}: {symbol}")
                    
                    try:
                        symbol_result, df = future.result()
                        if df is not None and not df.empty:
                            # Flatten MultiIndex columns if present
                            if isinstance(df.columns, pd.MultiIndex):
                                df.columns = ['_'.join([str(lvl) for lvl in col if lvl]) for col in df.columns.values]
                            results[symbol_result] = df
                        else:
                            errors.append(f"{symbol}: No data returned")
                    except Exception as e:
                        errors.append(f"{symbol}: {str(e)}")
                        
        except ImportError:
            # Fallback to sequential downloads if concurrent.futures not available
            status_text.text("Using sequential downloads...")
            for i, symbol in enumerate(symbols):
                progress = (i + 1) / total_symbols
                progress_bar.progress(progress)
                status_text.text(f"Downloading {i + 1}/{total_symbols}: {symbol}")
                
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
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = ['_'.join([str(lvl) for lvl in col if lvl]) for col in df.columns.values]
                        results[symbol] = df
                    else:
                        errors.append(f"{symbol}: No data returned")
                except Exception as e:
                    errors.append(f"{symbol}: {str(e)}")
                    
        finally:
            progress_bar.empty()
            status_text.empty()
            
        # Show results summary
        if results:
            st.success(f"✅ Successfully downloaded {len(results)}/{total_symbols} symbols")
        if errors:
            st.warning(f"⚠️ {len(errors)} download(s) failed")
            with st.expander("Error Details", expanded=False):
                for error in errors:
                    st.text(error)
                    
        return results if results else None

    def _save_data_safely(self, symbol: str, df: pd.DataFrame) -> bool:
        """Save dataframe using safe file operations from dashboard_utils."""
        try:
            file_path = self.config.DATA_DIR / f"{symbol}_{self.interval}.csv"
            
            # Ensure 'timestamp' column exists for compatibility
            df_to_save = df.copy()
            if isinstance(df_to_save.index, pd.DatetimeIndex):
                df_to_save = df_to_save.reset_index().rename(columns={df_to_save.index.name or "index": "timestamp"})
            elif "date" in df_to_save.columns:
                df_to_save = df_to_save.rename(columns={"date": "timestamp"})
            
            # Convert to CSV string and use safe file write
            csv_content = df_to_save.to_csv(index=False)
            success, message, backup_path = safe_file_write(file_path, csv_content, create_backup=True)
            
            if success:
                self.saved_paths.append(file_path)
                logger.info(f"Data for {symbol} saved safely to {file_path}")
                return True
            else:
                logger.error(f"Failed to save data for {symbol}: {message}")
                st.error(f"⚠️ Failed to save data for {symbol}: {message}")
                return False
                
        except Exception as e:
            handle_streamlit_error(e, f"saving data for {symbol}")
            return False

    @handle_streamlit_exception
    def _fetch_and_display_data(self):
        """Fetch data for current symbols and date range, then display it in the app."""
        if not self.symbols:
            st.error("Please enter at least one valid stock symbol.")
            return 0
        result = validate_dates(self.start_date, self.end_date)
        if not result.is_valid:
            st.error(f"Cannot fetch data: {result.errors}")
            return 0

        # Add progress indicator
        with st.spinner("Downloading data..."):
            try:
                logger.info(f"Starting download for symbols: {self.symbols}")
                logger.info(f"Date range: {self.start_date} to {self.end_date}")
                logger.info(f"Interval: {self.interval}")

                if self.clean_old:
                    self._clean_existing_files()

                raw_df = self._download(self.symbols, self.start_date, self.end_date, self.interval)
                
                if not raw_df:
                    st.error("No data returned from download_stock_data.")
                    return 0

                logger.info(f"Downloaded data for {len(raw_df)} symbols")
                
                self.saved_paths = []
                successful = 0
                
                # Validate and save data for each symbol
                for symbol, df in raw_df.items():
                    # Batch-validate the DataFrame using FinancialData model
                    valid, errors, summary = batch_validate_dataframe(df, FinancialData, return_summary=True)
                    if summary['valid'] > 0:
                        if self._save_data_safely(symbol, df):
                            successful += 1
                        st.success(f"{symbol}: {summary['valid']} valid records, {summary['invalid']} errors.")
                    else:
                        st.error(f"{symbol}: No valid records. Errors: {errors}")
                  # Update session state using safe operations
                st.session_state['saved_paths'] = [str(p) for p in self.saved_paths]

                if successful > 0:
                    # Calculate total records across all symbols
                    total_records = sum(len(df) for df in raw_df.values())
                    
                    st.session_state["data_fetched"] = True
                    st.session_state["last_fetch_time"] = datetime.now()
                    st.session_state["fetch_count"] = st.session_state.get("fetch_count", 0) + 1
                    st.session_state["total_records"] = st.session_state.get("total_records", 0) + total_records
                    
                    # Enhanced success message with details
                    st.success(f"🎉 Downloaded {successful} symbol(s) • {total_records:,} total records")
                    
                for symbol, df in raw_df.items():
                    self._display_symbol_data(symbol, df)
                return successful
                
            except Exception as e:
                logger.error(f"Data fetch/display error: {e}")
                st.error(f"❌ Data fetch/display failed: {str(e)}")
                return 0

    def _display_symbol_data(self, symbol: str, df: pd.DataFrame):
        """Display symbol data with enhanced charting from dashboard_utils."""
        st.subheader(f"Data Preview: {symbol}")
        tab1, tab2, tab3 = st.tabs(["📋 Recent Data", "📈 Chart", "📊 Metrics"])
        
        with tab1:
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                st.write(df.tail(10))
                
                # Use safe metrics display
                col1, col2, col3 = st.columns(3)
                with col1:
                    safe_streamlit_metric("Total Records", str(len(df)))
                with col2:
                    safe_streamlit_metric("Start Date", str(df['timestamp'].min().date()))
                with col3:
                    safe_streamlit_metric("End Date", str(df['timestamp'].max().date()))
            else:
                st.write(df.tail(10))
                
        with tab2:
            try:
                # Simple approach using plotly directly
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    # Create candlestick if OHLC data available
                    fig.add_trace(go.Candlestick(
                        x=df.index if 'timestamp' not in df.columns else df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=symbol
                    ))
                elif 'close' in df.columns:
                    # Create line chart if only close price available
                    fig.add_trace(go.Scatter(
                        x=df.index if 'timestamp' not in df.columns else df['timestamp'],
                        y=df['close'],
                        mode='lines',
                        name=symbol
                    ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart",
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Price ($)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                handle_streamlit_error(e, f"creating chart for {symbol}")
                # Fallback to simple line chart
                if 'close' in df.columns:
                    st.line_chart(df['close'])
                else:
                    st.error("No suitable price data for charting")
                    
        with tab3:
            # Display data quality metrics
            if 'close' in df.columns:
                try:
                    col1, col2 = st.columns(2)
                    with col1:
                        safe_streamlit_metric("Latest Price", f"${df['close'].iloc[-1]:.2f}")
                        safe_streamlit_metric("Min Price", f"${df['close'].min():.2f}")
                    with col2:
                        safe_streamlit_metric("Max Price", f"${df['close'].max():.2f}")
                        safe_streamlit_metric("Avg Price", f"${df['close'].mean():.2f}")
                except Exception as e:
                    handle_streamlit_error(e, f"calculating metrics for {symbol}")

    def _show_export_interface(self):
        """Display simple export interface"""
        st.subheader("📁 Export Downloaded Data")
        
        if self.session_manager.create_button("❌ Close Export", "close_export"):
            st.session_state["show_export"] = False
            st.rerun()
            return
        
        if not self.saved_paths:
            st.warning("⚠️ No data files available for export")
            return
        
        # Show available files
        st.write(f"**Found {len(self.saved_paths)} data files:**")
        for path in self.saved_paths:
            if path.exists():
                try:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    mod_time = datetime.fromtimestamp(path.stat().st_mtime)
                    st.write(f"• {path.name} ({size_mb:.1f} MB) - {mod_time.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.write(f"• {path.name}")
        
        # Export options
        export_format = self.session_manager.create_selectbox(
            "Export Format:",
            options=["ZIP Archive", "Individual CSV"],
            selectbox_name="export_format"
        )
        
        if self.session_manager.create_button("📥 Create Export", "create_export", type="primary"):
            try:
                with st.spinner("Creating export..."):
                    export_dir = self.config.DATA_DIR / "exports"
                    export_dir.mkdir(exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    if export_format == "ZIP Archive":
                        import zipfile
                        zip_path = export_dir / f"stockdata_export_{timestamp}.zip"
                        
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for path in self.saved_paths:
                                if path.exists():
                                    zipf.write(path, path.name)
                        
                        # Offer download
                        with open(zip_path, 'rb') as f:
                            st.download_button(
                                label=f"📁 Download {zip_path.name}",
                                data=f.read(),
                                file_name=zip_path.name,
                                mime="application/zip"
                            )
                    else:
                        # Individual CSV download buttons
                        st.subheader("📄 Download Individual Files")
                        for path in self.saved_paths:
                            if path.exists():
                                with open(path, 'rb') as f:
                                    st.download_button(
                                        label=f"📄 {path.name}",
                                        data=f.read(),
                                        file_name=path.name,
                                        mime="text/csv"
                                    )
                    
                    st.success("✅ Export ready for download!")
                    
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
                logger.error(f"Export error: {e}")

    def run(self):
        """Main dashboard application entry point."""
        
        # Enhanced sidebar controls with performance metrics
        with st.sidebar:
            st.subheader("📊 Dashboard Metrics")
            
            # Core session stats
            col1, col2 = st.columns(2)
            with col1:
                safe_streamlit_metric("Downloads", str(st.session_state.get("fetch_count", 0)))
                safe_streamlit_metric("Cache Hits", str(st.session_state.get("cache_hits", 0)))
            with col2:
                safe_streamlit_metric("Total Records", str(st.session_state.get("total_records", 0)))
                if st.session_state.get("last_fetch_time"):
                    safe_streamlit_metric("Last Fetch", str(st.session_state["last_fetch_time"].strftime("%H:%M:%S")))
                else:
                    safe_streamlit_metric("Last Fetch", "Never")
            
            # Data files summary
            if self.saved_paths:
                st.subheader("📁 Data Files")
                total_size_mb = 0
                for path in self.saved_paths:
                    if path.exists():
                        try:
                            size_bytes = path.stat().st_size
                            total_size_mb += size_bytes / (1024 * 1024)
                        except:
                            pass
                safe_streamlit_metric("Saved Files", str(len(self.saved_paths)))
                safe_streamlit_metric("Total Size", f"{total_size_mb:.1f} MB")
            
            st.divider()
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if self.session_manager.create_button("🗑️ Clear Cache", "clear_cache", help="Clear all cached data"):
                    try:
                        clear_cache()
                        st.success("Cache cleared!")
                        import time
                        time.sleep(1)  # Brief pause before rerun
                        st.rerun()
                    except Exception as e:
                        handle_streamlit_error(e, "clearing cache")
            
            with col2:
                if self.session_manager.create_button("📊 Export Data", "export_data", help="Export downloaded data"):
                    st.session_state["show_export"] = True
                    st.rerun()
          # Main content area
        self._render_inputs()
        
        # Download button wrapped in form to prevent immediate rerun
        with self.session_manager.form_container("download_form"):
            st.markdown("### Ready to Download?")
            st.write(f"Click below to download data for: **{', '.join(self.symbols) if self.symbols else 'No symbols selected'}**")
            
            fetch_clicked = st.form_submit_button(
                "📥 Download Data", 
                type="primary", 
                use_container_width=True,
                disabled=not self.symbols  # Disable if no symbols
            )
        
        if fetch_clicked:
            success_count = self._fetch_and_display_data()
            if success_count > 0:
                st.balloons()  # Celebrate success!        else:
            # Show cached data if available
            if st.session_state.get("data_fetched") and self.saved_paths:
                st.subheader("📂 Previously Downloaded Data")
                for path in self.saved_paths:
                    if path.exists():
                        symbol = path.name.split('_')[0]
                        try:
                            df = pd.read_csv(path, parse_dates=["timestamp"])
                            self._display_symbol_data(symbol, df)
                        except Exception as e:
                            handle_streamlit_error(e, f"reading cached data for {symbol}")
                    else:
                        logger.warning(f"Cached file not found: {path}")
            else:                # Show helpful message when no data is available
                st.info(
                    "👋 **Welcome!** Select your symbols and date range above, "
                    "then click **Download Data** to get started."
                )
        
        # Show export interface if requested
        if st.session_state.get("show_export", False):
            self._show_export_interface()
                
        # Show SessionManager debug info in a sidebar expandable section
        with st.sidebar.expander("🔧 Session Debug Info", expanded=False):
            show_session_debug_info()

# Direct execution like data_dashboard_v2.py
initialize_dashboard_session_state()
dashboard = DataDashboard()
dashboard.run()
