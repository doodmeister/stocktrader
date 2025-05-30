from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import re
import functools
import time

import pandas as pd
import streamlit as st
import plotly.graph_objects as go  # Add missing import

from utils.config.config import DashboardConfig
from utils.data_validator import DataValidator
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
from core.session_manager import create_session_manager, show_session_debug_info

# Dashboard logger setup
from utils.logger import get_dashboard_logger
logger = get_dashboard_logger(__name__)

# Initialize the page at module level (like data_dashboard_v2.py)
setup_page(
    title="📈 Stock Data Download Dashboard", 
    logger_name=__name__,
    sidebar_title="Dashboard Controls"
)

# Add local DataValidator class if the import is failing
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
    
    def validate_dates(self, start_date: date, end_date: date) -> bool:
        return start_date <= end_date and end_date <= date.today()

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

    Features:
    - Robust input validation and error handling using dashboard_utils
    - Modular design for maintainability and extensibility
    - Secure handling of user inputs and file operations
    - Auto-refresh and session state management
    """

    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        validator: Optional[DataValidator] = None,
        notifier: Optional[Notifier] = None    ):
        """Initialize the dashboard with configuration and dependencies."""
        self.config: DashboardConfig = config or DashboardConfig()
        self.validator: DataValidator = validator or DataValidator()
        self.notifier: Notifier = notifier or Notifier()
        
        # Initialize SessionManager for conflict-free button handling
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
        st.info("Note: Intraday intervals (e.g. '1m', '5m') typically limit history to ~60 days.")

        # Symbol input with validation (updated to match v2 approach)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symbols_input = st.text_input(
                "🎯 Ticker Symbols",
                value=", ".join(self.config.DEFAULT_SYMBOLS),
                help="Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)",
                placeholder="AAPL, MSFT, GOOGL",                key="symbol_input"
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
            self.symbols = self.validator.validate_symbols(symbols_input)
            if self.symbols != previous_symbols:
                st.session_state["data_fetched"] = False
                logger.info(f"Symbols updated: {previous_symbols} → {self.symbols}")
                
        except ValueError as e:
            st.error(f"❌ Invalid symbols: {e}")
            self.symbols = []

        self._display_symbol_status()

        # Interval selection
        self.interval = st.selectbox(
            "Data Interval",
            options=self.config.VALID_INTERVALS,
            index=0,
            help="Choose frequency of data points (daily='1d', hourly='1h', etc.)",
            key="interval_select"  # Static key
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
        """Render start and end date inputs with validation."""
        col1, col2 = st.columns(2)
        with col1:
            new_start = st.date_input(
                "Start Date",
                value=self.start_date,
                max_value=date.today(),
                help="Start date for historical data",
                key="start_date_input"  # Static key
            )
            
        with col2:
            new_end = st.date_input(
                "End Date",
                value=self.end_date,
                max_value=date.today(),
                help="End date for historical data",
                key="end_date_input"  # Static key
            )

        # Update dates and reset data_fetched if they changed
        if new_start != self.start_date or new_end != self.end_date:
            self.start_date = new_start
            self.end_date = new_end
            st.session_state["data_fetched"] = False

        if not self.validator.validate_dates(self.start_date, self.end_date):
            st.error("Invalid date range: Start date must be <= End date and not in the future.")

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
        Download OHLCV data for given symbols and date range using data_loader utility.
        Handles MultiIndex columns by flattening them.
        """
        if not symbols:
            logger.warning("No symbols provided to download.")
            return None
        data = download_stock_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            notifier=self.notifier
        )
        if not data:
            logger.warning("No data returned from download_stock_data.")
            return None

        # Flatten MultiIndex columns if present
        for symbol, df in data.items():
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join([str(lvl) for lvl in col if lvl]) for col in df.columns.values]
                data[symbol] = df
        return data

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
        if not self.validator.validate_dates(self.start_date, self.end_date):
            st.error("Cannot fetch data: Invalid date range.")
            return 0

        # Add progress indicator
        with st.spinner("Downloading data..."):
            try:
                # Add debug logging
                logger.info(f"Starting download for symbols: {self.symbols}")
                logger.info(f"Date range: {self.start_date} to {self.end_date}")
                logger.info(f"Interval: {self.interval}")

                if self.clean_old:
                    self._clean_existing_files()

                raw_df = self._download(self.symbols, self.start_date, self.end_date, self.interval)
                
                if raw_df is None:
                    st.error("❌ No data fetched. Please check the inputs or try again.")
                    logger.error("Download returned None")
                    return 0

                logger.info(f"Downloaded data for {len(raw_df)} symbols")
                
                data_dict = raw_df  # Already a dictionary of dataframes
                
                if not data_dict:
                    st.error("No valid data processed. Please check the logs or try again.")
                    return 0

                self.saved_paths = []
                successful = 0
                
                # Validate and save data for each symbol
                for symbol, df in data_dict.items():
                    # Validate OHLC structure using dashboard_utils
                    is_valid, validation_msg = validate_ohlc_dataframe(df)
                    if not is_valid:
                        logger.warning(f"Data validation failed for {symbol}: {validation_msg}")
                        st.warning(f"⚠️ Data quality issue for {symbol}: {validation_msg}")
                    
                    # Save data safely
                    if self._save_data_safely(symbol, df):
                        successful += 1
                        
                # Update session state using safe operations
                st.session_state['saved_paths'] = [str(p) for p in self.saved_paths]

                if successful > 0:
                    st.session_state["data_fetched"] = True
                    st.session_state["last_fetch_time"] = datetime.now()
                    st.session_state["fetch_count"] = st.session_state.get("fetch_count", 0) + 1
                    
                    # Simple success message without unnecessary action buttons
                    st.success(f"✅ Successfully downloaded data for {successful} symbol(s)!")
                    
                for symbol, df in data_dict.items():
                    self._display_symbol_data(symbol, df)
                return successful
                
            except Exception as e:
                logger.exception(f"Error during data fetch: {e}")
                st.error(f"❌ Download failed: {str(e)}")
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

    def run(self):
        """Main dashboard application entry point."""
        
        # Add sidebar controls using dashboard state manager
        with st.sidebar:
            st.subheader("📊 Session Stats")
            safe_streamlit_metric("Fetch Count", str(st.session_state.get("fetch_count", 0)))
            if st.session_state.get("last_fetch_time"):
                safe_streamlit_metric("Last Fetch", str(st.session_state["last_fetch_time"].strftime("%H:%M:%S")))
              # Clear cache button with SessionManager
            if self.session_manager.create_button("🗑️ Clear Cache", "clear_cache", help="Clear all cached data"):
                try:
                    clear_cache()
                    st.success("Cache cleared!")
                    import time
                    time.sleep(1)  # Brief pause before rerun
                    st.rerun()
                except Exception as e:
                    handle_streamlit_error(e, "clearing cache")
        
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
                st.balloons()  # Celebrate success!
        else:
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
                
        # Show SessionManager debug info in a sidebar expandable section
        with st.sidebar.expander("🔧 Session Debug Info", expanded=False):
            show_session_debug_info()

# Direct execution like data_dashboard_v2.py
initialize_dashboard_session_state()
dashboard = DataDashboard()
dashboard.run()
