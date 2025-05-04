import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
import streamlit as st

from data.model_trainer import ModelTrainer, TrainingParams
from data.config import DashboardConfig
from utils.stock_validation import validate_ticker, get_valid_tickers  # If you want direct ticker validation
from data.data_validator import DataValidator
from utils.validation import sanitize_input             # For input validation and sanitization
from utils.notifier import Notifier
from data.data_loader import (
    download_stock_data,
    process_downloaded_data,   # Use this instead of duplicating logic
    save_to_csv,               # Use for file saving
    clear_cache                # For cache clearing UI
)
from utils.io import create_zip_archive  # For batch ZIP downloads

# Configure logging for the dashboard module
logging.basicConfig(
    format="%(asctime)s %(levelname)s [DataDashboard]: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def handle_streamlit_exception(method):
    """
    Decorator to catch exceptions in Streamlit callbacks and methods.
    Logs the exception and displays an error message in the UI.
    """
    import functools
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {method.__name__}: {e}")
            st.error(f"An error occurred: {e}")
            return None
    return wrapper

class DataDashboard:
    """
    Streamlit dashboard for downloading stock data and training models.

    Features:
    - Robust input validation and error handling
    - Modular design for maintainability and extensibility
    - Secure handling of user inputs and file operations
    - Auto-refresh and session state management
    - Model training with progress feedback and results display
    """

    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        validator: Optional[DataValidator] = None,
        model_trainer: Optional[ModelTrainer] = None,
        notifier: Optional[Notifier] = None
    ):
        """Initialize the dashboard with configuration and dependencies."""
        self.config: DashboardConfig = config or DashboardConfig()
        self.validator: DataValidator = validator or DataValidator()
        self.model_trainer: ModelTrainer = model_trainer or ModelTrainer(self.config)
        self.notifier: Notifier = notifier or Notifier()

        self._setup_directories()
        self.saved_paths: List[Path] = []
        self.symbols: List[str] = []
        self.start_date: date = date.today() - timedelta(days=365)
        self.end_date: date = date.today()
        self.interval: str = "1d"
        self.clean_old: bool = True
        self.auto_refresh: bool = False

        logger.info("Initialized DataDashboard with default state variables.")
        self._init_session_state()
        logger.debug("Session state initialized with defaults.")

        if 'saved_paths' in st.session_state:
            self.saved_paths = st.session_state['saved_paths']
            logger.info(f"Restored {len(self.saved_paths)} saved data file paths from previous session.")

    def _init_session_state(self):
        """Ensure required session state variables exist with default values."""
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

        # Symbol input with sanitization and validation
        symbols_input = st.text_input(
            "Ticker Symbols (comma-separated)",
            value=", ".join(self.config.DEFAULT_SYMBOLS),
            help="Enter one or more stock ticker symbols, separated by commas."
        )
        previous_symbols = self.symbols.copy()
        try:
            self.symbols = self.validator.validate_symbols(symbols_input)
        except ValueError as e:
            st.error(f"Invalid symbols: {e}")
            self.symbols = []
        else:
            if self.symbols != previous_symbols:
                st.session_state["data_fetched"] = False
                logger.info(f"Symbols changed from {previous_symbols} to {self.symbols}. Reset data_fetched flag.")

        self._show_symbol_status()

        # Interval selection
        self.interval = st.selectbox(
            "Data Interval",
            options=self.config.VALID_INTERVALS,
            index=0,
            help="Choose frequency of data points (daily='1d', hourly='1h', etc.)"
        )

        # Date range selection with validation
        self._render_date_inputs()

        # Options: clean old data files and auto-refresh
        col1, col2 = st.columns(2)
        with col1:
            self.clean_old = st.checkbox(
                "Clean old CSVs before fetching?",
                value=self.clean_old,
                help="Delete previously saved CSV files for these symbols before downloading new data."
            )
        with col2:
            self.auto_refresh = st.checkbox(
                "Auto-refresh every 5 minutes?",
                value=self.auto_refresh,
                help="Automatically refresh data every 5 minutes (for live data monitoring)."
            )
        st.divider()

    def _show_symbol_status(self):
        """Display a status message about the currently selected symbols (if any)."""
        if self.symbols:
            st.success(f"Selected Stocks: {', '.join(self.symbols)}")
        else:
            st.info("No valid stock symbols selected yet.")

    def _render_date_inputs(self):
        """Render start and end date inputs with validation, and handle changes."""
        col1, col2 = st.columns(2)
        with col1:
            new_start = st.date_input(
                "Start Date",
                value=self.start_date,
                max_value=date.today(),
                help="Start date for historical data"
            )
            if new_start != self.start_date:
                self.start_date = new_start
                st.session_state["data_fetched"] = False
        with col2:
            new_end = st.date_input(
                "End Date",
                value=self.end_date,
                max_value=date.today(),
                help="End date for historical data"
            )
            if new_end != self.end_date:
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

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def _download(symbols: List[str], start_date: date, end_date: date, interval: str, _notifier=None) -> Optional[Dict[str, pd.DataFrame]]:
        """Download OHLCV data for given symbols and date range using data_loader utility."""
        if not symbols:
            logger.warning("No symbols provided to download.")
            return None
        data = download_stock_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            notifier=_notifier
        )
        if not data:
            logger.warning("No data returned from download_stock_data.")
            return None
        return data

    @handle_streamlit_exception
    def _fetch_and_display_data(self):
        """Fetch data for current symbols and date range, then display it in the app."""
        if not self.symbols:
            st.error("Please enter at least one valid stock symbol.")
            return 0
        if not self.validator.validate_dates(self.start_date, self.end_date):
            st.error("Cannot fetch data: Invalid date range.")
            return 0

        if self.clean_old:
            self._clean_existing_files()

        raw_df = self._download(self.symbols, self.start_date, self.end_date, self.interval, _notifier=self.notifier)
        if raw_df is None:
            st.error("No data fetched. Please check the inputs or try again.")
            return 0

        # raw_df is already a dictionary of dataframes, so we can use it directly
        data_dict = raw_df  # Skip processing since raw_df is already in the right format
        
        if not data_dict:
            st.error("No valid data processed. Please check the logs or try again.")
            return 0

        self.saved_paths = []
        successful = 0
        for symbol, df in data_dict.items():
            file_path = self.config.DATA_DIR / f"{symbol}_{self.interval}.csv"
            try:
                df.to_csv(file_path, index=True)
                self.saved_paths.append(file_path)
                successful += 1
                logger.info(f"Data for {symbol} saved to {file_path}")
            except Exception as e:
                logger.exception(f"Error saving data for {symbol}: {e}")
                st.error(f"⚠️ Failed to save data for {symbol}: {e}")
        st.session_state['saved_paths'] = self.saved_paths

        if successful > 0:
            st.session_state["data_fetched"] = True
            st.session_state["last_fetch_time"] = datetime.now()
            st.session_state["fetch_count"] += 1
            st.success(f"✅ Downloaded data for {successful} symbol(s).")
        for symbol, df in data_dict.items():
            self._display_symbol_data(symbol, df)
        return successful

    def _display_symbol_data(self, symbol: str, df: pd.DataFrame):
        """Display the downloaded data for a given symbol in the Streamlit app."""
        st.subheader(f"Data Preview: {symbol}")
        tab1, tab2 = st.tabs(["📋 Recent Data", "📈 Chart"])
        with tab1:
            st.write(df.tail(10))
            st.caption(f"{len(df)} records from {df.index.min().date()} to {df.index.max().date()}")
        with tab2:
            try:
                st.line_chart(df['close'])
            except Exception as e:
                st.write("Unable to plot chart:", e)

    def _render_model_training_ui(self) -> TrainingParams:
        """Render the model training parameter inputs and return a TrainingParams object."""
        st.subheader("Model Training Configuration")
        n_estimators = st.slider(
            "Number of Trees (Estimators)", min_value=10, max_value=500, value=100, step=10,
            help="Number of trees in the Random Forest model."
        )
        min_samples_split = st.slider(
            "Min Samples Split", min_value=2, max_value=50, value=2, step=1,
            help="Minimum samples required to split an internal node."
        )
        params = TrainingParams(n_estimators=n_estimators, min_samples_split=min_samples_split)
        return params

    def _display_training_results(self, symbol: str, metrics: dict, cm: Any, report: str, model_path: Path):
        """Display the results of training (metrics, confusion matrix, etc.) for one symbol."""
        st.subheader(f"Results for {symbol} model")
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f"{symbol} Confusion Matrix")
            st.pyplot(fig)
        except Exception as e:
            logger.warning(f"Could not display confusion matrix: {e}")
        if report:
            st.text(report)
        if metrics:
            for m, val in metrics.items():
                st.write(f"**{m}:** {val}")
        st.caption(f"Model saved to: {model_path}")

    @handle_streamlit_exception
    def _train_models(self, params: TrainingParams):
        """Train a model for each downloaded dataset in saved_paths using the given parameters."""
        logger.info(f"_train_models called with params: {params}")
        if not self.saved_paths:
            st.error("No data available for training. Please download data first.")
            logger.warning("No data available for training (empty saved_paths)")
            return
        st.info("Training models... This may take a moment.")
        
        logger.info(f"Starting training for {len(self.saved_paths)} data files")
        progress_bar = st.progress(0)
        status_text = st.empty()

        trained_models = []
        for idx, data_path in enumerate(self.saved_paths):
            progress = (idx / len(self.saved_paths))
            progress_bar.progress(progress)
            symbol = data_path.name.split('_')[0]
            logger.info(f"Processing file {data_path} for symbol {symbol}")
            status_text.text(f"Training model for {symbol}...")
            try:
                df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                logger.info(f"Read {len(df)} rows of data for {symbol}")
                logger.info(f"Calling train_model with data shape: {df.shape}")
                model, metrics, cm, report = self.model_trainer.train_model(df, params)
                model_path = self.model_trainer.save_model(model, symbol, self.interval)
                trained_models.append(symbol)
                logger.info(f"Model trained for {symbol} and saved to {model_path}")
                self._display_training_results(symbol, metrics, cm, report, model_path)
            except Exception as e:
                logger.error(f"Training failed for {symbol}: {e}", exc_info=True)
                st.error(f"🚫 Training failed for {symbol}: {e}")
        
        progress_bar.progress(1.0)
        status_text.text(f"Training complete! Trained {len(trained_models)}/{len(self.saved_paths)} models.")

    def _handle_auto_refresh(self):
        """Automatically refresh data if the auto_refresh flag is True and interval elapsed."""
        if self.auto_refresh and st.session_state.get("data_fetched"):
            last_fetch = st.session_state.get("last_fetch_time")
            if last_fetch:
                elapsed = datetime.now() - last_fetch
                REFRESH_INTERVAL = 300  # 5 minutes in seconds
                if elapsed.total_seconds() >= REFRESH_INTERVAL:
                    st.info("⏳ Auto-refreshing data...")
                    self._fetch_and_display_data()

    def run(self):
        """
        Main method to run the dashboard UI and handle user interactions.

        - Renders input controls and handles user actions
        - Displays fetched data and model training section
        - Handles auto-refresh and session state
        """
        self._render_inputs()
        fetch_clicked = st.button("Download Data", type="primary", use_container_width=True)
        if fetch_clicked:
            self._fetch_and_display_data()
        else:
            if st.session_state.get("data_fetched") and self.saved_paths:
                for path in self.saved_paths:
                    symbol = path.name.split('_')[0]
                    try:
                        df = pd.read_csv(path, index_col=0, parse_dates=True)
                    except Exception as e:
                        logger.error(f"Failed to read cached data for {symbol}: {e}")
                        continue
                    self._display_symbol_data(symbol, df)
        self._handle_auto_refresh()

        # Modified condition: check only for saved_paths, not data_fetched flag
        if self.saved_paths:
            st.divider()
            st.header("Model Training")
            logger.info(f"Showing model training UI for {len(self.saved_paths)} saved data files")
            training_params = self._render_model_training_ui()
            if st.button("Train Model", type="primary", use_container_width=True):
                logger.info("Train Model button clicked, starting training process...")
                self._train_models(training_params)

if __name__ == "__main__" or st._is_running_with_streamlit:
    dashboard = DataDashboard()
    dashboard.run()