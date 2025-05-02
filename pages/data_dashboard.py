"""Stock Data Dashboard Module

Streamlit dashboard for stock OHLCV download, model training, and evaluation.
Includes model visualization and dynamic hyperparameter tuning.
"""
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import functools

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from data.config import DashboardConfig
from data.logger import setup_logger
from data.model_trainer import ModelTrainer, TrainingParams
from data.data_validator import DataValidator
from utils.validation import sanitize_input
from utils.io import create_zip_archive
from utils.notifier import Notifier
from data.data_loader import download_stock_data, clear_cache


# Configure logger
logger = setup_logger(__name__)


def handle_streamlit_exception(method):
    """Decorator to handle exceptions in Streamlit methods.
    
    Catches exceptions, logs them, and displays user-friendly error messages.
    
    Args:
        method: The method to wrap with exception handling
        
    Returns:
        Wrapped method with exception handling
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {method.__name__}: {e}")
            st.error(f"An error occurred: {str(e)}")
            return None
    return wrapper

class DataDashboard:
    """
    Stock data dashboard for downloading financial data, training models,
    and visualizing trading patterns.

    Provides a UI for:
    1. Downloading historical stock data.
    2. Training ML models on the data.
    3. Visualizing model performance.
    """

    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        validator: Optional[DataValidator] = None,
        model_trainer: Optional[ModelTrainer] = None,
        notifier: Optional[Notifier] = None,
    ) -> None:
        """
        Initialize the DataDashboard with dependencies.

        Injects dependencies or creates default instances. Sets up necessary
        directories and initializes Streamlit session state.

        Args:
            config: Configuration settings object.
            validator: Data validation component.
            model_trainer: Model training component.
            notifier: Notification service.

        Raises:
            OSError: If required directories cannot be created.
            Exception: For other initialization errors.
        """
        try:
            # 1. Initialize Configuration (Highest Priority)
            self.config: DashboardConfig = config or DashboardConfig()
            logger.info("DashboardConfig initialized.")

            # 2. Initialize Core Components
            self.validator: DataValidator = validator or DataValidator()
            self.notifier: Notifier = notifier or Notifier()
            self.model_trainer: ModelTrainer = model_trainer or ModelTrainer(self.config)
            logger.info("Core components initialized.")

            # 3. Setup Environment
            self._setup_directories()

            # 4. Initialize State Variables
            self.saved_paths: List[Path] = []
            self.symbols: List[str] = []
            self.start_date: date = date.today() - timedelta(days=365)
            self.end_date: date = date.today()
            self.interval: str = "1d"
            self.clean_old: bool = True
            self.auto_refresh: bool = False
            logger.info("State variables initialized.")

            # 5. Initialize Streamlit Session State
            self._init_state()
            logger.info("Streamlit session state initialized.")

        except OSError as e:
            logger.critical(f"Failed to setup directories: {e}", exc_info=True)
            st.error(f"Fatal Error: Could not create required directories ({e})")
            raise
        except Exception as e:
            logger.critical(f"Failed to initialize DataDashboard: {e}", exc_info=True)
            st.error(f"Fatal Error: Initialization failed ({e})")
            raise

    def _init_state(self) -> None:
        """Initialize Streamlit session state variables if they don't exist."""
        session_defaults = {
            "data_fetched": False,
            "items_per_page": 20,
            "last_fetch_time": None,
            "fetch_count": 0,
        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                
        logger.debug("Session state initialized.")

    def _setup_directories(self) -> None:
        """Create necessary data and model directories if they don't exist."""
        try:
            self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directories created: {self.config.DATA_DIR}, {self.config.MODEL_DIR}")
        except OSError as e:
            error_msg = f"Failed to create directory {e.filename}: {e.strerror}"
            logger.error(error_msg, exc_info=True)
            raise OSError(error_msg)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise

    @handle_streamlit_exception
    def _render_inputs(self) -> None:
        """Render the user input controls for the dashboard."""
        st.info("Note: Intraday intervals are limited to 60 days.")
        
        # Symbol input with validation
        symbols_input = st.text_input(
            "Ticker Symbols (comma-separated)", 
            value=self.config.DEFAULT_SYMBOLS
        )
        
        try:
            # Validate symbols
            previous_symbols = self.symbols.copy() if hasattr(self, 'symbols') else []
            self.symbols = self.validator.validate_symbols(symbols_input)
            
            # Reset session state if symbols have changed
            if previous_symbols != self.symbols and previous_symbols:
                st.session_state["data_fetched"] = False
                logger.info(f"Symbols changed from {previous_symbols} to {self.symbols}")
        except ValueError as e:
            st.error(f"Invalid symbols: {e}")
            self.symbols = []

        # Show currently selected stocks
        self._show_symbol_status()
            
        # Interval selection
        self.interval = st.selectbox(
            "Data Interval", 
            options=self.config.VALID_INTERVALS, 
            index=self.config.VALID_INTERVALS.index(self.interval) if self.interval in self.config.VALID_INTERVALS else 0,
            help="Time interval for data points (1d = daily, 1h = hourly, etc.)"
        )
        
        # Date range selection
        self._render_date_inputs()

        # Additional options
        self._render_options()
        
        # Add a divider for visual clarity
        st.divider()

        # Show query parameters for debugging/confirmation
        if self.symbols:
            st.info(f"Query: symbols={self.symbols}, timeframe={self.start_date} to {self.end_date}, interval={self.interval}")

    def _show_symbol_status(self) -> None:
        """Display the status of loaded symbols."""
        if self.symbols:
            st.success(f"Loaded stocks: {', '.join(self.symbols)}")
        else:
            st.info("No valid stocks loaded yet.")

    def _render_date_inputs(self) -> None:
        """Render and validate date inputs."""
        col1, col2 = st.columns(2)
        
        today = date.today()
        
        with col1:
            new_start_date = st.date_input("Start Date", value=self.start_date)
            if new_start_date != self.start_date:
                self.start_date = new_start_date
                # Reset fetch state if date changed
                st.session_state["data_fetched"] = False
                
        with col2:
            new_end_date = st.date_input("End Date", value=self.end_date, max_value=today)
            if new_end_date != self.end_date:
                self.end_date = new_end_date
                # Reset fetch state if date changed
                st.session_state["data_fetched"] = False
            
        # Date validation with clear error messages
        if self.end_date > today:
            st.error(f"End date cannot be in the future. Please select a date on or before {today}.")
            self.end_date = today  # Automatically fix the end date
            
        if self.start_date >= self.end_date:
            st.error("Invalid date range: Start date must be before end date.")
            
        # If using intraday data, enforce the limit
        if self.interval != "1d" and (self.end_date - self.start_date).days > self.config.MAX_INTRADAY_DAYS:
            st.warning(
                f"Intraday data is limited to {self.config.MAX_INTRADAY_DAYS} days. " 
                f"Adjusting start date accordingly."
            )
            self.start_date = self.end_date - timedelta(days=self.config.MAX_INTRADAY_DAYS)

    def _render_options(self) -> None:
        """Render additional options for data fetching."""
        col1, col2 = st.columns(2)
        
        with col1:
            self.clean_old = st.checkbox(
                "🧹 Clean old CSVs before fetching?", 
                value=self.clean_old,
                help="If checked, will delete existing CSV files before downloading new data"
            )
            
        with col2:
            self.auto_refresh = st.checkbox(
                "🔄 Auto-refresh every 5 minutes?", 
                value=self.auto_refresh,
                help="Automatically refresh data every 5 minutes"
            )
            
            # Add clear cache button
            if st.button("🧹 Clear Data Cache", help="Clear cached stock data"):
                from data.data_loader import clear_cache
                clear_cache()
                st.success("Data cache cleared!")

    @handle_streamlit_exception
    def _clean_existing_files(self) -> None:
        """Remove existing CSV files from the data directory."""
        try:
            file_count = 0
            for file in self.config.DATA_DIR.glob("*.csv"):
                try:
                    # Ensure we're only deleting CSV files in our data directory
                    if file.is_file() and file.parent.resolve() == self.config.DATA_DIR.resolve():
                        file.unlink()
                        file_count += 1
                except (PermissionError, OSError) as e:
                    logger.error(f"Could not delete file {file}: {e}")
                    
            logger.info(f"Cleaned {file_count} CSV files from {self.config.DATA_DIR}")
            
            if file_count > 0:
                st.success(f"Cleaned {file_count} existing data files")
        except Exception as e:
            logger.exception(f"Error cleaning files: {e}")
            st.error(f"Could not clean old files: {str(e)}")

    @handle_streamlit_exception
    def _fetch_and_display_data(self) -> None:
        """Fetch data for all symbols and display in the UI."""
        self.saved_paths.clear()
        
        # Ensure end date is not in the future - do this right before fetching
        today = date.today()
        if self.end_date > today:
            self.end_date = today
            st.warning(f"Adjusted end date to today ({today}) as future dates are not available.")
        
        # Block with invalid dates
        if not self.validator.validate_dates(self.start_date, self.end_date):
            st.error("Cannot fetch data with invalid date range.")
            return
            
        # Check for symbols
        if not self.symbols:
            st.warning("Please enter at least one valid symbol")
            return
            
        # Show spinner during operation
        with st.spinner("Downloading stock data..."):
            if self.clean_old:
                self._clean_existing_files()
            logger.info(f"Fetching data with dates: {self.start_date} to {self.end_date}")
            data_dict = download_stock_data(
                self.symbols,
                self.start_date,
                self.end_date,
                self.interval,
                notifier=self.notifier
            )

        # === guard against no data ===
        if data_dict is None or (isinstance(data_dict, dict) and not data_dict):
            st.error("No data returned for those symbols – please check ticker symbols and date range.")
            return

        # === save & display ===
        count = self._save_and_display_data(data_dict)
        if count == 0:
            st.warning("Downloaded data but nothing matched your columns – check logs for missing fields.")
        else:
            st.success(f"✅ Data fetched for {count}/{len(self.symbols)} symbols.")
            st.session_state["last_fetch_time"] = datetime.now()
            st.session_state["fetch_count"] += 1

    def _save_and_display_data(self, data_dict: Dict[str, pd.DataFrame]) -> int:
        """
        Save downloaded data to files and display in the UI.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame
            
        Returns:
            Number of successfully saved symbols
        """
        successful_downloads = 0
        
        for symbol, df in data_dict.items():
            try:
                # Ensure filename is safe and predictable
                safe_symbol = sanitize_input(symbol)
                path = self.config.DATA_DIR / f"{safe_symbol}_{self.interval}.csv"
                
                # Include date in index name for clarity when reading later
                df.index.name = "date"
                df.to_csv(path)
                self.saved_paths.append(path)
                successful_downloads += 1
                
                # Display the data
                self._display_symbol_data(symbol, df)
                
            except Exception as e:
                logger.exception(f"Error saving/displaying data for {symbol}: {e}")
                st.error(f"Error processing {symbol}: {str(e)}")
                
        return successful_downloads

    def _display_symbol_data(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Display data for a single symbol with tabs for different views.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with price data
        """
        st.subheader(f"📊 {symbol}")
        
        # Create tabs for different data views
        tab1, tab2 = st.tabs(["Recent Data", "Chart"])
        
        with tab1:
            # Show basic statistics
            st.caption(f"{len(df)} rows from {df.index.min()} to {df.index.max()}")
            
            # Date range and statistics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            metrics_col1.metric("Average Volume", f"{df['volume'].mean():,.0f}")
            metrics_col2.metric("Price Range", f"${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            # Calculate returns if enough data
            if len(df) > 1:
                returns = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
                metrics_col3.metric("Period Return", f"{returns:.2f}%")
            
            # Show paginated data
            self._display_paginated_data(symbol, df)
        
        with tab2:
            # Interactive chart
            st.line_chart(df["close"])

    def _display_paginated_data(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Display paginated data for a symbol.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with price data
        """
        # Ensure page key exists
        page_key = f"page_number_{symbol}"
        if page_key not in st.session_state:
            st.session_state[page_key] = 0

        # Calculate pagination values
        items_per_page = st.session_state.items_per_page
        total_pages = max(1, (len(df) + items_per_page - 1) // items_per_page)
        
        # Ensure page number is valid (could be invalid after new data with fewer rows)
        if st.session_state[page_key] >= total_pages:
            st.session_state[page_key] = total_pages - 1
            
        # Calculate start and end indices
        start_idx = st.session_state[page_key] * items_per_page
        end_idx = min(start_idx + items_per_page, len(df))

        # Display the data for current page
        st.dataframe(df.iloc[start_idx:end_idx])

        # Add pagination controls
        cols = st.columns([1, 1, 2])
        
        # Previous button
        prev_disabled = st.session_state[page_key] <= 0
        if cols[0].button("⬅️ Previous", key=f"prev_{symbol}", disabled=prev_disabled) and not prev_disabled:
            st.session_state[page_key] -= 1
            st.experimental_rerun()
            
        # Next button
        next_disabled = st.session_state[page_key] >= total_pages - 1
        if cols[1].button("Next ➡️", key=f"next_{symbol}", disabled=next_disabled) and not next_disabled:
            st.session_state[page_key] += 1
            st.experimental_rerun()
            
        # Page indicator
        cols[2].write(f"Page {st.session_state[page_key] + 1} of {total_pages}")

    def _offer_batch_download(self, successful_downloads: int) -> None:
        """
        Offer a batch download of all data files.
        
        Args:
            successful_downloads: Number of successfully downloaded symbols
        """
        if len(self.saved_paths) > 1:
            try:
                zip_data = create_zip_archive(self.saved_paths)
                st.download_button(
                    "📦 Download All as ZIP",
                    data=zip_data,
                    file_name=f"stock_data_{date.today().isoformat()}.zip",
                    mime="application/zip",
                    help="Download all fetched data as a ZIP archive"
                )
                st.success(f"Successfully downloaded data for {successful_downloads} symbols")
            except Exception as e:
                logger.exception(f"Error creating ZIP archive: {e}")
                st.error("Could not create ZIP archive for download")

    @handle_streamlit_exception
    def _render_model_training_ui(self) -> TrainingParams:
        """
        Render the model training UI controls.
        
        Returns:
            TrainingParams object with the selected parameters
        """
        st.subheader("📚 Model Training Parameters")
        
        # Organize parameters into columns
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider(
                "Number of Estimators",
                min_value=10,
                max_value=300,
                value=100,
                step=10,
                help="Number of trees in the forest"
            )
            
            max_depth = st.slider(
                "Max Depth", 
                min_value=1,
                max_value=20,
                value=10,
                help="Maximum depth of each tree"
            )
        
        with col2:
            min_samples_split = st.slider(
                "Min Samples Split",
                min_value=2,
                max_value=20,
                value=10,
                help="Minimum samples required to split a node"
            )
            
        # Create and return the parameters object
        return TrainingParams(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )

    @handle_streamlit_exception
    def _train_models(self, params: TrainingParams) -> None:
        """
        Train models for each downloaded dataset.
        
        Args:
            params: Training parameters
        """
        if not self.saved_paths:
            st.warning("No data available for training. Please download data first.")
            return
            
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        trained_models = []
        
        for idx, path in enumerate(self.saved_paths):
            # Update progress
            progress_value = idx / len(self.saved_paths)
            progress_bar.progress(progress_value)
            
            try:
                # Extract symbol from filename
                symbol = path.stem.split("_", 1)[0]
                status_text.text(f"Training model for {symbol}...")
                
                # Read the data
                df = pd.read_csv(path, index_col="date", parse_dates=True)
                
                # Validate dataset
                if df.empty or len(df) < 30:  # Minimum rows for meaningful training
                    st.error(f"Insufficient data for {symbol} ({len(df)} rows), skipping training")
                    continue
                
                missing_cols = {"open", "high", "low", "close", "volume"} - set(df.columns)
                if missing_cols:
                    st.error(f"Missing columns in {symbol} data: {missing_cols}")
                    continue
                    
                # Train the model
                with st.spinner(f"Training model for {symbol}..."):
                    model, metrics, cm, report = self.model_trainer.train_model(df, params)
                    model_path = self.model_trainer.save_model(model, symbol, self.interval)
                    trained_models.append(symbol)
                
                # Display results
                self._display_training_results(symbol, metrics, cm, report, model_path)
                    
            except Exception as e:
                logger.exception(f"Error training model for {path.stem}: {e}")
                st.error(f"Training failed for {path.stem}: {str(e)}")
        
        # Complete progress
        progress_bar.progress(1.0)
        status_text.text(f"Training complete! Trained {len(trained_models)}/{len(self.saved_paths)} models.")
        
        # Notify about completion if any models were trained
        if trained_models:
            try:
                self.notifier.send_notification(
                    f"Training completed for {len(trained_models)} models: {', '.join(trained_models)}"
                )
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")

    def _display_training_results(
        self, 
        symbol: str, 
        metrics: Dict[str, float],
        cm: Any,  # Confusion matrix
        report: str,
        model_path: Path
    ) -> None:
        """
        Display training results for a model.
        
        Args:
            symbol: Stock symbol
            metrics: Performance metrics
            cm: Confusion matrix
            report: Classification report text
            model_path: Path where model was saved
        """
        st.success(f"✅ {symbol} model trained successfully")
        
        # Create expander for detailed results
        with st.expander(f"📊 Results for {symbol}", expanded=True):
            # Show metrics
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame([metrics])
            st.dataframe(metrics_df)
            
            # Show confusion matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            st.pyplot(fig)
            
            # Show classification report
            st.subheader("Classification Report")
            st.text(report)
            
            # Show model path
            st.caption(f"Model saved to: {model_path}")

    def _handle_user_actions(self) -> None:
        """Handle user button interactions and auto-refresh."""
        fetch_col, refresh_col = st.columns([3, 1])
        
        # Fetch button - triggers data download
        if fetch_col.button("📥 Fetch Data", use_container_width=True):
            st.session_state["data_fetched"] = True
            
        # Auto-refresh logic
        self._handle_auto_refresh()

        # Display data if it's been fetched
        if st.session_state.get("data_fetched", False):
            self._fetch_and_display_data()
            
            # If we have data, show model training section
            if self.saved_paths:
                st.divider()
                st.header("🤖 Model Training")
                
                params = self._render_model_training_ui()
                if st.button("🧠 Train Models", use_container_width=True):
                    self._train_models(params)

    def _handle_auto_refresh(self) -> None:
        """Handle auto-refresh logic based on time intervals."""
        if self.auto_refresh and st.session_state.get("data_fetched"):
            last_fetch = st.session_state.get("last_fetch_time")
            
            if last_fetch:
                # Calculate time since last refresh
                now = datetime.now()
                elapsed = now - last_fetch
                
                # Auto-refresh every REFRESH_INTERVAL seconds
                if elapsed.total_seconds() >= self.config.REFRESH_INTERVAL:
                    st.info(f"Auto-refreshing data (last refresh: {elapsed.total_seconds():.0f}s ago)")
                    self._fetch_and_display_data()
                else:
                    # Show countdown
                    remaining = self.config.REFRESH_INTERVAL - elapsed.total_seconds()
                    st.caption(f"Next auto-refresh in {int(remaining)} seconds")

    @handle_streamlit_exception
    def render_dashboard(self) -> None:
        """Render the complete dashboard UI."""
        try:
            # Configure page
            st.set_page_config(
                page_title="Stock Data Dashboard",
                page_icon="📈",
                layout="centered",
                initial_sidebar_state="expanded"
            )
            
            # Header
            st.title("📈 Stock OHLCV Data Downloader and Trainer")
            
            # Add some explanatory text
            st.markdown("""
            This dashboard allows you to:
            1. Download historical stock price data
            2. Train machine learning models on the data
            3. Visualize model performance
            """)
            
            # Render the input controls
            self._render_inputs()
            
            # Handle user interactions
            self._handle_user_actions()
            
        except Exception as e:
            logger.exception(f"Error rendering dashboard: {e}")
            st.error(f"An error occurred while rendering the dashboard: {str(e)}")
            st.info("Please check the logs for more details or contact support.")


# Entry point
if __name__ == "__main__":
    try:
        dashboard = DataDashboard()
        dashboard.render_dashboard()
    except Exception as e:
        logging.exception("Critical error in Data Dashboard")
        st.error(f"Critical error: {str(e)}")
