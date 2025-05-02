"""Stock Data Dashboard Module

Streamlit dashboard for stock OHLCV download, model training, and evaluation.
Includes model visualization and dynamic hyperparameter tuning.
"""
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from data.config import DashboardConfig
from data.logger import setup_logger
from data.model_trainer import ModelTrainer, TrainingParams
from data.data_validator import DataValidator
from utils.validation import sanitize_input
from utils.io import create_zip_archive
from utils.notifier import Notifier


logger = setup_logger(__name__)


class DataDashboard:
    """
    Stock data dashboard for downloading financial data, training models,
    and visualizing trading patterns.
    
    This dashboard provides a UI for:
    1. Downloading historical stock data for specified symbols
    2. Training machine learning models on the downloaded data
    3. Visualizing model performance and trading patterns
    """
    
    def __init__(self, 
                 config: Optional[DashboardConfig] = None, 
                 validator: Optional[DataValidator] = None,
                 model_trainer: Optional[ModelTrainer] = None,
                 notifier: Optional[Notifier] = None):
        """
        Initialize the DataDashboard with dependencies.
        
        Args:
            config: Configuration settings (uses defaults if None)
            validator: Data validation component (creates new if None)
            model_trainer: Model training component (creates new if None) 
            notifier: Notification service (creates new if None)
        """
        self.config = config or DashboardConfig()
        self.validator = validator or DataValidator()
        self.notifier = notifier or Notifier()
        self._setup_directories()
        self.saved_paths: List[Path] = []
        self.model_trainer = model_trainer or ModelTrainer(self.config)
        self._init_state()
    
    def _init_state(self) -> None:
        """Initialize the dashboard's state variables with defaults."""
        self.symbols: List[str] = []
        self.interval = "1d"
        self.start_date = date.today() - timedelta(days=365)
        self.end_date = date.today()
        self.clean_old = True
        self.auto_refresh = False
        
        # Initialize session state for pagination if not present
        if "page_number" not in st.session_state:
            st.session_state.page_number = 0
        if "items_per_page" not in st.session_state:
            st.session_state.items_per_page = 10

    def _setup_directories(self) -> None:
        """Create necessary directories for data and models."""
        try:
            self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directories set up: {self.config.DATA_DIR}, {self.config.MODEL_DIR}")
        except PermissionError as e:
            logger.error(f"Permission error creating directories: {e}")
            raise RuntimeError(f"Cannot create necessary directories. Permission denied: {e}")
        except Exception as e:
            logger.error(f"Error setting up directories: {e}")
            raise RuntimeError(f"Failed to set up directories: {e}")

    def _render_inputs(self) -> None:
        """Render the user input controls for the dashboard."""
        st.info("Note: Intraday intervals are limited to 60 days.")
        
        # Symbol input with validation
        symbols_input = st.text_input(
            "Ticker Symbols (comma-separated)", 
            value=self.config.DEFAULT_SYMBOLS
        )
        try:
            self.symbols = self.validator.validate_symbols(symbols_input)
        except ValueError as e:
            st.error(f"Invalid symbols: {e}")
            self.symbols = []
            
        # Interval selection
        self.interval = st.selectbox(
            "Data Interval", 
            options=self.config.VALID_INTERVALS, 
            index=0,
            help="Time interval for data points (1d = daily, 1h = hourly, etc.)"
        )
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            self.start_date = st.date_input("Start Date", value=self.start_date)
        with col2:
            self.end_date = st.date_input("End Date", value=self.end_date)
            
        # Date validation with clear error message
        if not self.validator.validate_dates(self.start_date, self.end_date):
            st.error("Invalid date range: Start date must be before end date and not in the future.")
            
        # If using intraday data, enforce the 60-day limit
        if self.interval != "1d" and (self.end_date - self.start_date).days > self.config.MAX_INTRADAY_DAYS:
            st.warning(f"Intraday data is limited to {self.config.MAX_INTRADAY_DAYS} days. " 
                       f"Adjusting start date accordingly.")
            self.start_date = self.end_date - timedelta(days=self.config.MAX_INTRADAY_DAYS)

        # Additional options
        self.clean_old = st.checkbox(
            "🧹 Clean old CSVs before fetching?", 
            value=True,
            help="If checked, will delete existing CSV files before downloading new data"
        )
        self.auto_refresh = st.checkbox(
            "🔄 Auto-refresh every 5 minutes?", 
            value=False,
            help="Automatically refresh data every 5 minutes"
        )
        
        # Add a divider for visual clarity
        st.divider()

    @st.cache_data(ttl=3600, show_spinner=True)
    def _download(_self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Download and validate OHLCV stock data for a given symbol.

        Args:
            symbol (str): Stock ticker symbol.

        Returns:
            Optional[pd.DataFrame]: Cleaned DataFrame with columns ['open', 'high', 'low', 'close', 'volume'],
                                    or None if download/validation fails.

        Raises:
            None. All exceptions are logged and handled gracefully.
        """
        # Sanitize and validate symbol input
        symbol = sanitize_input(symbol.strip().upper())
        if not symbol or not symbol.isalnum():
            logger.warning("Invalid or empty symbol provided to download function: '%s'", symbol)
            return None

        try:
            # Add a timeout to prevent hanging on slow connections
            start_time = time.time()
            df = yf.download(
                symbol,
                start=_self.start_date.strftime("%Y-%m-%d"),
                end=_self.end_date.strftime("%Y-%m-%d"),
                interval=_self.interval,
                progress=False,
                timeout=30
            )
            download_time = time.time() - start_time
            logger.info("Downloaded data for %s in %.2fs", symbol, download_time)

            # Validate the downloaded data
            if df is None or df.empty:
                logger.warning("No data returned for %s", symbol)
                return None

            # Ensure all required columns are present (case-insensitive check)
            required_cols = {'open', 'high', 'low', 'close', 'volume'}
            df_cols_lower = {col.lower() for col in df.columns}
            if not required_cols.issubset(df_cols_lower):
                missing = required_cols - df_cols_lower
                logger.error("Missing required columns for %s: %s", symbol, missing)
                return None

            # Map the actual column names to our expected lowercase names
            col_mapping = {col: col.lower() for col in df.columns if col.lower() in required_cols}
            result_df = df[list(col_mapping.keys())].copy()
            result_df.columns = list(col_mapping.values())

            # Drop rows where all required columns are NaN
            result_df.dropna(subset=list(required_cols), how='all', inplace=True)

            # Check for excessive NaN values
            nan_percentage = result_df.isna().mean().mean() * 100
            if nan_percentage > 20:
                logger.warning("%s data has %.1f%% missing values", symbol, nan_percentage)
                # Optionally, drop rows with NaNs or impute missing values here

            # Handle timezone issues that might arise from different exchanges
            if hasattr(result_df.index, 'tzinfo') and result_df.index.tzinfo is not None:
                result_df.index = result_df.index.tz_localize(None)

            # Additional validation: ensure at least 2 rows for downstream processing
            if len(result_df) < 2:
                logger.warning("Insufficient data rows for %s after cleaning", symbol)
                return None

            # Security: Ensure no unexpected columns are present (defense in depth)
            result_df = result_df[sorted(required_cols)]

            return result_df

        except Exception as e:
            logger.exception("Error downloading data for %s: %s", symbol, str(e))
            # Send alert for critical download failures
            try:
                _self.notifier.send_alert(f"Failed to download {symbol} data: {str(e)}")
            except Exception as notify_err:
                logger.error("Notifier failed: %s", notify_err)
            return None

    def _clean_existing_files(self) -> None:
        """Remove existing CSV files from the data directory."""
        try:
            file_count = 0
            for file in self.config.DATA_DIR.glob("*.csv"):
                try:
                    file.unlink()
                    file_count += 1
                except (PermissionError, OSError) as e:
                    logger.error(f"Could not delete file {file}: {e}")
            logger.info(f"Cleaned {file_count} CSV files from {self.config.DATA_DIR}")
        except Exception as e:
            logger.exception(f"Error while cleaning existing files: {e}")
            st.error(f"Could not clean old files: {str(e)}")

    def _fetch_and_display_data(self) -> None:
        """Fetch data for all symbols and display in the UI."""
        # Clear saved paths before downloading new data
        self.saved_paths.clear()
        
        # Clean old files if requested
        if self.clean_old:
            self._clean_existing_files()

        # Check if there are symbols to process
        if not self.symbols:
            st.warning("Please enter at least one valid symbol")
            return
            
        # Process each symbol
        successful_downloads = 0
        for symbol in self.symbols:
            with st.spinner(f"Downloading data for {symbol}..."):
                df = self._download(symbol)
                
                if df is None:
                    st.error(f"No data available for {symbol}")
                    continue

                # Save data to CSV
                try:
                    # Ensure filename is safe
                    safe_symbol = sanitize_input(symbol)
                    path = self.config.DATA_DIR / f"{safe_symbol}_{self.interval}.csv"
                    
                    # Include date in index name for clarity when reading later
                    df.index.name = "date"
                    df.to_csv(path)
                    self.saved_paths.append(path)
                    successful_downloads += 1
                    
                    # Display data with pagination for large datasets
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
                        
                        # Show recent data with pagination
                        pages = len(df) // st.session_state.items_per_page + 1
                        start_idx = st.session_state.page_number * st.session_state.items_per_page
                        end_idx = min(start_idx + st.session_state.items_per_page, len(df))
                        
                        st.dataframe(df.iloc[start_idx:end_idx])
                        
                        # Add pagination controls
                        cols = st.columns([1, 1, 2])
                        if cols[0].button("Previous", key=f"prev_{symbol}") and st.session_state.page_number > 0:
                            st.session_state.page_number -= 1
                            st.experimental_rerun()
                        if cols[1].button("Next", key=f"next_{symbol}") and st.session_state.page_number < pages - 1:
                            st.session_state.page_number += 1
                            st.experimental_rerun()
                        cols[2].write(f"Page {st.session_state.page_number + 1} of {pages}")
                    
                    with tab2:
                        # Interactive chart
                        st.line_chart(df["close"])
                        
                except Exception as e:
                    logger.exception(f"Error saving or displaying data for {symbol}: {e}")
                    st.error(f"Error processing data for {symbol}: {str(e)}")

        # Offer batch download if multiple files were saved
        if len(self.saved_paths) > 1:
            try:
                zip_data = create_zip_archive(self.saved_paths)
                st.download_button(
                    "📦 Download All as ZIP",
                    data=zip_data,
                    file_name=f"stock_data_{date.today().isoformat()}.zip",
                    mime="application/zip"
                )
                st.success(f"Successfully downloaded data for {successful_downloads} symbols")
            except Exception as e:
                logger.exception(f"Error creating ZIP archive: {e}")
                st.error("Could not create ZIP archive for download")

    def _render_model_training_ui(self) -> TrainingParams:
        """Render the model training UI controls.
        
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
            
            # Additional parameters could be added here
        
        # Create and return the parameters object
        return TrainingParams(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )

    def _train_models(self, params: TrainingParams) -> None:
        """Train models for each downloaded dataset.
        
        Args:
            params: Training parameters
        """
        if not self.saved_paths:
            st.warning("No data available for training. Please download data first.")
            return
            
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
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
                
                if df.empty:
                    st.error(f"Empty dataset for {symbol}, skipping training")
                    continue
                    
                # Train the model
                with st.spinner(f"Training model for {symbol}..."):
                    model, metrics, cm, report = self.model_trainer.train_model(df, params)
                    model_path = self.model_trainer.save_model(model, symbol, self.interval)
                
                # Display success message and results
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
                    
            except Exception as e:
                logger.exception(f"Error training model for {path.stem}: {e}")
                st.error(f"Training failed for {path.stem}: {str(e)}")
        
        # Complete progress bar
        progress_bar.progress(1.0)
        status_text.text("Training complete!")
        
        # Notify about completion
        if len(self.saved_paths) > 0:
            self.notifier.send_notification(
                f"Training completed for {len(self.saved_paths)} models"
            )

    def _handle_user_actions(self) -> None:
        """Process user actions from the UI."""
        # Fetch data button
        fetch_col, refresh_col = st.columns([3, 1])
        if fetch_col.button("📥 Fetch Data", use_container_width=True):
            self._fetch_and_display_data()
            
        # Implement auto-refresh if enabled
        if self.auto_refresh:
            refresh_time = int(time.time()) % (5 * 60)  # 5-minute cycle
            refresh_col.caption(f"Auto-refreshing in: {300 - refresh_time}s")
            
            # If we're at the beginning of the cycle, refresh
            if refresh_time < 5:  # Within first 5 seconds of cycle
                self._fetch_and_display_data()

        # Model training section
        if st.checkbox("📚 Train models after fetching?", value=False):
            training_params = self._render_model_training_ui()
            
            if st.button("🧠 Train Models", use_container_width=True):
                self._train_models(training_params)

    def render_dashboard(self) -> None:
        """Render the complete dashboard UI."""
        try:
            st.set_page_config(
                page_title="Stock Data Dashboard",
                page_icon="📈",
                layout="centered",
                initial_sidebar_state="expanded"
            )
            
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

if __name__ == "__main__":
    try:
        dashboard = DataDashboard()
        dashboard.render_dashboard()
    except Exception as e:
        logging.exception("Critical error in Data Dashboard")
        st.error(f"Critical error: {str(e)}")
