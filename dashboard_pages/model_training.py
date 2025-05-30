import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, NamedTuple, List
from datetime import datetime
from dataclasses import dataclass, asdict
from pydantic import BaseModel
import torch
import json

# Dashboard logger setup
from utils.logger import get_dashboard_logger
logger = get_dashboard_logger(__name__)

# Utilities
from utils.preprocessing_config import save_preprocessing_config

# Import the enhanced data validator
from core.data_validator import (
    validate_dataframe, 
    DataFrameValidationResult,
    ValidationResult,
    get_global_validator
)

# Define backwards-compatible DataValidationResult
from typing import NamedTuple
class DataValidationResult(NamedTuple):
    is_valid: bool
    error_message: Optional[str]
    stats: Optional[Dict[str, Any]] = None

# Deep learning pipeline
from train.model_training_pipeline import MLPipeline
from train.model_manager import ModelManager, ModelMetadata
from patterns.patterns_nn import PatternNN
from utils.technicals.performance_utils import st_error_boundary, generate_combined_signals
from utils.synthetic_trading_data import add_to_model_training_ui, generate_synthetic_data
from train.ml_config import MLConfig
from utils.config.stockticker_yahoo_validation import get_valid_tickers
from train.deeplearning_trainer import train_pattern_model

# Classic ML pipeline
from train.ml_trainer import ModelTrainer, TrainingParams
from patterns.patterns import CandlestickPatterns, create_pattern_detector
from sklearn.base import BaseEstimator
from core.dashboard_utils import (
    initialize_dashboard_session_state,
    setup_page,
    handle_streamlit_error
)
from core.session_manager import create_session_manager, show_session_debug_info
from utils.technicals.technical_analysis import TechnicalAnalysis

# Initialize the page (setup_page returns a logger, but we already have one)
setup_page(
    title="🤖 Model Training Dashboard",
    logger_name=__name__,
    sidebar_title="Training Configuration"
)

# Initialize SessionManager for conflict-free widget handling
session_manager = create_session_manager("model_training")

@dataclass
class TrainingConfigUnified:
    backend: str
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 10
    cv_folds: int = 5

    def as_dict(self):
        return asdict(self)

    def validate(self) -> Tuple[bool, Optional[str]]:
        if self.backend.startswith("Deep"):
            if not 1 <= self.epochs <= 100:
                return False, "Epochs must be between 1 and 100"
            if not 16 <= self.batch_size <= 256:
                return False, "Batch size must be between 16 and 256"
            if not 0.00001 <= self.learning_rate <= 0.1:
                return False, "Learning rate must be between 0.00001 and 0.1"
        else:
            if not 10 <= self.n_estimators <= 500:
                return False, "n_estimators must be between 10 and 500"
            if not 1 <= self.max_depth <= 50:
                return False, "max_depth must be between 1 and 50"
            if not 2 <= self.min_samples_split <= 50:
                return False, "min_samples_split must be between 2 and 50"
            if not 2 <= self.cv_folds <= 10:
                return False, "cv_folds must be between 2 and 10"
        return True, None

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume", "timestamp"]
MAX_FILE_SIZE_MB = 5
MODELS_DIR = Path("models/")
MIN_SAMPLES = 100

@st.cache_resource
def get_model_manager() -> ModelManager:
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return ModelManager(base_directory=str(MODELS_DIR))

def validate_training_data(df: pd.DataFrame) -> DataValidationResult:
    """
    Validate OHLCV data for model training using the core data validator.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        DataValidationResult with validation status and details
    """
    try:
        logger.info("Starting data validation step")
        st.info("Validating data...")
        
        # Check for NaN or infinite values first (quick check)
        numeric_df = df.select_dtypes(include=[np.number])
        if not np.isfinite(numeric_df.values).all():
            return DataValidationResult(False, "Dataset contains NaN or infinity values.", None)
            
        # Use the comprehensive DataValidator from core module
        validation_result = validate_dataframe(
            df, 
            required_cols=REQUIRED_COLUMNS,
            validate_ohlc=True, 
            check_statistical_anomalies=True
        )
        
        # Process validation results
        if not validation_result.is_valid:
            error_message = "; ".join(validation_result.errors)
            logger.warning(f"Validation failed: {error_message}")
            return DataValidationResult(False, error_message, None)
            
        # Check minimum sample size requirement
        if len(df) < MIN_SAMPLES:
            logger.warning(f"Dataset too small (minimum {MIN_SAMPLES} samples required)")
            return DataValidationResult(False, f"Dataset too small (minimum {MIN_SAMPLES} samples required)", None)
            
        # Extract stats from validation result
        stats = {
            "samples": validation_result.row_count,
            "columns": validation_result.column_count,
            "null_counts": validation_result.null_counts,
            "data_types": validation_result.data_types,
            "memory_usage_mb": validation_result.statistics.get('memory_usage_mb', 0)
        }
        
        # Add additional price range info from OHLC data
        if 'price_range' in validation_result.statistics:
            ohlc_stats = validation_result.statistics.get('price_range', {})
            stats["price_range"] = f"{ohlc_stats.get('low', {}).get('min', 0):.2f} - {ohlc_stats.get('high', {}).get('max', 0):.2f}"
        
        # Add date range info if available
        date_info = validation_result.statistics.get('date_range', {})
        if date_info:
            start_date = date_info.get('start_date')
            end_date = date_info.get('end_date')
            if start_date and end_date:
                stats["timeframe"] = f"{start_date} to {end_date}"
        
        # Add volume info
        if 'volume' in df.columns:
            stats["mean_volume"] = df['volume'].mean()
            
        # Display warnings if any
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"Data warning: {warning}")

        logger.info("Data validation step completed")
        st.info("Data validated successfully.")
        
        return DataValidationResult(True, None, stats)
        
    except Exception as e:
        logger.exception("Data validation error")
        return DataValidationResult(False, f"Validation error: {str(e)}", None)

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add key technical indicators and candlestick pattern flags to the DataFrame.
    
    This is a unified preprocessing step suitable for training ML models.
    """
    df = df.copy()

    # Compute Technical Indicators
    ta = TechnicalAnalysis(df)
    df['RSI'] = ta.rsi(period=14)
    df['MACD'], df['MACD_Signal'] = ta.macd(fast_period=12, slow_period=26)
    df['BB_Upper'], df['BB_Lower'] = ta.bollinger_bands(period=20, std_dev=2)

    # Compute ATR for use in model or risk estimation
    df['ATR'] = ta.atr(period=3)    # Add Candlestick Pattern flags
    pattern_detector = create_pattern_detector()
    for pattern_name in pattern_detector.get_pattern_names():
        try:
            method = getattr(CandlestickPatterns, pattern_name)
            df[pattern_name.replace(" ", "")] = df.apply(
                lambda row: int(method(df.loc[max(0, row.name-4):row.name])), axis=1
            )
        except Exception:
            df[pattern_name.replace(" ", "")] = 0

    # Drop rows with NaN values introduced by indicators
    df.dropna(inplace=True)

    return df

def train_model_deep_learning(
    data: pd.DataFrame,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    selected_patterns: List[str] = []
) -> Tuple[Any, Dict[str, float]]:
    available_tickers = list(data['symbol'].unique()) if 'symbol' in data.columns else []
    selected_symbols = session_manager.create_multiselect(        "Select stocks to include (Deep Learning)",
        options=available_tickers,
        default=available_tickers[:min(3, len(available_tickers))],
        multiselect_name="dl_stocks"
    ) if available_tickers else []

    if not selected_symbols and available_tickers:
        st.warning("Please select at least one stock ticker")
        selected_symbols = [available_tickers[0]]
    
    # Filter data for selected symbols if applicable
    if selected_symbols and 'symbol' in data.columns:
        data = data[data['symbol'].isin(selected_symbols)]

    pattern_detector = create_pattern_detector()
    available_patterns = pattern_detector.get_pattern_names()
    selected_patterns = session_manager.create_multiselect(
        "Select candlestick patterns to use for training",
        options=available_patterns,
        default=available_patterns,
        multiselect_name="dl_pattern_multiselect"
    )

    logger.info("Starting model training step")
    st.info("Training model...")

    try:
        # --- Shape validation ---
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data is not a pandas DataFrame.")
        if data.empty:
            raise ValueError("Input data is empty.")
        # Example: check for expected columns
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Optionally, check for sequence dimension if using LSTM/CNN
        # (Assume your data loader will reshape as needed, but you can check here if you do it manually)

        model, metrics = train_pattern_model(
            symbols=selected_symbols,
            data=data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            selected_patterns=selected_patterns,
        )

    except IndexError as e:
        logger.error(f"Shape error during model training: {e}")
        st.error(f"Shape error during model training: {e}. "
                 "Please check that your data has the correct dimensions and sequence length.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model training: {e}")
        st.error(f"Unexpected error during model training: {e}")
        raise

    logger.info("Model training step completed")
    st.info("Model training completed.")

    return model, metrics

def train_model_classic_ml(
    data: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 10,
    cv_folds: int = 5
) -> Tuple[Any, Dict[str, Any]]:
    symbols = list(data['symbol'].unique()) if 'symbol' in data.columns else ["UNKNOWN"]
    config = MLConfig(
        symbols=symbols,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        cv_folds=cv_folds
    )
    trainer = ModelTrainer(config)
    params = TrainingParams(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        cv_folds=config.cv_folds,
        random_state=config.random_state
    )
    model_result, metrics, cm, report = trainer.train_model(data, params=params)
    return model_result, {"metrics": metrics, "confusion_matrix": cm, "classification_report": report}
 
def save_trained_model(
    model: Any,
    config: Any,
    metrics: Dict[str, float],
    backend: str,
    optimizer=None,
    epoch=None,
    loss=None,
    csv_filename=None,
    df=None
) -> Tuple[bool, Optional[str]]:
    if model is None:
        st.error("Model training failed. No model to save.")
        logger.error("Model training failed. No model to save.")
        return False, "No model to save."
    try:
        logger.info("Starting model saving step")
        st.info("Saving trained model...")
        logger.debug(f"Model object: {repr(model)}")
        st.write(f"Model type being saved: {type(model)}")
        logger.info(f"Model type being saved: {type(model)}")

        model_manager = get_model_manager()

        # Before saving, extract architecture params from the model
        if backend.startswith("Deep"):
            parameters = {
                "input_size": model.input_size,
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "output_size": model.output_size,
                "dropout": model.dropout,
                # ... other params ...
            }
        else:
            parameters = config.__dict__ if hasattr(config, "__dict__") else dict(config)

        metadata = ModelMetadata(
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            saved_at=datetime.now().isoformat(),
            accuracy=metrics.get("accuracy") if metrics else None,
            parameters=parameters,
            backend=backend
        )

        logger.info(f"Saving model with backend: {backend}")
        st.write(f"Saving model with backend: {backend}")
        save_path = model_manager.save_model(
            model, metadata=metadata, backend=backend,
            optimizer=optimizer, epoch=epoch, loss=loss,
            csv_filename=csv_filename, df=df
        )
        logger.info(f"Model saved to: {save_path} (absolute: {Path(save_path).resolve()})")
        st.write(f"Model saved to: {save_path} (absolute: {Path(save_path).resolve()})")
        model_manager.cleanup_old_models(keep_versions=5)
        logger.info(f"Model saved successfully: {save_path}")

        try:
            # --- Load metadata from JSON ---
            model_path = Path(save_path)
            metadata_path = model_path.with_suffix('.json')
            if metadata_path.exists():
                with metadata_path.open() as f:
                    loaded_metadata = json.load(f)
                params = loaded_metadata.get("parameters", {})
                # Only for Deep Learning models
                if backend.startswith("Deep"):
                    from patterns.patterns_nn import PatternNN  # Import your model class
                    loaded_model = PatternNN(
                        input_size=params.get("input_size", 10),
                        hidden_size=params.get("hidden_size", 64),
                        num_layers=params.get("num_layers", 2),
                        output_size=params.get("output_size", 3),
                        dropout=params.get("dropout", 0.2)
                    )
                    import torch
                    checkpoint = torch.load(model_path, map_location="cpu")
                    loaded_model.load_state_dict(checkpoint["state_dict"])
                    # Now loaded_model matches the saved architecture!
                else:
                    # Classic ML
                    loaded_model, loaded_metadata = model_manager.load_model(type(model), str(model_path))
            else:
                loaded_model, loaded_metadata = model_manager.load_model(type(model), str(model_path))
            logger.info("Model loaded successfully after save.")
        except Exception as e:
            logger.error(f"Failed to load model after save: {e}")
            st.error(f"Failed to load model after save: {e}")

        # Example preprocessing configuration
        preprocessing_config = {
            'feature_order': ['RSI', 'MACD', 'BullishEngulfing', 'Hammer'],
            'normalization': {
                'mean': [0.5, 0.3, 0.0, 0.0],
                'std': [0.1, 0.2, 1.0, 1.0]
            }
        }

        # Save with a filename that matches your model, e.g.:
        model_basename = Path(save_path).stem
        preproc_path = f"data/{model_basename}_preprocessing.json"
        save_preprocessing_config(preprocessing_config, path=preproc_path)

        logger.info("Model saving step completed")
        st.info("Model saved successfully.")

        return True, None
    except Exception as e:
        logger.exception("Failed to save model")
        st.error(f"Failed to save model: {e}")
        return False, str(e)

def display_signal_analysis(config: MLConfig, model_trainer: ModelTrainer) -> None:
    st.header("Signal Analysis")
    model_manager = get_model_manager()
    model_files = model_manager.list_models(pattern="*.*")
    if not model_files:
        st.warning("No trained models found. Train and save a model first.")
        return

    selected_model_file = session_manager.create_selectbox("Select Trained Model", options=model_files, selectbox_name="model_file")
    uploaded_data = session_manager.create_file_uploader("Upload Data for Signal Analysis (CSV)", type="csv", file_uploader_name="signal_data")
    if not uploaded_data:
        st.info("Upload a CSV file to analyze signals.")
        return

    logger.info("Starting data loading step")
    st.info("Loading data...")

    data = pd.read_csv(uploaded_data)

    logger.info("Data loading step completed")
    st.info("Data loaded successfully.")

    validation_result = validate_training_data(data)
    if not validation_result.is_valid:
        st.error(validation_result.error_message)
        return

    st.dataframe(data.head())

    with st.spinner("Loading model and running predictions..."):
        # Classic ML
        model, metadata = model_manager.load_model(selected_model_file)
        # Deep Learning
        model, metadata = model_manager.load_model(PatternNN, selected_model_file)
        # Classic ML or DL
        if metadata.backend.startswith("Classic"):
            preds = model.predict(data)
        else:
            # For DL, assume model has a predict method
            preds = model.predict(data)

        # Combine with candlestick patterns
        signals = generate_combined_signals(data, preds)
        st.subheader("Buy/Sell Signals")
        st.dataframe(signals.head(20))

        st.line_chart(signals[['close', 'buy_signal', 'sell_signal']])

def render_training_page():
    st.title("🧠 Model Training and Deployment")
    
    if "training_config" not in st.session_state:
        st.session_state.training_config = TrainingConfigUnified(backend="Deep Learning (PatternNN)")
    
    backend = session_manager.create_radio(
        "Select Training Backend",
        options=["Deep Learning (PatternNN)", "Classic ML (RandomForest)"],
        radio_name="backend_radio",
        index=0
    )
    st.session_state.training_config.backend = backend

    logger.info(f"Selected backend: {backend}")
    st.write(f"Selected backend: {backend}")

    use_synthetic = session_manager.create_checkbox("Generate synthetic data instead of uploading a file", "use_synthetic")
    if use_synthetic:
        add_to_model_training_ui()
    
    # Place the file uploader outside the form so it's available immediately
    uploaded_file = session_manager.create_file_uploader(
        "Upload Training Data (CSV)" if not use_synthetic else "Or upload your own data (CSV)", 
        type="csv",
        file_uploader_name="training_data_uploader",
        help=f"CSV file with {', '.join(REQUIRED_COLUMNS)} columns"
    )
      # Always show the form
    with session_manager.form_container("training_form"):  # renamed to avoid st.session_state["training_config"] conflict
        st.subheader("Training Configuration")
        config = st.session_state.training_config
        
        if st.session_state.training_config.backend == "Deep Learning (PatternNN)":
            col1, col2 = st.columns(2)
            with col1:
                config.epochs = session_manager.create_slider("Epochs", 1, 100, value=config.epochs, slider_name="epochs_slider")
                config.batch_size = session_manager.create_slider("Batch Size", 16, 256, value=config.batch_size, step=16, slider_name="batch_size_slider")
            with col2:
                config.learning_rate = session_manager.create_number_input(
                    "Learning Rate",
                    min_value=0.00001,
                    max_value=0.1,
                    value=config.learning_rate,                    number_input_name="learning_rate_input"
                )
        else:
            col1, col2 = st.columns(2)
            with col1:
                config.n_estimators = session_manager.create_slider("n_estimators", 10, 500, value=config.n_estimators, step=10, slider_name="n_estimators_slider")
                config.max_depth = session_manager.create_slider("max_depth", 1, 50, value=config.max_depth, slider_name="max_depth_slider")
            with col2:
                config.min_samples_split = session_manager.create_slider("min_samples_split", 2, 50, value=config.min_samples_split, slider_name="min_samples_split_slider")
                config.cv_folds = session_manager.create_slider("cv_folds", 2, 10, value=config.cv_folds, slider_name="cv_folds_slider")

        is_valid, error_msg = config.validate()
        submitted = st.form_submit_button("Train Model")

    # Only process after form is submitted
    if submitted:
        logger.info("Train Model button clicked")
        st.info("Starting training process...")
        if not is_valid:
            logger.error(f"Invalid config: {error_msg}")
            st.error(error_msg)
            return

        # Data loading and validation
        try:
            if use_synthetic:
                logger.info("Generating synthetic data for training")
                st.info("Generating synthetic data...")
                data = generate_synthetic_data()
            else:
                if not uploaded_file:
                    logger.error("No training data file uploaded")
                    st.error("Please upload a training data file to begin.")
                    return
                file_size_mb = uploaded_file.size / (1024 * 1024)
                logger.info(f"Uploaded file size: {file_size_mb:.2f} MB")
                if file_size_mb > MAX_FILE_SIZE_MB:
                    logger.error("Uploaded file too large")
                    st.error(f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB")
                    return
                logger.info("Starting data loading step")
                st.info("Loading data...")
                data = pd.read_csv(uploaded_file)
                logger.info("Data loading step completed")
                st.info("Data loaded successfully.")

            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
                data = data.set_index("timestamp")

            data = compute_technical_features(data)

            logger.info("Validating training data")
            st.info("Validating training data...")
            validation_result = validate_training_data(data.reset_index() if "timestamp" in data.index.names else data)
            if not validation_result.is_valid:
                logger.error(f"Data validation failed: {validation_result.error_message}")
                st.error(validation_result.error_message)
                return

            st.subheader("Data Preview")
            st.dataframe(data.head())
            if validation_result.stats:
                st.subheader("Dataset Statistics")
                st.json(validation_result.stats)

            logger.info(f"Starting model training with backend: {backend}")
            with st.spinner("Training model..."):
                if backend == "Deep Learning (PatternNN)":
                    model, metrics = train_model_deep_learning(
                        data,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate
                    )
                    optimizer = epoch = loss = None  # or set as needed
                else:
                    model, metrics = train_model_classic_ml(
                        data,
                        n_estimators=config.n_estimators,
                        max_depth=config.max_depth,
                        min_samples_split=config.min_samples_split,
                        cv_folds=config.cv_folds
                    )
                    optimizer = None
                    epoch = None
                    loss = None
            logger.info("Training completed")
            st.success("Training completed!")

            # After training
            st.session_state.trained_model = model
            st.session_state.training_metrics = metrics
            st.session_state.training_config = config
            st.session_state.training_backend = backend

            # Add a description
            st.markdown("""
            ### Model Evaluation Metrics

            - **Mean/Std**: These are the average and standard deviation of metrics (recall, accuracy, f1, precision) across cross-validation folds.
            - **Final Metrics**: These are the metrics computed on the final validation/test set.
            - **Confusion Matrix**: Shows the number of correct and incorrect predictions for each class.
            - **Classification Report**: Detailed precision, recall, f1-score, and support for each class.
            """)

            # Show mean/std/final metrics
            metrics_dict = metrics  # Assuming 'metrics' is the dictionary returned from training
            cv_metrics = metrics_dict.get("metrics", {})
            mean_metrics = cv_metrics.get("mean", {})
            std_metrics = cv_metrics.get("std", {})
            final_metrics = cv_metrics.get("final_metrics", {})

            st.subheader("Cross-Validation Metrics (Mean ± Std)")
            for key in ["recall", "accuracy", "f1", "precision"]:
                mean = mean_metrics.get(key, None)
                std = std_metrics.get(key, None)
                if mean is not None and std is not None:
                    st.write(f"**{key.capitalize()}**: {mean:.3f} ± {std:.3f}")

            st.subheader("Final Validation Metrics")
            for key, value in final_metrics.items():
                st.write(f"**{key.capitalize()}**: {value:.3f}")

            # Show confusion matrix
            confusion_matrix = metrics_dict.get("confusion_matrix", None)
            if confusion_matrix is not None:
                st.subheader("Confusion Matrix")
                st.write("Rows = Actual, Columns = Predicted")
                st.dataframe(confusion_matrix)

            # Show classification report
            classification_report = metrics_dict.get("classification_report", None)
            if classification_report is not None:
                st.subheader("Classification Report")
                st.text(classification_report)

            # Save the trained model
            success, error = save_trained_model(
                model, config, metrics, backend,
                optimizer=optimizer, epoch=epoch, loss=loss,
                csv_filename=uploaded_file.name if uploaded_file else None,
                df=data
            )
            if success:
                st.success("Model saved successfully.")
            else:
                st.error(f"Model save failed: {error}")

        except Exception as e:
            logger.exception("Training error")
            st.error(f"Training failed: {str(e)}")



    # ✅ Independent Save Block (always visible if model is present)
    if "trained_model" in st.session_state:
        st.subheader("Save Trained Model")
        if session_manager.create_button("Save Trained Model", "save_trained_model"):
            logger.info("Save Trained Model button clicked")
            try:
                model = st.session_state.get("trained_model")
                metrics = st.session_state.get("training_metrics")
                config = st.session_state.get("training_config")
                backend = st.session_state.get("training_backend")

                st.write("Model in session:", type(model))  # Confirm session content

                if model is None:
                    st.error("Model missing from session. Please train again.")
                    return

                success, error = save_trained_model(model, config, metrics, backend)
                if success:
                    st.success("Model saved successfully.")
                else:
                    st.error(f"Model save failed: {error}")
            except Exception as e:
                logger.exception("Exception during model save")
                st.error(f"Exception during model save: {e}")

                logger.exception("Exception during model save")
                st.error(f"Exception during model save: {e}")

class ModelTrainingDashboard:
    def __init__(self):
        # Initialize SessionManager for conflict-free button handling
        self.session_manager = create_session_manager("model_training")
    
    def run(self):
        """Main dashboard application entry point."""
        initialize_dashboard_session_state()
        with st_error_boundary():
            config = MLConfig(
                seq_len=10,
                epochs=10,
                batch_size=32,
                learning_rate=0.001,
                test_size=0.2,
                random_state=42,
                device="cuda" if torch.cuda.is_available() else "cpu",
                model_dir=MODELS_DIR,
                symbols=["AAPL"]  # <-- Use a valid list of symbols
            )
            model_trainer = ModelTrainer({'MODEL_DIR': str(MODELS_DIR)})
            model_manager = get_model_manager()
            model_files = model_manager.list_models()
            tab1, tab2 = st.tabs(["Model Training", "Signal Analysis"])
            with tab1:
                render_training_page()
            with tab2:
                display_signal_analysis(config, model_trainer)
                
            # Show SessionManager debug info in a sidebar expandable section
            with st.sidebar.expander("🔧 Session Debug Info", expanded=False):
                show_session_debug_info()

# Execute the main function
if __name__ == "__main__":
    try:
        dashboard = ModelTrainingDashboard()
        dashboard.run()
    except Exception as e:
        handle_streamlit_error(e, "Model Training Dashboard")

