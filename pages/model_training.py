import streamlit as st
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, NamedTuple, List
from datetime import datetime
from dataclasses import dataclass, asdict
from pydantic import BaseModel

import torch

# Deep learning pipeline
from utils.ml_pipeline import MLPipeline
from utils.model_manager import ModelManager, ModelMetadata
from utils.patterns_nn import PatternNN
from utils.performance_utils import st_error_boundary, generate_combined_signals
from utils.synthetic_trading_data import add_to_model_training_ui, generate_synthetic_data
from utils.etrade_candlestick_bot import ETradeClient
from data.ml_config import MLConfig
from utils.stock_validation import get_valid_tickers
from train.deeplearning_trainer import train_pattern_model

# Classic ML pipeline
from data.tradml_model_trainer import ModelTrainer, TrainingParams

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

class DataValidationResult(NamedTuple):
    is_valid: bool
    error_message: Optional[str]
    stats: Optional[Dict[str, Any]] = None

class MLConfig(BaseModel):
    # ...fields...
    symbols: List[str]

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
    try:
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return DataValidationResult(False, f"Missing required columns: {', '.join(missing_cols)}", None)
        null_counts = df.isnull().sum()
        if null_counts.any():
            return DataValidationResult(
                False,
                f"Dataset contains null values: {dict(null_counts[null_counts > 0])}",
                None
            )
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return DataValidationResult(False, f"Column {col} must be numeric", None)
            if (df[col] < 0).any():
                return DataValidationResult(False, f"Negative values found in {col}", None)
        if not (df['high'] >= df['low']).all():
            return DataValidationResult(False, "High prices must be >= low prices", None)
        if not ((df['high'] >= df['open']) & (df['high'] >= df['close'])).all():
            return DataValidationResult(False, "High price must be >= open and close prices", None)
        if not ((df['low'] <= df['open']) & (df['low'] <= df['close'])).all():
            return DataValidationResult(False, "Low price must be <= open and close prices", None)
        if len(df) < MIN_SAMPLES:
            return DataValidationResult(False, f"Dataset too small (minimum {MIN_SAMPLES} samples required)", None)
        stats = {
            "samples": len(df),
            "timeframe": f"{df.index[0]} to {df.index[-1]}",
            "mean_volume": df['volume'].mean(),
            "price_range": f"{df['low'].min():.2f} - {df['high'].max():.2f}"
        }
        return DataValidationResult(True, None, stats)
    except Exception as e:
        logger.exception("Data validation error")
        return DataValidationResult(False, f"Validation error: {str(e)}", None)

def train_model_deep_learning(
    data: pd.DataFrame,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[Any, Dict[str, float]]:
    available_tickers = get_valid_tickers()
    selected_symbols = st.multiselect(
        "Select stocks to include (Deep Learning)",
        options=available_tickers,
        default=available_tickers[:min(3, len(available_tickers))]
    )
    if not selected_symbols:
        st.warning("Please select at least one stock ticker")
        selected_symbols = [available_tickers[0]]
    client = ETradeClient(sandbox=True)
    # Delegate all training logic to train_pattern_model
    model, metrics = train_pattern_model(
        client=client,
        symbols=selected_symbols,
        data=data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        # Add other parameters as needed
    )
    return model, metrics

def train_model_classic_ml(
    data: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 10,
    cv_folds: int = 5
) -> Tuple[Any, Dict[str, Any]]:
    st.info("Training classic ML model (RandomForest)...")
    trainer = ModelTrainer({'MODEL_DIR': str(MODELS_DIR)})
    params = TrainingParams(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        cv_folds=cv_folds,
        random_state=42
    )
    model, metrics, cm, report = trainer.train_model(data, params=params)
    return model, {
        "metrics": metrics,
        "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
        "classification_report": report
    }

def save_trained_model(
    model: Any,
    config: Any,
    metrics: Dict[str, float],
    backend: str
) -> Tuple[bool, Optional[str]]:
    try:
        model_manager = get_model_manager()
        metadata = ModelMetadata(
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            metrics=metrics,
            params=config.__dict__ if hasattr(config, "__dict__") else dict(config),
            backend=backend
        )
        save_path = model_manager.save_model(model, metadata=metadata)
        model_manager.cleanup_old_models(keep_versions=5)
        logger.info(f"Model saved successfully: {save_path}")
        return True, None
    except Exception as e:
        logger.exception("Failed to save model")
        return False, str(e)

def display_signal_analysis(config: MLConfig, model_trainer: ModelTrainer) -> None:
    st.header("Signal Analysis")
    model_manager = get_model_manager()
    model_files = model_manager.list_models()
    if not model_files:
        st.warning("No trained models found. Train and save a model first.")
        return

    selected_model_file = st.selectbox("Select Trained Model", options=model_files)
    uploaded_data = st.file_uploader("Upload Data for Signal Analysis (CSV)", type="csv")
    if not uploaded_data:
        st.info("Upload a CSV file to analyze signals.")
        return

    data = pd.read_csv(uploaded_data)
    validation_result = validate_training_data(data)
    if not validation_result.is_valid:
        st.error(validation_result.error_message)
        return

    st.dataframe(data.head())

    with st.spinner("Loading model and running predictions..."):
        model, metadata = model_manager.load_model(selected_model_file)
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

    backend = st.radio(
        "Select Training Backend",
        options=["Deep Learning (PatternNN)", "Classic ML (RandomForest)"],
        index=0,
        key="backend_radio"
    )
    st.session_state.training_config.backend = backend

    use_synthetic = st.checkbox("Generate synthetic data instead of uploading a file")
    if use_synthetic:
        add_to_model_training_ui()

    uploaded_file = st.file_uploader(
        "Upload Training Data (CSV)" if not use_synthetic else "Or upload your own data (CSV)", 
        type="csv",
        help=f"CSV file with {', '.join(REQUIRED_COLUMNS)} columns"
    )

    # Always show the form
    with st.form("training_config"):
        st.subheader("Training Configuration")
        config = st.session_state.training_config

        if backend.startswith("Deep"):
            col1, col2 = st.columns(2)
            with col1:
                config.epochs = st.slider("Epochs", 1, 100, config.epochs)
                config.batch_size = st.slider("Batch Size", 16, 256, config.batch_size, step=16)
            with col2:
                config.learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=0.00001,
                    max_value=0.1,
                    value=config.learning_rate,
                    format="%.5f"
                )
        else:
            col1, col2 = st.columns(2)
            with col1:
                config.n_estimators = st.slider("n_estimators", 10, 500, config.n_estimators, step=10)
                config.max_depth = st.slider("max_depth", 1, 50, config.max_depth)
            with col2:
                config.min_samples_split = st.slider("min_samples_split", 2, 50, config.min_samples_split)
                config.cv_folds = st.slider("cv_folds", 2, 10, config.cv_folds)

        is_valid, error_msg = config.validate()
        submitted = st.form_submit_button("Train Model")

    # Only process after form is submitted
    if submitted:
        if not is_valid:
            st.error(error_msg)
            return

        # Data loading and validation
        try:
            if use_synthetic:
                data = generate_synthetic_data()
            else:
                if not uploaded_file:
                    st.error("Please upload a training data file to begin.")
                    return
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > MAX_FILE_SIZE_MB:
                    st.error(f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB")
                    return
                data = pd.read_csv(uploaded_file)

            validation_result = validate_training_data(data)
            if not validation_result.is_valid:
                st.error(validation_result.error_message)
                return

            st.subheader("Data Preview")
            st.dataframe(data.head())
            if validation_result.stats:
                st.subheader("Dataset Statistics")
                st.json(validation_result.stats)

            with st.spinner("Training model..."):
                if backend == "Deep Learning (PatternNN)":
                    model, metrics = train_model_deep_learning(
                        data,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate
                    )
                else:
                    model, metrics = train_model_classic_ml(
                        data,
                        n_estimators=config.n_estimators,
                        max_depth=config.max_depth,
                        min_samples_split=config.min_samples_split,
                        cv_folds=config.cv_folds
                    )
            st.success("Training completed!")
            st.json(metrics)
            if st.button("Save Trained Model"):
                success, error = save_trained_model(model, config, metrics, backend)
                if success:
                    st.success("Model saved successfully")
                else:
                    st.error(f"Failed to save model: {error}")
        except Exception as e:
            logger.exception("Training error")
            st.error(f"Training failed: {str(e)}")

if __name__ == "__main__":
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
