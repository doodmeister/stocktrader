import streamlit as st
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, NamedTuple
from datetime import datetime
from dataclasses import dataclass

# Import torch at the top level to avoid circular imports
import torch

# Now import your utility modules after torch is imported
from utils.ml_pipeline import MLPipeline
from utils.model_manager import ModelManager, ModelMetadata
from utils.patterns_nn import PatternNN
from utils.performance_utils import st_error_boundary
from config import get_settings
from utils.synthetic_trading_data import add_to_model_training_ui, generate_synthetic_data
from utils.etrade_candlestick_bot import ETradeClient
from data.ml_config import MLConfig
from utils.stock_validation import get_valid_tickers

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

# Type definitions
@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    epochs: int 
    batch_size: int
    learning_rate: float

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate training parameters are within acceptable ranges."""
        if not 1 <= self.epochs <= 100:
            return False, "Epochs must be between 1 and 100"
        if not 16 <= self.batch_size <= 256:
            return False, "Batch size must be between 16 and 256"
        if not 0.00001 <= self.learning_rate <= 0.1:
            return False, "Learning rate must be between 0.00001 and 0.1"
        return True, None

class DataValidationResult(NamedTuple):
    """Result of data validation checks."""
    is_valid: bool
    error_message: Optional[str]
    stats: Optional[Dict[str, Any]] = None

# Constants moved to settings
settings = get_settings()
REQUIRED_COLUMNS = settings.required_columns  # lowercase
MAX_FILE_SIZE_MB = settings.max_file_size_mb  # lowercase
MODELS_DIR = Path(settings.models_dir)        # lowercase
MIN_SAMPLES = settings.min_samples            # lowercase

@st.cache_resource
def get_model_manager() -> ModelManager:
    """Initialize ModelManager as a cached resource."""
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return ModelManager(base_directory=str(MODELS_DIR))

def validate_training_data(df: pd.DataFrame) -> DataValidationResult:
    """
    Validate the uploaded training data format and content.
    
    Args:
        df: Input DataFrame containing training data
        
    Returns:
        DataValidationResult containing validation status and statistics
    """
    try:
        # Check required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return DataValidationResult(False, f"Missing required columns: {', '.join(missing_cols)}", None)
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            return DataValidationResult(
                False,
                f"Dataset contains null values: {dict(null_counts[null_counts > 0])}",
                None
            )
            
        # Check data types and ranges
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return DataValidationResult(False, f"Column {col} must be numeric", None)
            
            # Check for negative values in price/volume
            if (df[col] < 0).any():
                return DataValidationResult(False, f"Negative values found in {col}", None)
                
        # Validate OHLC relationships
        if not (df['high'] >= df['low']).all():
            return DataValidationResult(False, "High prices must be >= low prices", None)
            
        if not ((df['high'] >= df['open']) & (df['high'] >= df['close'])).all():
            return DataValidationResult(False, "High price must be >= open and close prices", None)
            
        if not ((df['low'] <= df['open']) & (df['low'] <= df['close'])).all():
            return DataValidationResult(False, "Low price must be <= open and close prices", None)
                
        # Check data size
        if len(df) < MIN_SAMPLES:
            return DataValidationResult(False, f"Dataset too small (minimum {MIN_SAMPLES} samples required)", None)
            
        # Calculate basic statistics
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

def train_model(
    data: pd.DataFrame,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[PatternNN, Dict[str, float]]:
    """
    Train a pattern recognition model using MLPipeline.
    
    Args:
        data: Training data as a DataFrame
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    # Get available tickers
    available_tickers = get_valid_tickers()

    # Let user select from validated tickers
    selected_symbols = st.multiselect(
        "Select stocks to include",
        options=available_tickers,
        default=available_tickers[:min(3, len(available_tickers))]  # Default to first 3 valid tickers
    )

    # Ensure at least one symbol is selected
    if not selected_symbols:
        st.warning("Please select at least one stock ticker")
        selected_symbols = [available_tickers[0]]  # Default to first ticker
    
    # Create configuration
    config = MLConfig(
        seq_len=10,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        test_size=0.2,
        random_state=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_dir=MODELS_DIR,
        symbols=selected_symbols
    )
    
    # Create client and pipeline
    client = ETradeClient(sandbox=True)
    pipeline = MLPipeline(client, config)
    
    # Initialize model
    model = PatternNN()
    
    # Train model
    metrics = pipeline.train_and_evaluate(model)
    
    return model, metrics

def save_trained_model(
    model: PatternNN,
    config: TrainingConfig,
    metrics: Dict[str, float]
) -> Tuple[bool, Optional[str]]:
    """
    Save trained model with metadata.
    
    Args:
        model: Trained model instance
        config: Training configuration
        metrics: Training metrics
        
    Returns:
        Success status and optional error message
    """
    try:
        model_manager = get_model_manager()
        metadata = ModelMetadata(
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            metrics=metrics,
            params=config.__dict__
        )
        
        save_path = model_manager.save_model(model, metadata=metadata)
        model_manager.cleanup_old_models(keep_versions=5)
        
        logger.info(f"Model saved successfully: {save_path}")
        return True, None
        
    except Exception as e:
        logger.exception("Failed to save model")
        return False, str(e)

def render_training_page():
    """Main training page render function with error boundary."""
    st.title("🧠 Model Training and Deployment")
    
    # Add option to generate synthetic data
    use_synthetic = st.checkbox("Generate synthetic data instead of uploading a file")
    
    if use_synthetic:
        add_to_model_training_ui()
    
    # Original file upload code continues below
    uploaded_file = st.file_uploader(
        "Upload Training Data (CSV)" if not use_synthetic else "Or upload your own data (CSV)", 
        type="csv",
        help=f"CSV file with {', '.join(REQUIRED_COLUMNS)} columns"
    )

    if not uploaded_file and not use_synthetic:
        st.info("Please upload a training data file to begin.")
        return

    # Validate file size
    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB")
            return

    try:
        if use_synthetic:
            data = generate_synthetic_data()
        else:
            data = pd.read_csv(uploaded_file)
        
        validation_result = validate_training_data(data)
        
        if not validation_result.is_valid:
            st.error(validation_result.error_message)
            return
            
        # Display data preview and statistics
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        if validation_result.stats:
            st.subheader("Dataset Statistics")
            st.json(validation_result.stats)
        
        with st.form("training_config"):
            st.subheader("Training Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.slider("Epochs", 1, 100, 10)
                batch_size = st.slider("Batch Size", 16, 256, 32, step=16)
                
            with col2:
                learning_rate = st.number_input(
                    "Learning Rate", 
                    min_value=0.00001,
                    max_value=0.1,
                    value=0.001, 
                    format="%.5f"
                )
                
            config = TrainingConfig(epochs, batch_size, learning_rate)
            is_valid, error_msg = config.validate()
            
            if not is_valid:
                st.error(error_msg)
                return
                
            submitted = st.form_submit_button("Train Model")
            
            if submitted:
                try:
                    with st.spinner("Training model..."):
                        model, metrics = train_model(
                            data,
                            epochs=config.epochs,
                            batch_size=config.batch_size,
                            learning_rate=config.learning_rate
                        )
                    
                    st.success("Training completed!")
                    st.json(metrics)
                    
                    if st.button("Save Trained Model"):
                        success, error = save_trained_model(model, config, metrics)
                        if success:
                            st.success("Model saved successfully")
                        else:
                            st.error(f"Failed to save model: {error}")
                        
                except Exception as e:
                    logger.exception("Training error")
                    st.error(f"Training failed: {str(e)}")

    except Exception as e:
        logger.exception("Error processing uploaded file")
        st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    with st_error_boundary():
        render_training_page()
