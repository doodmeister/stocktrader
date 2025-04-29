import streamlit as st
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, NamedTuple
from datetime import datetime
from dataclasses import dataclass

from ml_pipeline import train_model
from model_manager import ModelManager, ModelMetadata
from patterns_nn import PatternNN
from performance_utils import st_error_boundary
from config import get_settings

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
REQUIRED_COLUMNS = settings.REQUIRED_COLUMNS
MAX_FILE_SIZE_MB = settings.MAX_FILE_SIZE_MB
MODELS_DIR = Path(settings.MODELS_DIR)
MIN_SAMPLES = settings.MIN_SAMPLES

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

@st_error_boundary
def render_training_page():
    """Main training page render function with error boundary."""
    st.title("🧠 Model Training and Deployment")
    
    uploaded_file = st.file_uploader(
        "Upload Training Data (CSV)", 
        type="csv",
        help=f"CSV file with {', '.join(REQUIRED_COLUMNS)} columns"
    )

    if not uploaded_file:
        st.info("Please upload a training data file to begin.")
        return

    # Validate file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB")
        return

    try:
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
    render_training_page()
