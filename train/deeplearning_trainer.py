"""
Deep Learning Model Trainer for Candlestick Pattern Recognition

This module provides a robust training pipeline for neural network models that detect
candlestick patterns in financial time series data. It includes comprehensive data
preprocessing, model training with early stopping, and thorough evaluation metrics.

Key Features:
- Time-series aware train/validation splitting
- Robust feature engineering with technical indicators
- Multi-label pattern classification
- Comprehensive error handling and logging
- Model persistence and versioning
- Performance monitoring and metrics tracking
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union # Added Union
from datetime import datetime
import warnings

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, hamming_loss, precision_score, 
    recall_score, f1_score, classification_report,
    multilabel_confusion_matrix
)

from utils.logger import setup_logger
from train.deeplearning_config import TrainingConfig
from patterns.patterns_nn import PatternNN, PatternNNConfig # Added PatternNNConfig
from patterns.patterns import create_pattern_detector
from train.model_manager import ModelManager
from utils.technicals.feature_engineering import compute_technical_features
from patterns.pattern_utils import add_candlestick_pattern_features

# Import centralized data validation system
from core.data_validator import (
    validate_dataframe
)

# Suppress sklearn warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logger = setup_logger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training session metrics and metadata."""
    subset_accuracy: float
    hamming_loss: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    classification_report: Dict[str, Any]
    confusion_matrices: Optional[np.ndarray] = None
    training_time: Optional[float] = None
    best_epoch: Optional[int] = None
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None


class TrainingError(Exception):
    """Custom exception for training-related errors."""
    pass


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible training across all libraries.
    
    Args:
        seed: Random seed value
        
    Note:
        This ensures reproducible results but may impact performance slightly.
    """
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Additional CUDA determinism (may impact performance)
            if hasattr(torch, 'backends'):
                backends_mod = torch.backends # type: ignore
                if hasattr(backends_mod, 'cudnn') and backends_mod.cudnn.is_available():
                    backends_mod.cudnn.deterministic = True
                    backends_mod.cudnn.benchmark = False
        logger.debug(f"Random seed set to {seed} for reproducible training")
    except Exception as e:
        logger.warning(f"Failed to set random seed: {e}")


class PatternModelTrainer:
    """
    Handles the training of neural network models for candlestick pattern recognition.
    
    This class provides a complete pipeline for:
    - Data preprocessing and feature engineering
    - Model training with early stopping and learning rate scheduling
    - Comprehensive evaluation and metrics calculation
    - Model persistence and versioning
    
    Attributes:
        model_manager: Handles model saving/loading operations
        config: Training configuration parameters
        selected_patterns: List of candlestick patterns to train on
        scaler: Feature scaling transformer
    """

    def __init__(
        self,
        model_manager: ModelManager,
        config: TrainingConfig,
        selected_patterns: List[str]
    ):
        """
        Initialize the trainer with configuration and dependencies.
        
        Args:
            model_manager: ModelManager instance for persistence
            config: TrainingConfig with hyperparameters
            selected_patterns: List of pattern names to train on
            
        Raises:
            ValueError: If configuration is invalid or patterns list is empty
        """
        self.model_manager = model_manager
        self.config = config
        self.selected_patterns = selected_patterns
        self.scaler: Optional[Union[StandardScaler, RobustScaler]] = None # Changed type hint
        self._last_df: Optional[pd.DataFrame] = None
        
        # Validate inputs during initialization
        self._validate_training_params()
        self._validate_patterns()
        logger.info(f"Initialized PatternModelTrainer with {len(selected_patterns)} patterns")

    def _validate_training_params(self) -> None:
        """Validate training configuration parameters."""
        try:
            # Local validation to avoid circular import with core.data_validator
            if self.config.epochs <= 0:
                raise ValueError("epochs must be > 0")
            if not (0 < self.config.validation_split < 1):
                raise ValueError("validation_split must be in (0,1)")
            if self.config.learning_rate <= 0:
                raise ValueError("learning_rate must be > 0")
            if self.config.batch_size <= 0:
                raise ValueError("batch_size must be > 0")
        except Exception as e:
            raise TrainingError(f"Training configuration validation failed: {e}")

    def _validate_patterns(self) -> None:
        """Validate selected patterns list."""
        if not self.selected_patterns:
            raise ValueError("No candlestick patterns selected for training")
        
        if not isinstance(self.selected_patterns, list):
            raise ValueError("Selected patterns must be a list")
        
        if len(self.selected_patterns) > 50:  # Reasonable upper limit
            raise ValueError(f"Too many patterns selected: {len(self.selected_patterns)}. Maximum is 50")
        
        logger.debug(f"Validated {len(self.selected_patterns)} patterns for training")

    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """
        Validate input DataFrame for training using centralized validation system.
        
        Args:
            data: Input DataFrame to validate
            
        Raises:
            DataValidationError: If data validation fails
        """
        logger.debug("Starting deep learning data validation")
        
        # Basic type and structure validation
        if not isinstance(data, pd.DataFrame):
            raise DataValidationError("Input data must be a pandas DataFrame")
        
        if data.empty:
            raise DataValidationError("Input DataFrame is empty")
        
        # Use centralized validation system for core validation
        try:
            logger.debug("Performing core data validation (OHLC integrity, statistical anomalies)")
            validation_result = validate_dataframe(
                data,
                required_cols=['open', 'high', 'low', 'close'],
                check_ohlcv=True # Parameter kept
                # detect_anomalies_level='basic' # Parameter removed
            )
            
            if not validation_result.is_valid:
                error_messages = validation_result.errors if validation_result.errors is not None else []
                error_msg = "; ".join(error_messages) if error_messages else "Core data validation failed without specific error messages."
                logger.error(f"Core data validation failed: {error_msg}")
                raise DataValidationError(f"Core validation failed: {error_msg}")
            
            # Log validation warnings from core system
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.warning(f"Core validation warning: {warning}")
                    
            if validation_result.dataframe_shape:
                rows, cols = validation_result.dataframe_shape
                logger.debug(f"Core validation passed: {rows}x{cols} DataFrame")
            else:
                logger.debug("Core validation passed, but DataFrame shape not available.")
            
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise
            logger.error(f"Core validation system error: {e}")
            raise DataValidationError(f"Validation system error: {str(e)}")
        
        # Deep learning specific validation requirements
        logger.debug("Performing deep learning specific validation")
        
        # Check sequence length requirements for deep learning models
        min_rows_needed = self.config.seq_len * 2
        if len(data) < min_rows_needed:
            raise DataValidationError(
                f"Insufficient data for sequence modeling: {len(data)} rows available, "
                f"need at least {min_rows_needed} rows (seq_len={self.config.seq_len} * 2)"
            )
        
        # Check for NaN values in OHLC data (deep learning models are sensitive to NaN)
        required_cols = ['open', 'high', 'low', 'close']
        if data[required_cols].isnull().any().any():
            nan_info = data[required_cols].isnull().sum()
            logger.warning(f"Found NaN values in OHLC data: {nan_info.to_dict()}")
            logger.warning("Deep learning models require clean data - consider data preprocessing")
        
        # Additional deep learning specific checks
        # Check for extreme values that might cause training instability
        for col in required_cols:
            if (data[col] <= 0).any():
                zero_count = (data[col] <= 0).sum()
                logger.warning(f"Found {zero_count} non-positive values in {col} - may cause log transformation issues")
        
        logger.debug(f"Deep learning validation passed: {len(data)} rows, {len(data.columns)} columns")
        logger.info(f"Data validation successful for sequence modeling (seq_len={self.config.seq_len})")

    def prepare_training_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and engineer features for training.
        
        This method performs comprehensive data preprocessing including:
        - Technical indicator calculation
        - Candlestick pattern feature engineering
        - Sequence generation for time-series modeling
        - Data validation and cleaning
        
        Args:
            data: Raw OHLC DataFrame
            
        Returns:
            Tuple of (features, labels) as numpy arrays
            
        Raises:
            DataValidationError: If data validation fails
            TrainingError: If feature engineering fails
        """
        logger.info("Starting data preparation for training")
        
        # Validate input data
        self._validate_input_data(data)
        
        try:
            # Create a copy to avoid modifying original data
            data_copy = data.copy()
            
            # Handle missing values
            data_copy = self._handle_missing_values(data_copy)
            
            # Feature engineering
            logger.debug("Computing technical features")
            data_copy = compute_technical_features(data_copy)
            
            logger.debug("Adding candlestick pattern features")
            data_copy = add_candlestick_pattern_features(data_copy)
            
            # Store processed data for inspection
            self._last_df = data_copy.copy()
            
            # Generate sequences and labels
            features, labels = self._process_data(data_copy)
            
            if not features:
                raise TrainingError("No valid sequences generated for training")
            
            X = np.array(features, dtype=np.float32)
            y = np.array(labels, dtype=np.float32)
            
            # Validate and reshape arrays
            X, y = self._validate_and_reshape_arrays(X, y)
            
            logger.info(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise TrainingError(f"Failed to prepare training data: {e}")

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Forward fill OHLC data
        ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
        existing_ohlc = [col for col in ohlc_cols if col in data.columns]
        
        if data[existing_ohlc].isnull().any().any():
            logger.warning("Found missing OHLC data, applying forward fill")
            # data[existing_ohlc] = data[existing_ohlc].fillna(method='ffill') # Deprecated
            data.loc[:, existing_ohlc] = data.loc[:, existing_ohlc].ffill()
            
            # If still have NaN (at beginning), use backward fill
            # data[existing_ohlc] = data[existing_ohlc].fillna(method='bfill') # Deprecated
            data.loc[:, existing_ohlc] = data.loc[:, existing_ohlc].bfill()
        
        return data

    def _validate_and_reshape_arrays(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and reshape feature and label arrays."""
        # Ensure X is 3D for LSTM: (batch, seq_len, features)
        if X.ndim == 2 and self.config.seq_len > 1:
            num_features = X.shape[1] // self.config.seq_len
            if X.shape[1] % self.config.seq_len == 0:
                X = X.reshape(-1, self.config.seq_len, num_features)
                logger.info(f"Reshaped X from 2D to 3D: {X.shape}")
            else:
                raise TrainingError(
                    f"Cannot reshape X of shape {X.shape} to (batch, seq_len, features)"
                )
        
        if X.ndim != 3:
            raise TrainingError(
                f"Expected X to be 3D (batch, seq_len, features), got shape {X.shape}"
            )
        
        if len(X) == 0 or len(y) == 0:
            raise TrainingError("No data available for training after preprocessing")
        
        if len(X) != len(y):
            raise TrainingError(f"Feature and label count mismatch: {len(X)} vs {len(y)}")
        
        return X, y

    def _process_data(
        self, data: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process DataFrame into sequences and labels.
        
        Args:
            data: Preprocessed DataFrame with features
            
        Returns:
            Tuple of (feature_sequences, pattern_labels)
        """
        features, labels = [], []
        
        # Identify feature columns (exclude metadata)
        exclude_cols = {"timestamp", "symbol", "date", "datetime"}
        feature_cols = [col for col in data.columns if col.lower() not in exclude_cols]
        
        if not feature_cols:
            raise TrainingError("No feature columns found in processed data")
        
        logger.debug(f"Processing {len(feature_cols)} feature columns")
        
        # Generate sequences
        valid_sequences = 0
        for i in range(len(data) - self.config.seq_len):
            try:
                # Extract sequence window
                window = data.iloc[i: i + self.config.seq_len][feature_cols].values
                
                # Skip if window contains NaN or infinite values
                if not np.isfinite(window).all():
                    continue
                
                # Normalize sequence (relative to first value to handle different scales)
                seq = self._normalize_sequence(window)
                  # Detect patterns for this window
                pattern_window = data.iloc[i: i + self.config.seq_len]
                pattern_detector = create_pattern_detector()
                detected_results = pattern_detector.detect_patterns(pattern_window)
                # Extract pattern names from PatternResult objects
                patterns = [result.name for result in detected_results if result.detected]
                
                # Encode patterns as multi-label
                label = self._encode_patterns(patterns, self.selected_patterns)
                
                # Only include sequences with at least one pattern
                if np.any(label):
                    features.append(seq)
                    labels.append(label)
                    valid_sequences += 1
                    
            except Exception as e:
                logger.debug(f"Skipping sequence at index {i}: {e}")
                continue
        
        logger.info(f"Generated {valid_sequences} valid sequences from {len(data)} data points")
        
        if valid_sequences == 0:
            raise TrainingError("No valid sequences with patterns found")
        
        return features, labels

    def _normalize_sequence(self, window: np.ndarray) -> np.ndarray:
        """
        Normalize a sequence window for training.
        
        Args:
            window: Raw feature window
            
        Returns:
            Normalized sequence
        """
        # Use robust normalization to handle outliers
        first_row = window[0]
        
        # Avoid division by zero
        safe_first = np.where(np.abs(first_row) < 1e-8, 1e-8, first_row)
        
        # Relative change normalization
        seq = (window - first_row) / np.abs(safe_first)
        
        # Add small amount of noise for regularization (if enabled)
        if getattr(self.config, 'add_noise', False) and np.random.rand() < 0.3:
            noise_scale = getattr(self.config, 'noise_scale', 0.01)
            seq += np.random.normal(0, noise_scale, seq.shape)
        
        return seq

    def _encode_patterns(
        self,
        patterns: List[str],
        selected_patterns: List[str]
    ) -> np.ndarray:
        """
        One-hot encode detected patterns for multi-label classification.
        
        Args:
            patterns: List of detected pattern names
            selected_patterns: List of target pattern names
            
        Returns:
            Binary encoded label array
        """
        label = np.zeros(len(selected_patterns), dtype=np.float32)
        
        for idx, pattern in enumerate(selected_patterns):
            if pattern in patterns:
                label[idx] = 1.0
        
        return label

    def train_model(
        self,
        data: pd.DataFrame,
        model: Optional[PatternNN] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[PatternNN, TrainingMetrics]:
        """
        Train a neural network model for pattern recognition.
        
        This method orchestrates the complete training pipeline including:
        - Data preparation and validation
        - Model initialization and configuration
        - Training loop with early stopping
        - Model evaluation and metrics calculation
        
        Args:
            data: Training DataFrame with OHLC data
            model: Pre-initialized model (optional)
            metadata: Additional metadata to track
            
        Returns:
            Tuple of (trained_model, training_metrics)
            
        Raises:
            TrainingError: If training fails
        """
        start_time = datetime.now()
        logger.info("Starting neural network training pipeline")
        
        try:
            # Set random seed for reproducibility
            if hasattr(self.config, 'seed') and self.config.seed is not None:
                set_seed(self.config.seed)
            
            # Prepare device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            
            # Prepare training data
            X, y = self.prepare_training_data(data)
            
            # Split data chronologically (time-aware split)
            X_train, X_val, y_train, y_val = self._create_time_split(X, y)
            
            # Scale features
            X_train_scaled, X_val_scaled = self._scale_features(X_train, X_val)
            
            # Initialize or validate model
            model = self._initialize_model(model, X_train_scaled.shape[2])
            model.to(device)
            
            # Create data loaders
            train_loader = self._create_data_loader(X_train_scaled, y_train, shuffle=True)
            val_loader = self._create_data_loader(X_val_scaled, y_val, shuffle=False)
            
            # Train the model
            trained_model, best_epoch, final_losses = self._train_loop(
                model, train_loader, val_loader, device
            )
            
            # Evaluate model
            metrics = self._evaluate_model(trained_model, val_loader, device)
            
            # Add training metadata
            training_time = (datetime.now() - start_time).total_seconds()
            metrics.training_time = training_time
            metrics.best_epoch = best_epoch
            metrics.final_train_loss = final_losses[0]
            metrics.final_val_loss = final_losses[1]
            
            logger.info(f"Training completed in {training_time:.2f}s with validation accuracy: {metrics.subset_accuracy:.4f}")
            
            return trained_model, metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise TrainingError(f"Model training failed: {e}")

    def _create_time_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create time-aware train/validation split."""
        split_idx = int(len(X) * (1 - self.config.validation_split))
        
        # Ensure minimum validation size
        min_val_size = max(32, self.config.batch_size)
        if len(X) - split_idx < min_val_size:
            split_idx = len(X) - min_val_size
            logger.warning(f"Adjusted split to ensure minimum validation size: {min_val_size}")
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Data split - Train: {len(X_train)}, Validation: {len(X_val)}")
        return X_train, X_val, y_train, y_val

    def _scale_features(
        self, X_train: np.ndarray, X_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using robust scaling."""
        batch_size, seq_len, feat_dim = X_train.shape
        
        # Use RobustScaler for better handling of outliers
        self.scaler = RobustScaler()
        
        # Flatten for scaling
        X_train_flat = X_train.reshape(-1, feat_dim)
        X_train_scaled = self.scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        
        X_val_flat = X_val.reshape(-1, feat_dim)
        X_val_scaled = self.scaler.transform(X_val_flat).reshape(X_val.shape)
        
        logger.debug("Applied robust feature scaling")
        return X_train_scaled, X_val_scaled

    def _initialize_model(
        self, model: Optional[PatternNN], input_size: int
    ) -> PatternNN:
        """Initialize or validate the neural network model."""
        if model is None:
            # Map TrainingConfig to PatternNNConfig
            nn_config_params = {
                'input_size': input_size,
                'output_size': len(self.selected_patterns),
                'sequence_length': self.config.seq_len,
                'dropout': self.config.dropout_rate,
                'activation': self.config.activation_function
            }
            if self.config.hidden_layers:
                if len(self.config.hidden_layers) > 0:
                    nn_config_params['hidden_size'] = self.config.hidden_layers[0]
                    nn_config_params['num_layers'] = len(self.config.hidden_layers)
                else:
                    logger.warning("TrainingConfig.hidden_layers is empty, relying on PatternNNConfig defaults for hidden_size and num_layers.")
            else:
                 logger.warning("TrainingConfig.hidden_layers not specified, relying on PatternNNConfig defaults for hidden_size and num_layers.")

            # use_batch_norm and use_residual will use PatternNNConfig defaults 
            # as TrainingConfig does not specify them.
            
            nn_config = PatternNNConfig(**nn_config_params)
            model = PatternNN(config=nn_config)
            logger.info(f"Initialized new PatternNN model with config: {nn_config.to_dict()}")
        else:
            # Validate existing model dimensions
            if not hasattr(model, 'config'):
                 raise TrainingError("Provided model does not have a 'config' attribute.")
            if model.config.input_size != input_size:
                raise TrainingError(
                    f"Model input size mismatch: expected {input_size}, got {model.config.input_size}"
                )
            if model.config.output_size != len(self.selected_patterns):
                raise TrainingError(
                    f"Model output size mismatch: expected {len(self.selected_patterns)}, got {model.config.output_size}"
                )
        
        return model

    def _train_loop(
        self,
        model: PatternNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ) -> Tuple[PatternNN, int, Tuple[float, float]]:
        """Execute the main training loop with early stopping."""
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=getattr(self.config, 'weight_decay', 1e-5)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
        
        # Use focal loss for imbalanced data or BCEWithLogitsLoss
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.ones(len(self.selected_patterns)).to(device)
        )
        
        # Early stopping variables
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        best_epoch = 0
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss = self._train_epoch(model, train_loader, optimizer, loss_fn, device)
            
            # Validation phase
            val_loss = self._validate_epoch(model, val_loader, loss_fn, device)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Log progress
            if epoch % 10 == 0 or epoch < 5:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model from epoch {best_epoch}")
        
        return model, best_epoch, (train_loss, val_loss)

    def _train_epoch(
        self,
        model: PatternNN,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device
    ) -> float:
        """Execute one training epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            try:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                # Validate outputs
                if outputs is None:
                    raise RuntimeError("Model returned None output")
                
                # Calculate loss
                loss = loss_fn(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                clip_grad_norm_(model.parameters(), max_norm=1.0) # Changed call
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.warning(f"Skipping batch due to error: {e}")
                continue
        
        return total_loss / max(num_batches, 1)

    def _validate_epoch(
        self,
        model: PatternNN,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device
    ) -> float:
        """Execute one validation epoch."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                try:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    
                    if outputs is not None:
                        loss = loss_fn(outputs, batch_y)
                        total_loss += loss.item()
                        num_batches += 1
                        
                except Exception as e:
                    logger.warning(f"Skipping validation batch due to error: {e}")
                    continue
        
        return total_loss / max(num_batches, 1)

    def _evaluate_model(
        self, model: PatternNN, val_loader: DataLoader, device: torch.device
    ) -> TrainingMetrics:
        """
        Comprehensive model evaluation with multi-label metrics.
        
        Args:
            model: Trained model to evaluate
            val_loader: Validation data loader
            device: Compute device
            
        Returns:
            TrainingMetrics containing all evaluation results
        """
        model.eval()
        all_y_true, all_y_pred = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                try:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    
                    if outputs is not None:
                        # Convert to probabilities and predictions
                        scores = torch.sigmoid(outputs).cpu().numpy()
                        preds = (scores > 0.5).astype(int)
                        
                        all_y_pred.extend(preds)
                        all_y_true.extend(batch_y.cpu().numpy())
                        
                except Exception as e:
                    logger.warning(f"Skipping evaluation batch due to error: {e}")
                    continue
        
        if not all_y_true:
            logger.warning("No valid evaluation data found")
            return TrainingMetrics(0.0, 1.0, 0.0, 0.0, 0.0, {})
        
        # Convert to numpy arrays
        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)

        # Calculate comprehensive metrics
        try:
            subset_acc = accuracy_score(y_true, y_pred)
            ham_loss = hamming_loss(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            # Classification report with proper handling
            # Ensure report is a dictionary, even if sklearn typeshed is imperfect
            _report_dict_or_str = classification_report(
                y_true, y_pred, 
                target_names=self.selected_patterns,
                output_dict=True,
                zero_division=0
            )
            
            report: Dict[str, Any]
            if isinstance(_report_dict_or_str, str):
                # This case should ideally not happen with output_dict=True,
                # but handling defensively for type checker and robustness.
                logger.warning("classification_report returned a string despite output_dict=True. Converting to basic dict.")
                report = {"report_string": _report_dict_or_str} 
            elif isinstance(_report_dict_or_str, dict):
                report = _report_dict_or_str
            else:
                logger.error(f"Unexpected type for classification_report: {type(_report_dict_or_str)}. Using empty dict.")
                report = {}
            
            # Confusion matrices for each label
            confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
            
            logger.info(f"Evaluation completed - Subset Accuracy: {subset_acc:.4f}, F1: {f1:.4f}")
            
            return TrainingMetrics(
                subset_accuracy=float(subset_acc), # Cast to float
                hamming_loss=float(ham_loss),       # Cast to float
                precision_macro=float(precision),   # Cast to float
                recall_macro=float(recall),       # Cast to float
                f1_macro=float(f1),               # Cast to float
                classification_report=report, # Now ensured to be Dict[str, Any]
                confusion_matrices=confusion_matrices
            )
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            # Ensure a Dict[str, Any] is passed for classification_report even in error cases
            return TrainingMetrics(0.0, 1.0, 0.0, 0.0, 0.0, {"error": str(e)})

    def _create_data_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader with proper configuration.
        
        Args:
            X: Feature array
            y: Label array
            shuffle: Whether to shuffle data
            
        Returns:
            Configured DataLoader
        """
        try:
            tensor_x = torch.tensor(X, dtype=torch.float32)
            tensor_y = torch.tensor(y, dtype=torch.float32)
            dataset = TensorDataset(tensor_x, tensor_y)
            
            # Calculate optimal number of workers
            num_workers = min(4, torch.get_num_threads()) if len(X) > 1000 else 0
            
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False  # Keep all data
            )
            
        except Exception as e:
            logger.error(f"Failed to create DataLoader: {e}")
            raise TrainingError(f"DataLoader creation failed: {e}")

    def get_feature_importance(self, model: PatternNN) -> Optional[Dict[str, float]]:
        """
        Calculate feature importance using gradient-based methods.
        
        Args:
            model: Trained model
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self._last_df is None or self.scaler is None:
            logger.warning("No processed data available for feature importance calculation")
            return None
        
        try:
            # This is a simplified implementation
            # In practice, you might want to use integrated gradients or SHAP
            feature_cols = [col for col in self._last_df.columns 
                          if col.lower() not in {"timestamp", "symbol", "date", "datetime"}]
            
            # Return uniform importance as placeholder
            importance = {col: 1.0 / len(feature_cols) for col in feature_cols}
            
            logger.debug(f"Calculated feature importance for {len(feature_cols)} features")
            return importance
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return None


def train_pattern_model(
    symbols: List[str],
    data: pd.DataFrame,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    selected_patterns: Optional[List[str]] = None, # Changed type hint
    **kwargs
) -> Tuple[PatternNN, TrainingMetrics]:
    """
    Convenience function for training a pattern recognition model.
    
    This function provides a simplified interface for model training with
    sensible defaults and error handling.
    
    Args:
        symbols: List of trading symbols (for metadata)
        data: Training DataFrame with OHLC data
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        selected_patterns: List of patterns to train on
        **kwargs: Additional configuration parameters
        
    Returns:
        Tuple of (trained_model, training_metrics)
        
    Raises:
        TrainingError: If training fails
        ValueError: If parameters are invalid
    """
    logger.info(f"Starting pattern model training for symbols: {symbols}")
    
    try:
        # Set default patterns if none provided
        if selected_patterns is None:
            selected_patterns = [
                'hammer', 'doji', 'engulfing_bullish', 'engulfing_bearish',
                'morning_star', 'evening_star', 'shooting_star'
            ]
        
        # Create configuration
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs
        )
        
        # Initialize components
        manager = ModelManager()
        trainer = PatternModelTrainer(
            model_manager=manager,
            config=config,
            selected_patterns=selected_patterns
        )
        
        # Train model
        model, metrics = trainer.train_model(data)
        
        logger.info("Pattern model training completed successfully")
        return model, metrics
        
    except Exception as e:
        logger.error(f"Pattern model training failed: {e}")
        raise TrainingError(f"Training failed: {e}")


# Export main classes and functions
__all__ = [
    'PatternModelTrainer',
    'TrainingMetrics',
    'TrainingError',
    'DataValidationError',
    'train_pattern_model',
    'set_seed'
]