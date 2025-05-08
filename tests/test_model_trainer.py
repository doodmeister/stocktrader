"""Unit tests for ModelTrainer"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier
import joblib

from train.ml_model_trainer import (
    ModelTrainer, 
    FeatureConfig,
    TrainingParams,
    ModelType,
    ValidationError,
    ModelError
)

@pytest.fixture
def sample_ohlcv_data():
    """Generate realistic sample OHLCV data"""
    dates = pd.date_range(
        start='2024-01-01', 
        periods=200, 
        freq='D'
    )
    
    # Generate more realistic price movements
    base_price = 100
    prices = np.random.normal(0, 0.02, 200).cumsum()
    prices = np.exp(prices) * base_price
    
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, 200)),
        'High': prices * (1 + abs(np.random.normal(0, 0.01, 200))),
        'Low': prices * (1 - abs(np.random.normal(0, 0.01, 200))),
        'Close': prices,
        'Volume': np.random.lognormal(10, 1, 200)
    }, index=dates)
    
    return df

@pytest.fixture
def model_trainer():
    """Create ModelTrainer with test configuration"""
    config = Mock()
    config.MODEL_DIR = Path('tests/test_models')
    return ModelTrainer(config)

def test_feature_config_validation():
    """Test FeatureConfig validation"""
    # Valid config
    config = FeatureConfig()
    config.validate()
    
    # Invalid configs
    with pytest.raises(ValidationError):
        config = FeatureConfig(PRICE_FEATURES=[])
        config.validate()
        
    with pytest.raises(ValidationError):
        config = FeatureConfig(ROLLING_WINDOWS=[])
        config.validate()
        
    with pytest.raises(ValidationError):
        config = FeatureConfig(TARGET_HORIZON=0)
        config.validate()
        
    with pytest.raises(ValidationError):
        config = FeatureConfig(MIN_SAMPLES=10)
        config.validate()

def test_training_params_validation():
    """Test TrainingParams validation"""
    # Valid params
    params = TrainingParams()
    params.validate()
    
    # Invalid params
    with pytest.raises(ValidationError):
        params = TrainingParams(n_estimators=0)
        params.validate()
        
    with pytest.raises(ValidationError):
        params = TrainingParams(max_depth=0)
        params.validate()
        
    with pytest.raises(ValidationError):
        params = TrainingParams(min_samples_split=1)
        params.validate()
        
    with pytest.raises(ValidationError):
        params = TrainingParams(cv_folds=1)
        params.validate()

def test_validate_input_data(model_trainer, sample_ohlcv_data):
    """Test input data validation"""
    # Valid data should pass
    model_trainer.validate_input_data(sample_ohlcv_data)
    
    # Test missing columns
    with pytest.raises(ValidationError):
        bad_df = sample_ohlcv_data.drop('Close', axis=1)
        model_trainer.validate_input_data(bad_df)
    
    # Test insufficient samples
    with pytest.raises(ValidationError):
        bad_df = sample_ohlcv_data.iloc[:50]
        model_trainer.validate_input_data(bad_df)
    
    # Test null values
    with pytest.raises(ValidationError):
        bad_df = sample_ohlcv_data.copy()
        bad_df.iloc[0, 0] = None
        model_trainer.validate_input_data(bad_df)

def test_feature_engineering(model_trainer, sample_ohlcv_data):
    """Test feature engineering pipeline"""
    df_processed = model_trainer.feature_engineering(sample_ohlcv_data)
    
    # Check expected features exist
    expected_features = model_trainer.get_feature_columns()
    assert all(f in df_processed.columns for f in expected_features)
    
    # Check target creation
    assert 'target' in df_processed.columns
    assert df_processed['target'].isin([0, 1]).all()
    
    # Check no null values
    assert not df_processed.isnull().any().any()
    
    # Check proper length after dropna
    expected_len = len(sample_ohlcv_data) - max(
        model_trainer.feature_config.ROLLING_WINDOWS
    ) - model_trainer.feature_config.TARGET_HORIZON
    assert len(df_processed) == expected_len

def test_train_model(model_trainer, sample_ohlcv_data):
    """Test model training pipeline"""
    model, metrics, cm, report = model_trainer.train_model(sample_ohlcv_data)
    
    # Check model type
    assert isinstance(model, RandomForestClassifier)
    
    # Check metrics structure
    expected_metric_types = ['train_metrics', 'test_metrics', 'final_metrics']
    assert all(m in metrics for m in expected_metric_types)
    
    expected_metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric_type in expected_metric_types:
        assert all(m in metrics[metric_type] for m in expected_metrics)
    
    # Check metric values are reasonable
    for metric_type in expected_metric_types:
        for metric in expected_metrics:
            assert 0 <= metrics[metric_type][metric]['mean'] <= 1
            assert metrics[metric_type][metric]['std'] >= 0
    
    # Check confusion matrix
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (2, 2)  # Binary classification
    
    # Check classification report
    assert isinstance(report, str)
    assert 'precision' in report
    assert 'recall' in report
    assert 'f1-score' in report

def test_save_model(model_trainer, sample_ohlcv_data):
    """Test model saving functionality"""
    model, metrics, _, _ = model_trainer.train_model(sample_ohlcv_data)
    
    path = model_trainer.save_model(model, "AAPL", "1d")
    assert path.exists()
    
    # Check metadata file
    metadata_path = path.with_suffix('.json')
    assert metadata_path.exists()
    
    # Load and verify saved artifacts
    artifacts = joblib.load(path)
    assert 'model' in artifacts
    assert 'scaler' in artifacts
    assert 'features' in artifacts
    assert 'metadata' in artifacts
    
    metadata = artifacts['metadata']
    assert metadata['symbol'] == "AAPL"
    assert metadata['interval'] == "1d"
    assert 'timestamp' in metadata
    assert 'feature_config' in metadata
    assert 'training_params' in metadata
    
    # Clean up
    path.unlink()
    metadata_path.unlink()
    
def test_error_handling(model_trainer):
    """Test error handling"""
    # Invalid model type
    with pytest.raises(ValidationError):
        model_trainer.save_model(Mock(), "AAPL", "1d")
    
    # Missing parameters
    with pytest.raises(ValidationError):
        model_trainer.save_model(RandomForestClassifier(), "", "")
        
    # Invalid data
    with pytest.raises(ValidationError):
        model_trainer.validate_input_data(pd.DataFrame())