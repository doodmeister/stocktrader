"""Unit tests for ModelTrainer"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from data.model_trainer import ModelTrainer, FeatureConfig, ModelType

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    df = pd.DataFrame({
        'Open': np.random.random(200) * 100,
        'High': np.random.random(200) * 100,
        'Low': np.random.random(200) * 100,
        'Close': np.random.random(200) * 100,
        'Volume': np.random.random(200) * 1000000
    }, index=dates)
    return df

@pytest.fixture
def model_trainer():
    """Create ModelTrainer instance with test config"""
    config = Mock()
    config.MODEL_DIR = Path('tests/test_models')
    return ModelTrainer(config)

def test_validate_input_data(model_trainer, sample_ohlcv_data):
    """Test input data validation"""
    # Valid data should pass
    model_trainer.validate_input_data(sample_ohlcv_data)
    
    # Missing columns should raise error
    bad_df = sample_ohlcv_data.drop('Close', axis=1)
    with pytest.raises(ValueError):
        model_trainer.validate_input_data(bad_df)
        
    # Null values should raise error
    bad_df = sample_ohlcv_data.copy()
    bad_df.iloc[0, 0] = None
    with pytest.raises(ValueError):
        model_trainer.validate_input_data(bad_df)

def test_feature_engineering(model_trainer, sample_ohlcv_data):
    """Test feature engineering pipeline"""
    df_processed = model_trainer.feature_engineering(sample_ohlcv_data)
    
    # Check expected features exist
    expected_features = [
        'returns',
        'volatility_5',
        'sma_5',
        'volume_sma_5',
        'target'
    ]
    assert all(f in df_processed.columns for f in expected_features)
    
    # Check no null values
    assert not df_processed.isnull().any().any()
    
    # Check target is binary
    assert df_processed['target'].isin([0, 1]).all()

def test_train_model(model_trainer, sample_ohlcv_data):
    """Test model training pipeline"""
    model, metrics = model_trainer.train_model(
        sample_ohlcv_data,
        ModelType.RANDOM_FOREST
    )
    
    # Check metrics structure
    expected_metrics = [
        'train_accuracy_mean',
        'train_accuracy_std',
        'test_accuracy_mean',
        'test_accuracy_std'
    ]
    assert all(m in metrics for m in expected_metrics)
    
    # Check metric values are reasonable
    assert 0 <= metrics['train_accuracy_mean'] <= 1
    assert 0 <= metrics['test_accuracy_mean'] <= 1
    
    # Invalid model type should raise error
    with pytest.raises(ValueError):
        model_trainer.train_model(sample_ohlcv_data, "invalid_model")

def test_save_model(model_trainer, sample_ohlcv_data):
    """Test model saving functionality"""
    model, _ = model_trainer.train_model(sample_ohlcv_data)
    
    path = model_trainer.save_model(model, "AAPL", "1d")
    assert path.exists()
    
    # Load and verify saved model package
    saved_package = joblib.load(path)
    assert 'model' in saved_package
    assert 'scaler' in saved_package
    assert 'features' in saved_package
    assert 'timestamp' in saved_package
    
    # Clean up
    path.unlink()