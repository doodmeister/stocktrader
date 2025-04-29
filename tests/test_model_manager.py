import pytest
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from model_manager import ModelManager, ModelMetadata, ModelError, ModelNotFoundError

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def model_manager(tmp_path):
    return ModelManager(base_directory=str(tmp_path))

@pytest.fixture
def sample_model():
    return SimpleModel()

def test_save_model(model_manager, sample_model):
    path = model_manager.save_model(sample_model, metadata={"test": True})
    assert Path(path).exists()
    
def test_load_model(model_manager, sample_model):
    # Save and load model
    save_path = model_manager.save_model(sample_model)
    loaded_model, metadata = model_manager.load_model(SimpleModel, save_path)
    
    # Verify model structure
    assert isinstance(loaded_model, SimpleModel)
    assert isinstance(metadata, ModelMetadata)
    
def test_model_not_found(model_manager):
    with pytest.raises(ModelNotFoundError):
        model_manager.load_model(SimpleModel, "nonexistent.pth")
        
def test_metadata_validation():
    with pytest.raises(ValueError):
        ModelMetadata(version=123, saved_at=datetime.now().isoformat())
        
def test_cleanup_old_models(model_manager, sample_model):
    # Create multiple versions
    versions = []
    for _ in range(10):
        path = model_manager.save_model(sample_model)
        versions.append(path)
    
    # Cleanup keeping 5 versions
    model_manager.cleanup_old_models(keep_versions=5)
    remaining = list(Path(model_manager.base_directory).glob("*.pth"))
    assert len(remaining) == 5