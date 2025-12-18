"""Unit tests for model management."""

import pytest
import tempfile
from pathlib import Path
from doorman.models import ModelManager, ModelInfo


@pytest.fixture
def temp_models_dir():
    """Create a temporary models directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def manager(temp_models_dir):
    """Create a ModelManager with temporary directory."""
    return ModelManager(str(temp_models_dir))


def test_model_registry_not_empty():
    """Test that model registry has entries."""
    assert len(ModelManager.MODELS) > 0
    assert "blazeface" in ModelManager.MODELS
    assert "mobilefacenet" in ModelManager.MODELS


def test_model_info_structure():
    """Test ModelInfo structure is correct."""
    for key, info in ModelManager.MODELS.items():
        assert isinstance(info, ModelInfo)
        assert info.name
        assert info.filename
        assert info.description
        assert info.size_mb >= 0


def test_ensure_models_dir(manager, temp_models_dir):
    """Test models directory creation."""
    manager.ensure_models_dir()
    assert temp_models_dir.exists()
    assert temp_models_dir.is_dir()


def test_is_model_installed_missing(manager):
    """Test checking for non-existent model."""
    assert not manager.is_model_installed("blazeface")


def test_is_model_installed_exists(manager, temp_models_dir):
    """Test checking for existing model."""
    # Create a fake model file
    model_file = temp_models_dir / "blazeface.onnx"
    model_file.write_bytes(b"fake model data")
    
    assert manager.is_model_installed("blazeface")


def test_get_installed_models_empty(manager):
    """Test getting installed models when none exist."""
    installed = manager.get_installed_models()
    assert isinstance(installed, list)
    assert len(installed) == 0


def test_get_installed_models_some_exist(manager, temp_models_dir):
    """Test getting installed models when some exist."""
    # Create fake model files
    (temp_models_dir / "blazeface.onnx").write_bytes(b"fake1")
    (temp_models_dir / "mobilefacenet.onnx").write_bytes(b"fake2")
    
    installed = manager.get_installed_models()
    assert "blazeface" in installed
    assert "mobilefacenet" in installed
    assert "liveness" not in installed  # This one doesn't exist


def test_get_missing_models_all_missing(manager):
    """Test getting missing models when none exist."""
    missing = manager.get_missing_models()
    assert isinstance(missing, list)
    assert len(missing) == len(ModelManager.MODELS)


def test_get_missing_models_some_exist(manager, temp_models_dir):
    """Test getting missing models when some exist."""
    # Create one model file
    (temp_models_dir / "blazeface.onnx").write_bytes(b"fake")
    
    missing = manager.get_missing_models()
    assert "blazeface" not in missing
    assert "mobilefacenet" in missing
    assert "liveness" in missing


def test_download_model_invalid_key(manager):
    """Test downloading with invalid model key."""
    with pytest.raises(KeyError):
        manager.download_model("nonexistent_model")


def test_list_models_conversion():
    """Test that list(MODELS) works correctly."""
    models_list = list(ModelManager.MODELS)
    assert isinstance(models_list, list)
    assert len(models_list) > 0
    assert all(isinstance(key, str) for key in models_list)
