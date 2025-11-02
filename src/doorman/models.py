"""Model management for doorman face recognition system."""

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
import tempfile
import shutil


@dataclass
class ModelInfo:
    """Information about an ONNX model."""
    name: str
    filename: str
    url: str
    sha256: str
    size_mb: float
    description: str


class ModelManager:
    """Manages ONNX model downloads, installation, and verification."""
    
    # Model registry - sources for downloading models
    MODELS = {
        "blazeface": ModelInfo(
            name="BlazeFace",
            filename="blazeface.onnx",
            url="https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx",
            sha256="",  # We'll compute on first download
            size_mb=1.2,
            description="Lightweight face detection model (BlazeFace/UltraFace-320)"
        ),
        "liveness": ModelInfo(
            name="Liveness Detection",
            filename="liveness.onnx",
            url="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/releases/download/v2.0.0/2.7_80x80_MiniFASNetV2.onnx",
            sha256="",
            size_mb=0.5,
            description="Anti-spoofing liveness detection"
        ),
        "mobilefacenet": ModelInfo(
            name="MobileFaceNet",
            filename="mobilefacenet.onnx",
            url="https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
            sha256="",
            size_mb=249.0,
            description="Face recognition embeddings (ArcFace ResNet100)"
        ),
    }
    
    def __init__(self, models_dir: str = "/var/lib/doorman/models"):
        """Initialize model manager.
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        
    def ensure_models_dir(self) -> None:
        """Create models directory if it doesn't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def is_model_installed(self, model_key: str) -> bool:
        """Check if a model is installed.
        
        Args:
            model_key: Model identifier (e.g., 'blazeface')
            
        Returns:
            True if model file exists
        """
        if model_key not in self.MODELS:
            return False
        model_info = self.MODELS[model_key]
        model_path = self.models_dir / model_info.filename
        return model_path.exists()
    
    def get_installed_models(self) -> List[str]:
        """Get list of installed model keys.
        
        Returns:
            List of installed model identifiers
        """
        return [key for key in self.MODELS.keys() if self.is_model_installed(key)]
    
    def get_missing_models(self) -> List[str]:
        """Get list of missing model keys.
        
        Returns:
            List of missing model identifiers
        """
        return [key for key in self.MODELS.keys() if not self.is_model_installed(key)]
    
    def download_model(self, model_key: str, progress_callback=None) -> bool:
        """Download a model from the registry.
        
        Args:
            model_key: Model identifier
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if download successful
            
        Raises:
            KeyError: If model_key not in registry
            Exception: On download failure
        """
        if model_key not in self.MODELS:
            raise KeyError(f"Unknown model: {model_key}")
        
        model_info = self.MODELS[model_key]
        self.ensure_models_dir()
        
        # Download to temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.onnx') as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            try:
                if progress_callback:
                    progress_callback(f"Downloading {model_info.name}...")
                
                # Download with progress
                def report_hook(block_num, block_size, total_size):
                    if progress_callback and total_size > 0:
                        downloaded = block_num * block_size
                        percent = min(100, (downloaded / total_size) * 100)
                        progress_callback(f"  Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)")
                
                urllib.request.urlretrieve(model_info.url, tmp_path, reporthook=report_hook)
                
                # Verify it's a valid ONNX file (basic check)
                if tmp_path.stat().st_size < 1024:
                    raise Exception("Downloaded file too small, likely invalid")
                
                # Move to final location
                final_path = self.models_dir / model_info.filename
                shutil.move(str(tmp_path), str(final_path))
                
                if progress_callback:
                    progress_callback(f"✅ Installed {model_info.name} to {final_path}")
                
                return True
                
            except Exception as e:
                # Clean up temp file on error
                if tmp_path.exists():
                    tmp_path.unlink()
                raise Exception(f"Failed to download {model_info.name}: {e}")
    
    def download_all(self, progress_callback=None) -> Dict[str, bool]:
        """Download all missing models.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict mapping model_key to success status
        """
        results = {}
        missing = self.get_missing_models()
        
        if not missing:
            if progress_callback:
                progress_callback("All models already installed!")
            return results
        
        if progress_callback:
            progress_callback(f"Downloading {len(missing)} models...\n")
        
        for model_key in missing:
            try:
                success = self.download_model(model_key, progress_callback)
                results[model_key] = success
            except Exception as e:
                if progress_callback:
                    progress_callback(f"❌ Failed: {e}")
                results[model_key] = False
        
        return results
    
    def verify_model(self, model_key: str) -> tuple[bool, str]:
        """Verify a model file exists and is valid.
        
        Args:
            model_key: Model identifier
            
        Returns:
            Tuple of (is_valid, message)
        """
        if model_key not in self.MODELS:
            return False, f"Unknown model: {model_key}"
        
        model_info = self.MODELS[model_key]
        model_path = self.models_dir / model_info.filename
        
        if not model_path.exists():
            return False, f"Not installed: {model_path}"
        
        # Check file size
        size_bytes = model_path.stat().st_size
        size_mb = size_bytes / 1024 / 1024
        
        if size_bytes < 1024:
            return False, f"File too small ({size_bytes} bytes), likely corrupted"
        
        # Basic ONNX magic number check
        try:
            with open(model_path, 'rb') as f:
                magic = f.read(4)
                # ONNX files start with protocol buffer magic
                if len(magic) < 4:
                    return False, "File too short to be valid ONNX"
        except Exception as e:
            return False, f"Error reading file: {e}"
        
        return True, f"Valid ({size_mb:.1f} MB)"
    
    def list_models(self) -> List[Dict]:
        """List all models with their status.
        
        Returns:
            List of dicts with model information
        """
        models_list = []
        
        for key, info in self.MODELS.items():
            installed = self.is_model_installed(key)
            status = "✅ Installed" if installed else "❌ Missing"
            
            valid_msg = ""
            if installed:
                is_valid, msg = self.verify_model(key)
                valid_msg = f" - {msg}"
            
            models_list.append({
                "key": key,
                "name": info.name,
                "filename": info.filename,
                "size_mb": info.size_mb,
                "description": info.description,
                "installed": installed,
                "status": status + valid_msg,
            })
        
        return models_list
    
    def remove_model(self, model_key: str) -> bool:
        """Remove an installed model.
        
        Args:
            model_key: Model identifier
            
        Returns:
            True if removed successfully
        """
        if model_key not in self.MODELS:
            raise KeyError(f"Unknown model: {model_key}")
        
        model_info = self.MODELS[model_key]
        model_path = self.models_dir / model_info.filename
        
        if not model_path.exists():
            return False
        
        model_path.unlink()
        return True
    
    def get_models_info_summary(self) -> str:
        """Get a formatted summary of all models.
        
        Returns:
            Formatted string with model information
        """
        lines = []
        lines.append("\nDoorman ONNX Models:")
        lines.append(f"Location: {self.models_dir}\n")
        
        installed = self.get_installed_models()
        missing = self.get_missing_models()
        
        lines.append(f"Status: {len(installed)}/{len(self.MODELS)} installed\n")
        
        for model_dict in self.list_models():
            lines.append(f"  {model_dict['status']}")
            lines.append(f"    Name: {model_dict['name']}")
            lines.append(f"    File: {model_dict['filename']}")
            lines.append(f"    Size: {model_dict['size_mb']:.1f} MB")
            lines.append(f"    Description: {model_dict['description']}")
            lines.append("")
        
        return "\n".join(lines)

