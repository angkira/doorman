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
    input_shape: Optional[tuple] = None  # Expected input shape (H, W, C) for validation
    output_size: Optional[int] = None    # Expected embedding/output size


class ModelManager:
    """Manages ONNX model downloads, installation, and verification."""
    
    @staticmethod
    def get_default_models_dir() -> Path:
        """Get the default models directory based on context.
        
        Returns:
            Path to models directory (local for dev, system for production)
        """
        import os
        
        system_dir = Path("/var/lib/doorman/models")
        
        # If running as root, always use system directory
        if os.geteuid() == 0:
            return system_dir
        
        # If system directory exists and we can write to it, use it
        try:
            if system_dir.exists() and os.access(system_dir, os.W_OK):
                return system_dir
        except PermissionError:
            # Can't access system directory, fall through to local
            pass
        
        # For development, use local directory
        # Try to find project root (where doorman.toml is)
        project_root = Path(__file__).parent.parent.parent
        if (project_root / "doorman.toml").exists():
            return project_root / "data" / "models"
        
        # Fallback to current directory
        return Path.cwd() / "data" / "models"
    
    # Model registry - Public HuggingFace models that don't require authentication
    MODELS = {
        "blazeface": ModelInfo(
            name="BlazeFace",
            filename="blazeface.onnx",
            url="https://huggingface.co/OAKwood/BlazeFace/resolve/main/blazeface.onnx",
            sha256="f4051d0e8a9a2621901ce57476c7c508d786505074934b6f711f02153f5c9a4e",
            size_mb=0.5,
            description="Lightweight face detection (Google MediaPipe), ~2ms on CPU",
            input_shape=(128, 128, 3),  # Input: 128x128 RGB
            output_size=None  # Outputs bounding boxes + landmarks
        ),
        "liveness": ModelInfo(
            name="Liveness Detection",
            filename="liveness.onnx",
            url="https://huggingface.co/onnx-community/LivenessNet/resolve/main/LivenessNet.onnx",
            sha256="placeholder_sha256_checksum",
            size_mb=1.2,
            description="Anti-spoofing model for detecting real vs fake faces",
            input_shape=(80, 80, 3),
            output_size=3
        ),
        "mobilefacenet": ModelInfo(
            name="MobileFaceNet",
            filename="mobilefacenet.onnx",
            url="https://huggingface.co/onnx-community/mobilefacenet/resolve/main/mobilefacenet.onnx",
            sha256="placeholder_sha256_checksum",
            size_mb=4.2,
            description="Face recognition model for generating 512-d embeddings",
            input_shape=(112, 112, 3),
            output_size=512
        ),
    }
    
    def __init__(self, models_dir: Optional[str] = None):
        """Initialize model manager.
        
        Args:
            models_dir: Directory to store models (defaults to auto-detected location)
        """
        if models_dir is None:
            self.models_dir = self.get_default_models_dir()
        else:
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
        
        # Check if URL is available (should always be available now)
        if not model_info.url:
            raise Exception(
                f"{model_info.name} has no download URL configured.\n"
                f"This is a configuration error - all models should have URLs.\n"
                f"Expected location: {self.models_dir / model_info.filename}"
            )
        
        self.ensure_models_dir()
        final_path = self.models_dir / model_info.filename

        # Check if URL is from HuggingFace
        if model_info.url.startswith("hf://"):
            # Parse HF URL: hf://user/repo/file.onnx
            hf_url = model_info.url[5:]  # Remove "hf://"
            parts = hf_url.split("/")
            if len(parts) < 3:
                raise Exception(f"Invalid HF URL format: {model_info.url}. Expected: hf://user/repo/file.onnx")

            # Reconstruct: repo_id = "user/repo", file_path = "file.onnx"
            repo_id = f"{parts[0]}/{parts[1]}"
            file_path = "/".join(parts[2:])

            if progress_callback:
                progress_callback(f"Downloading {model_info.name} from HuggingFace...")

            # Use HF CLI to download
            import subprocess
            try:
                # Download to temp dir then move
                with tempfile.TemporaryDirectory() as tmp_dir:
                    result = subprocess.run([
                        "hf", "download", repo_id, file_path,
                        "--local-dir", tmp_dir
                    ], capture_output=True, text=True, check=True)

                    # Find the downloaded file
                    downloaded_file = Path(tmp_dir) / file_path
                    if not downloaded_file.exists():
                        raise Exception(f"Downloaded file not found: {downloaded_file}")

                    # Move to final location with correct name
                    shutil.move(str(downloaded_file), str(final_path))

                    if progress_callback:
                        progress_callback(f"✅ Installed {model_info.name} to {final_path}")

                    return True

            except subprocess.CalledProcessError as e:
                raise Exception(f"HF download failed: {e.stderr}")

        # Standard HTTP/HTTPS download
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
