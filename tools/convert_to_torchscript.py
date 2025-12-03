#!/usr/bin/env python3
"""Convert ONNX models to TorchScript for PyTorch backend"""

import torch
import torch.onnx
import onnx
from pathlib import Path
import sys

def convert_onnx_to_torchscript(onnx_path: Path, output_path: Path, input_shape: tuple):
    """Convert ONNX model to TorchScript"""
    print(f"Converting {onnx_path.name} to TorchScript...")
    
    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))
    
    # Create a dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to TorchScript using trace
    # We'll use ONNX -> PyTorch conversion
    import onnx2torch
    pytorch_model = onnx2torch.convert(str(onnx_path))
    
    # Trace the model
    traced_model = torch.jit.trace(pytorch_model, dummy_input)
    
    # Save TorchScript model
    traced_model.save(str(output_path))
    print(f"✓ Saved TorchScript model to {output_path}")

def main():
    models_dir = Path.home() / ".local/share/doorman/models"
    
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        sys.exit(1)
    
    # Model configurations: (onnx_name, torchscript_name, input_shape)
    models = [
        ("blazeface.onnx", "blazeface.pt", (1, 3, 128, 128)),
        ("liveness.onnx", "liveness.pt", (1, 3, 80, 80)),
        ("mobilefacenet.onnx", "mobilefacenet.pt", (1, 3, 112, 112)),
    ]
    
    for onnx_name, ts_name, input_shape in models:
        onnx_path = models_dir / onnx_name
        output_path = models_dir / ts_name
        
        if not onnx_path.exists():
            print(f"Warning: ONNX model not found: {onnx_path}")
            continue
        
        try:
            convert_onnx_to_torchscript(onnx_path, output_path, input_shape)
        except Exception as e:
            print(f"Error converting {onnx_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Check for required packages
    try:
        import onnx2torch
    except ImportError:
        print("Installing onnx2torch...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "onnx2torch"], check=True)
        import onnx2torch
    
    main()
