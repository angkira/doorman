#!/usr/bin/env python3
"""Convert ONNX models to TorchScript for PyTorch ROCm backend"""

import torch
import onnx
from onnx2torch import convert
from pathlib import Path

def convert_model(onnx_path: Path, output_path: Path):
    """Convert ONNX model to TorchScript"""
    print(f"Converting {onnx_path.name}...")
    
    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))
    
    # Convert to PyTorch
    pytorch_model = convert(onnx_model)
    
    # Trace and save as TorchScript
    # Create dummy input based on model
    if "blazeface" in onnx_path.name:
        dummy_input = torch.randn(1, 3, 128, 128)
    elif "liveness" in onnx_path.name:
        dummy_input = torch.randn(1, 3, 112, 112)
    elif "mobilefacenet" in onnx_path.name:
        dummy_input = torch.randn(1, 3, 112, 112)
    else:
        raise ValueError(f"Unknown model: {onnx_path.name}")
    
    # Trace the model
    traced_model = torch.jit.trace(pytorch_model, dummy_input)
    
    # Save TorchScript model
    torch.jit.save(traced_model, str(output_path))
    print(f"✓ Saved to {output_path}")

def main():
    models_dir = Path.home() / ".local/share/doorman/models"
    
    models = [
        ("blazeface.onnx", "blazeface.pt"),
        ("liveness.onnx", "liveness.pt"),
        ("mobilefacenet.onnx", "mobilefacenet.pt"),
    ]
    
    for onnx_name, pt_name in models:
        onnx_path = models_dir / onnx_name
        output_path = models_dir / pt_name
        
        if not onnx_path.exists():
            print(f"⚠️  {onnx_name} not found, skipping")
            continue
        
        try:
            convert_model(onnx_path, output_path)
        except Exception as e:
            print(f"✗ Failed to convert {onnx_name}: {e}")
    
    print("\nConversion complete!")
    print("Models saved in:", models_dir)

if __name__ == "__main__":
    main()
