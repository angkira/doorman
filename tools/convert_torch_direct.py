#!/usr/bin/env python3
"""Load ONNX models directly in PyTorch for runtime execution"""

import torch
from pathlib import Path

def test_load_onnx_in_torch():
    """Test if PyTorch can load ONNX models directly"""
    models_dir = Path.home() / ".local/share/doorman/models"
    
    # PyTorch can load ONNX models at runtime using torch.onnx or onnxruntime
    # But for TorchScript, we need to convert differently
    
    # Let's use a simpler approach: wrap ONNX runtime in Python
    try:
        import onnxruntime as ort
        print("✓ ONNX Runtime available")
        
        # Check if ROCm provider is available
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        if 'ROCMExecutionProvider' in providers:
            print("✓ ROCm execution provider available!")
            
            # Test loading a model
            blazeface_path = models_dir / "blazeface.onnx"
            session = ort.InferenceSession(
                str(blazeface_path),
                providers=['ROCMExecutionProvider', 'CPUExecutionProvider']
            )
            print(f"✓ Loaded {blazeface_path.name} with ROCm")
            print(f"  Inputs: {[i.name for i in session.get_inputs()]}")
            print(f"  Outputs: {[o.name for o in session.get_outputs()]}")
        else:
            print("⚠️  ROCm provider not available")
            print("Available providers:", providers)
    
    except ImportError:
        print("✗ ONNX Runtime not installed")
        print("Install with: pip install onnxruntime-rocm")

if __name__ == "__main__":
    test_load_onnx_in_torch()
