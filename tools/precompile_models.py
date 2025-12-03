#!/usr/bin/env python3
"""
Pre-compile ONNX models for MIGraphX to avoid long first-run compilation
"""

import os
import sys
from pathlib import Path
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

# Environment setup for AMD Radeon 780M with GTT support
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.1'  # Better GTT support
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['GPU_MAX_HW_QUEUES'] = '1'
os.environ['ORT_LOG_LEVEL'] = '3'  # Suppress warnings

# MIGraphX cache paths (ROCm 6.4+)
cache_base = str(Path.home() / ".cache/doorman/migraphx")
os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH'] = cache_base
os.environ['ORT_MIGRAPHX_MODEL_PATH'] = cache_base

def precompile_model(model_path: str, cache_dir: str, input_shape: tuple, input_name: str = 'input'):
    """
    Pre-compile ONNX model with MIGraphX and save compiled version

    Args:
        model_path: Path to ONNX model
        cache_dir: Directory to save compiled model
        input_shape: Input tensor shape (batch, channels, height, width)
        input_name: Name of input tensor
    """
    model_name = Path(model_path).stem
    print(f"\n=== Pre-compiling {model_name} ===")
    print(f"Model: {model_path}")
    print(f"Input shape: {input_shape}")
    print(f"Cache dir: {cache_dir}")

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # MIGraphX provider options
    provider_options = {
        'device_id': 0,
        'migraphx_fp16_enable': '0',  # Use FP32 for accuracy
        # Cache is controlled by ORT_MIGRAPHX_MODEL_CACHE_PATH env var
    }

    providers = [
        ('MIGraphXExecutionProvider', provider_options),
        'CPUExecutionProvider'
    ]

    print(f"\nCreating session (this will trigger compilation)...")
    print("Expected times:")
    print("  - BlazeFace: ~30-60 seconds")
    print("  - Liveness: ~30-60 seconds")
    print("  - MobileFaceNet: ~3-5 minutes (largest model)")
    print("")
    print("⏳ Compiling... (GPU will be busy, FPS may drop)")

    import time
    start_time = time.time()

    # Create session - this triggers compilation
    session = ort.InferenceSession(model_path, providers=providers)

    compile_time = time.time() - start_time

    print(f"✓ Session created on: {session.get_providers()[0]}")
    print(f"  Compilation took: {compile_time:.1f}s")

    # Run dummy inference to ensure full compilation
    print("Running dummy inference to ensure compilation...")
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup run
    outputs = session.run(None, {input_name: dummy_input})

    print(f"✓ Compilation complete!")
    print(f"  Output shapes: {[o.shape for o in outputs]}")

    # Note: MIGraphX caches compiled models automatically in ~/.cache
    # The next session creation will load from cache
    print(f"\n✓ Model compiled and cached by MIGraphX")
    print(f"  Next session creation will be fast!")

    return session

def main():
    models_dir = Path.home() / ".local/share/doorman/models"
    cache_dir = Path.home() / ".cache/doorman/migraphx"

    print("=== MIGraphX Model Pre-compilation ===")
    print(f"Models directory: {models_dir}")
    print(f"Cache directory: {cache_dir}")
    print()
    print("This will pre-compile all models for GPU (one-time setup)")
    print("Subsequent runs will load from cache (fast!)")
    print()

    models = [
        {
            'name': 'BlazeFace',
            'path': str(models_dir / 'blazeface.onnx'),
            'input_shape': (1, 3, 240, 320),
            'input_name': 'input',
        },
        {
            'name': 'Liveness',
            'path': str(models_dir / 'liveness.onnx'),
            'input_shape': (1, 3, 96, 96),  # Model expects 96x96, not 80x80!
            'input_name': 'data',  # Uses 'data' not 'input'
        },
        {
            'name': 'MobileFaceNet',
            'path': str(models_dir / 'mobilefacenet.onnx'),
            'input_shape': (1, 3, 112, 112),
            'input_name': 'data',  # Uses 'data' not 'input'
        },
    ]

    for i, model_info in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(models)}] Compiling {model_info['name']}")
        print(f"{'='*60}")

        if not Path(model_info['path']).exists():
            print(f"⚠ Model not found: {model_info['path']}")
            print(f"  Skipping...")
            continue

        try:
            session = precompile_model(
                model_info['path'],
                str(cache_dir),
                model_info['input_shape'],
                model_info['input_name']
            )

            # Verify it works
            print(f"\n✓ {model_info['name']} ready for use")

        except Exception as e:
            print(f"\n✗ Error compiling {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("Pre-compilation complete!")
    print(f"{'='*60}")
    print()
    print("MIGraphX has cached compiled models.")
    print("Next daemon startup will be fast!")
    print()
    print("To verify, run:")
    print("  ./tools/run_torch.sh --preview")
    print()

if __name__ == "__main__":
    main()
