#!/bin/bash
set -e

echo "=== Installing ONNX Runtime with MIGraphX Execution Provider ==="
echo "This is the official AMD solution for ROCm 7.x"
echo

# Critical workaround for Radeon 780M (gfx1100 iGPU)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Install MIGraphX-enabled ONNX Runtime
uv pip install onnxruntime-migraphx==1.23.0

echo
echo "✓ Installed onnxruntime-migraphx 1.23.0"
echo
echo "IMPORTANT: Set this environment variable before running:"
echo "export HSA_OVERRIDE_GFX_VERSION=11.0.0"
echo
echo "Test with:"
echo "python3 -c 'import onnxruntime; print(onnxruntime.get_available_providers())'"
