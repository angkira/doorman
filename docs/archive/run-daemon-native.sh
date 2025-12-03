#!/bin/bash
set -e

# Set Python path to include venv
export PYTHONPATH="/home/angkira/Home/doorman/.venv/lib/python3.12/site-packages:$PYTHONPATH"

# Set library paths (system ONNX Runtime - no executable stack issues)
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export ORT_DYLIB_PATH="/usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.21.0"

# AMD GPU settings
export HSA_OVERRIDE_GFX_VERSION=11.0.1
export HIP_VISIBLE_DEVICES=0
export GPU_MAX_HW_QUEUES=1
export ORT_LOG_LEVEL=3

echo "Starting doorman daemon with Native PyTorch backend..."
echo "Python path: $PYTHONPATH"
echo "ORT library: $ORT_DYLIB_PATH"
echo ""

# Run daemon (will use user mode if not root)
exec ./target/release/doormand --config tools/configs/doorman-torch-native.toml
