#!/bin/bash
set -e

# ROCm environment for AMD Radeon 780M iGPU (gfx1103)
# Use 11.0.1 for better GTT memory support (vs 11.0.0)
export HSA_OVERRIDE_GFX_VERSION=11.0.1

# Force HIP to use unified memory (GTT + VRAM)
export HIP_VISIBLE_DEVICES=0
export GPU_MAX_HW_QUEUES=1  # Reduce concurrent ops to save VRAM

# MIGraphX model cache (speeds up subsequent runs)
export ORT_MIGRAPHX_MODEL_CACHE_PATH="$HOME/.cache/doorman/migraphx"
export ORT_MIGRAPHX_MODEL_PATH="$HOME/.cache/doorman/migraphx"
mkdir -p "$ORT_MIGRAPHX_MODEL_CACHE_PATH"

# Python virtual environment
VENV_PATH="$HOME/Home/doorman/.venv"
if [ -d "$VENV_PATH" ]; then
    export VIRTUAL_ENV="$VENV_PATH"
    export PATH="$VENV_PATH/bin:$PATH"
    echo "Using Python venv: $VENV_PATH"
else
    echo "Warning: venv not found at $VENV_PATH, using system python3"
fi

# ONNX Runtime library paths for ROCm
export ORT_LIB_LOCATION="$VENV_PATH/lib/python3.12/site-packages/onnxruntime/capi"
export LD_LIBRARY_PATH="$ORT_LIB_LOCATION:/opt/rocm/lib:$LD_LIBRARY_PATH"

# Suppress ONNX Runtime warnings
export ORT_LOG_LEVEL=3

# Models directory
export MODELS_DIR="$HOME/.local/share/doorman/models"

echo "=== Doorman PyTorch Backend + ROCm ==="
echo "HSA override: $HSA_OVERRIDE_GFX_VERSION"
echo "Models dir: $MODELS_DIR"
echo "ONNX Runtime lib: $ORT_LIB_LOCATION"
echo ""

# Build with torch backend feature
echo "Building with backend-torch feature..."
cargo build --release --features backend-torch

echo ""
echo "Starting daemon with torch backend..."
./target/release/doormand --user --config tools/configs/doorman-torch.toml "$@"
