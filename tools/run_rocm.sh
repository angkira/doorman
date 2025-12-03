#!/bin/bash
set -e

# Set library paths for ONNX Runtime ROCm
export ORT_LIB_LOCATION="/home/angkira/Home/doorman/.venv/lib/python3.12/site-packages/onnxruntime/capi"
export LD_LIBRARY_PATH="$ORT_LIB_LOCATION:/opt/rocm/lib:/opt/rocm-7.0.2/lib:$LD_LIBRARY_PATH"

# Use CPU version for now (ROCm library has executable stack issues)
export ORT_DYLIB_PATH="$ORT_LIB_LOCATION/libonnxruntime.so.1.23.2"

# Required for gfx1103 (Radeon 780M) - even for CPU execution
export HSA_OVERRIDE_GFX_VERSION=11.0.0

echo "Running doorman with ONNX Runtime ROCm/MIGraphX..."
echo "Library path: $ORT_LIB_LOCATION"
echo "HSA override: $HSA_OVERRIDE_GFX_VERSION"
echo ""

./target/release/doormand --user --preview --config doorman-ort-rocm.toml "$@"
