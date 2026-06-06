#!/bin/bash
# Run doorman with patched AMD ROCm ONNX Runtime (no exec stack)

export ORT_DYLIB_PATH="$HOME/.local/lib/onnxruntime-rocm-patched/libonnxruntime.so"
export LD_LIBRARY_PATH="$HOME/.local/lib/onnxruntime-rocm-patched:$LD_LIBRARY_PATH"
export HSA_OVERRIDE_GFX_VERSION="11.0.0"
export DOORMAN_CONFIG="${DOORMAN_CONFIG:-doorman-lightweight.toml}"

# MIOpen optimization settings
export MIOPEN_FIND_MODE=3  # Fast mode - use cached kernels
export MIOPEN_USER_DB_PATH="$HOME/.cache/miopen"

echo "Starting doorman with AMD ROCm GPU + OpenCL preprocessing..."
echo "Library: $HOME/.local/lib/onnxruntime-rocm-patched"

exec ./target/release/doormand "$@"
