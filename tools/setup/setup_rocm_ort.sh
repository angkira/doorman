#!/bin/bash
# Setup ONNX Runtime ROCm environment

# Create symlinks for compatibility
if [ ! -f /opt/rocm-7.0.2/lib/libhipblas.so.2 ]; then
    echo "Creating libhipblas.so.2 symlink..."
    sudo ln -sf /opt/rocm-7.0.2/lib/libhipblas.so.3 /opt/rocm-7.0.2/lib/libhipblas.so.2
fi

# Export environment
export ROCM_PATH=/opt/rocm-7.0.2
export LD_LIBRARY_PATH=/opt/rocm-7.0.2/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=11.0.0

echo "ROCm environment configured:"
echo "  ROCM_PATH=$ROCM_PATH"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
