#!/bin/bash
set -e

echo "=== Installing ONNX Runtime with ROCm support ==="

# Download pre-built ONNX Runtime with ROCm
ONNX_VERSION="1.22.2"
ROCM_VERSION="6.2"

echo "Downloading ONNX Runtime ${ONNX_VERSION} with ROCm ${ROCM_VERSION}..."

# Check if already downloaded
if [ ! -f "/tmp/onnxruntime-linux-x64-rocm-${ONNX_VERSION}.tgz" ]; then
    wget -P /tmp https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-rocm-${ONNX_VERSION}.tgz
fi

echo "Extracting..."
cd /tmp
tar -xzf onnxruntime-linux-x64-rocm-${ONNX_VERSION}.tgz

echo "Installing to /usr/local..."
sudo cp -r onnxruntime-linux-x64-rocm-${ONNX_VERSION}/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-x64-rocm-${ONNX_VERSION}/include/* /usr/local/include/

echo "Updating ldconfig..."
sudo ldconfig

echo "Verifying installation..."
ldconfig -p | grep onnxruntime

echo ""
echo "✅ ONNX Runtime with ROCm support installed!"
echo ""
echo "Set environment:"
echo "  export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH"
echo "  export ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so"
