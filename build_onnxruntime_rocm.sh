#!/bin/bash
# Build ONNX Runtime with ROCm support locally
# This avoids the executable stack issue in pre-built wheels

set -e

BUILD_DIR="${HOME}/onnxruntime-build"
INSTALL_DIR="${HOME}/.local/lib/onnxruntime-rocm-local"
# Pinned GPU stack — MUST match the `ort` crate so load-dynamic stays ABI-compatible:
#   ort 2.0.0-rc.12 (ort/api-24)  <->  ONNX Runtime 1.24.x  <->  ROCm 7.2.4 (gfx1103)
# Build the same ORT minor the crate's C API level (api-24 == ORT 1.24) expects.
ORT_VERSION="${ORT_VERSION:-v1.24.2}"   # ONNX Runtime 1.24.2 (matches ort 2.0.0-rc.12)
ROCM_VERSION="${ROCM_VERSION:-7.2.4}"   # latest production ROCm; gfx1103 (780M) supported in 7.x

echo "=== Building ONNX Runtime ${ORT_VERSION} with ROCm (target ROCm ${ROCM_VERSION}) ==="
echo "Build directory: ${BUILD_DIR}"
echo "Install directory: ${INSTALL_DIR}"
echo ""

# Check ROCm
if ! command -v hipcc &> /dev/null; then
    echo "ERROR: hipcc not found. Please install ROCm first."
    exit 1
fi

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
echo "Using ROCm at: ${ROCM_PATH}"

# Clone if needed
if [ ! -d "${BUILD_DIR}/onnxruntime" ]; then
    echo "Cloning ONNX Runtime..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    git clone --recursive https://github.com/microsoft/onnxruntime.git
    cd onnxruntime
    git checkout ${ORT_VERSION}
    git submodule update --init --recursive
else
    echo "Using existing source at ${BUILD_DIR}/onnxruntime"
    cd "${BUILD_DIR}/onnxruntime"
    git fetch
    git checkout ${ORT_VERSION}
    git submodule update --init --recursive
fi

# Create build directory
mkdir -p build/Release
cd build/Release

# Configure with CMake
# Key flags to avoid executable stack:
# - Use modern C++ standards
# - Disable assembly that requires exec stack
echo ""
echo "Configuring build..."
cmake ../.. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -Donnxruntime_BUILD_SHARED_LIB=ON \
    -Donnxruntime_USE_ROCM=ON \
    -Donnxruntime_ROCM_HOME="${ROCM_PATH}" \
    -DCMAKE_HIP_COMPILER="${ROCM_PATH}/bin/hipcc" \
    -DGPU_TARGETS="gfx1100;gfx1101;gfx1102;gfx1103" \
    -Donnxruntime_BUILD_UNIT_TESTS=OFF \
    -Donnxruntime_ENABLE_PYTHON=OFF \
    -DCMAKE_CXX_FLAGS="-Wl,-z,noexecstack" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-z,noexecstack" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,noexecstack"

# Build
echo ""
echo "Building (this may take 30-60 minutes)..."
NPROC=$(nproc)
cmake --build . --config Release --parallel $((NPROC / 2))

# Install
echo ""
echo "Installing to ${INSTALL_DIR}..."
cmake --install .

# Create symlinks
cd "${INSTALL_DIR}/lib"
ln -sf libonnxruntime.so.* libonnxruntime.so 2>/dev/null || true

echo ""
echo "=== Build Complete ==="
echo ""
echo "Library installed to: ${INSTALL_DIR}/lib"
echo ""
echo "To use with doorman, run:"
echo "  export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"
echo ""
echo "Or update Cargo.toml to use load-dynamic and set:"
echo "  ORT_DYLIB_PATH=${INSTALL_DIR}/lib/libonnxruntime.so"
echo ""

# Verify no exec stack
echo "Verifying library (should show no exec stack):"
readelf -l "${INSTALL_DIR}/lib/libonnxruntime.so" 2>/dev/null | grep -E "GNU_STACK|RWE" || echo "  ✓ No executable stack required"
