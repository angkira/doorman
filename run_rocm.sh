#!/bin/bash
# run_rocm.sh — Launch doorman on the AMD iGPU (Radeon 780M, gfx1103) via ROCm.
#
# VERIFIED WORKING 2026-06-08: detection runs on the iGPU (ROCm EP), GPU% 12-21%,
# no MIOpen/HIP errors. Stack: container-built ONNX Runtime 1.22.2 + ROCm EP
# (last ORT with the ROCm EP; api-22) against host ROCm 7.2.2.
#
# Key lessons baked in below:
#  - Use a SINGLE device filter: ROCR_VISIBLE_DEVICES=1 makes the iGPU the only
#    visible device (HIP device 0). Do NOT also set HIP_VISIBLE_DEVICES=1 — that
#    double-filters and HIP reports "no ROCm-capable device". We unset it here so
#    the global ~/.zshrc value (which protects the dGPU) can't leak in.
#  - HIP_PATH/ROCM_PATH must point at the ROCm install so MIOpen's runtime
#    (hiprtc) kernel JIT can find hip/hip_runtime.h (else MaxPool fails).
#  - dGPU (R9700, device 0) runs the user's training — this script never touches
#    it; ROCR_VISIBLE_DEVICES=1 isolates everything to the iGPU.

set -euo pipefail

ROCM="${ROCM_PATH:-/opt/rocm-7.2.2}"
ORT_DIR="$HOME/.local/lib/onnxruntime-rocm-local/lib"

# --- iGPU isolation (single device filter) ---
unset HIP_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=1                 # iGPU only -> becomes HIP device 0
export HSA_OVERRIDE_GFX_VERSION="11.0.0"      # spoof gfx1103 -> gfx1100 (prebuilt rocBLAS/MIOpen kernels)

# --- ROCm + ORT runtime ---
export HIP_PATH="$ROCM"
export ROCM_PATH="$ROCM"
export ORT_DYLIB_PATH="$ORT_DIR/libonnxruntime.so"            # ORT 1.22.2 + ROCm EP (api-22)
export LD_LIBRARY_PATH="$ORT_DIR:$ROCM/lib:$ROCM/lib64:${LD_LIBRARY_PATH:-}"

# --- MIOpen: fast (cached) kernel selection + persistent JIT cache ---
export MIOPEN_FIND_MODE=3
export MIOPEN_USER_DB_PATH="$HOME/.cache/miopen"
mkdir -p "$MIOPEN_USER_DB_PATH"

export DOORMAN_CONFIG="${DOORMAN_CONFIG:-doorman.toml}"

if [[ ! -x ./target/release/doormand ]]; then
    echo "ERROR: ./target/release/doormand not found. Build it with:" >&2
    echo "  cargo build --release --no-default-features --features backend-ort-rocm,camera-v4l2,camera-ffmpeg --bin doormand" >&2
    exit 1
fi

echo "Starting doorman on AMD iGPU (gfx1103 via ROCm)..."
echo "  ORT: $ORT_DYLIB_PATH"
echo "  ROCm: $ROCM   device: ROCR_VISIBLE_DEVICES=1 (iGPU)   override: gfx$HSA_OVERRIDE_GFX_VERSION"
echo "  (first run JIT-compiles MIOpen kernels into $MIOPEN_USER_DB_PATH; subsequent runs are fast)"

# --user => XDG paths (models in ~/.local/share/doorman/models); --device rocm => ROCm EP.
exec ./target/release/doormand --user --device rocm "$@"
