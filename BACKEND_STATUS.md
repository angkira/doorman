# Backend Integration Status

**Date**: 2025-12-02
**Hardware**: AMD Ryzen 7 8700G, Radeon 780M iGPU (gfx1103), RTX 4060 Ti dGPU
**ROCm**: 7.0.2

## Summary

Реализовано 4 ML бэкенда, работает iGPU через Python subprocess.

## Backends

### 1. Tract (Pure Rust CPU) ✅ Default
- **Performance**: ~3-5 FPS detection
- **Pros**: No deps, portable, stable
- **Cons**: Slow
- **Build**: `cargo build --features backend-tract`

### 2. ONNX Runtime CPU ✅ Working
- **Performance**: ~5.5 FPS (+10% vs Tract)
- **Pros**: AVX optimizations
- **Cons**: C++ dependency
- **Build**: `cargo build --features backend-ort-cpu`

### 3. ONNX Runtime + ROCm (Rust) ❌ Failed
- **Attempted**: Direct `ort` crate binding
- **Why failed**:
  - `download-binaries` lacks ROCm EP
  - Version hell (system 1.21 vs required 1.22)
  - Executable stack issues
  - Requires custom ONNX Runtime build from source
- **Decision**: Use Python subprocess instead

### 4. ONNX Runtime + MIGraphX (Python) ✅ **CURRENT**
- **Performance**: ~7.2 FPS (2.4x faster than Tract)
- **Architecture**: Rust → JSON-RPC → Python → ONNX Runtime → MIGraphX → iGPU
- **Files**:
  - `daemon/src/ml/torch_backend.rs`
  - `daemon/src/ml/torch_inference.py`
  - `tools/run_torch.sh`
- **Pros**: Actually uses iGPU, working
- **Cons**: Python overhead (~10ms IPC)
- **Build**: `cargo build --features backend-torch`
- **Run**: `./tools/run_torch.sh --user --preview`

### 5. MIGraphX (isolated benchmark)
- **Raw performance**: 731 FPS BlazeFace, 2918 FPS Liveness
- **Note**: Not integrated, benchmark only

## Key Issues Solved

1. **ONNX Runtime Distribution** - No pre-built ROCm binaries → Use Python package
2. **ROCm EP Deprecated** - ROCm 7.1+ drops ONNX RT support → Stick to 7.0.x
3. **Library versions** - System/Python/Rust version conflicts → Python subprocess bypasses
4. **GPU arch** - gfx1103 needs `HSA_OVERRIDE_GFX_VERSION=11.0.0`

## Installed Dependencies

```bash
# ROCm
/opt/rocm-7.0.2/

# Python (uv venv)
onnxruntime-rocm==1.22.2.post1  # With MIGraphX EP
torch==2.5.1+rocm6.2
pillow, numpy

# Rust features
backend-tract      # ✅ Working (default)
backend-ort-cpu    # ✅ Working
backend-ort-rocm   # ❌ Broken (needs custom build)
backend-torch      # ✅ Working (Python subprocess)
```

## Current Config

```toml
# doorman.toml
[ml]
backend = "torch"
device = "cuda"  # Actually ROCm/MIGraphX

[models]
detector_input_width = 128
detector_input_height = 128
liveness_input_width = 80
liveness_input_height = 80
recognizer_input_width = 112
recognizer_input_height = 112
recognizer_embedding_size = 128
```

## What's Left To Do

1. ✅ iGPU working
2. ⏭️ Implement actual BlazeFace decoder in Python (currently mock bbox)
3. ⏭️ Profile FPS limits (30/60/120)
4. ⏭️ Test enrollment end-to-end
5. ⏭️ Compare vs Nvidia 4060 Ti

## Project Structure After Cleanup

```
doorman/
├── BACKEND_STATUS.md         # This file
├── ARCHITECTURE.md           # Project architecture
├── CHANGELOG.md              # Release history
├── MODELS.md                 # ML models info
├── README.md                 # Main readme
├── doorman.toml              # Main config
├── QUICK_START.sh            # Quick start script
│
├── daemon/                   # Main daemon
├── shared/                   # Shared lib
├── cli/                      # CLI tool
├── pam_module/              # PAM integration
│
└── tools/
    ├── build/               # Build scripts
    ├── setup/               # Install/setup scripts
    ├── testing/             # Test scripts & data
    ├── configs/             # Test configs
    ├── benchmark_backends.sh
    ├── profile_fps.sh
    ├── run_torch.sh         # Run with iGPU
    └── torch_rocm_inference.py
```

## Quick Commands

```bash
# Build with iGPU support
cargo build --release --features backend-torch

# Run with iGPU
./tools/run_torch.sh --user --preview

# Run with CPU (default)
cargo build --release
./target/release/doormand --user --preview

# Test enrollment
doorman enroll  # (in another terminal)
```
