# Native Backend Deployment Guide

## Quick Start

### 1. Build Native Extension

```bash
cd daemon/native_ml
./build.sh
# Or: maturin develop --release
```

**First build:** ~2-3 minutes (compiling dependencies)  
**Subsequent:** ~30 seconds

### 2. Build Daemon with Native Backend

```bash
cd /path/to/doorman
cargo build --release --features backend-torch-native,camera-gstreamer
```

### 3. Configure

```bash
# Copy native backend config
cp tools/configs/doorman-torch-native.toml doorman.toml

# Or edit existing config
[ml]
backend = "torch-native"  # ← Change this
device = "cuda"
models_dir = "~/.local/share/doorman/models"
```

### 4. Run

```bash
# Set ORT library path
export ORT_DYLIB_PATH=$(pwd)/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so

# Run daemon
./target/release/doormand --config doorman.toml
```

## Performance

### Benchmarks

| Metric | Native | Direct Python | IPC |
|--------|--------|---------------|-----|
| **FPS** | **169** | 62 | 8 |
| **Latency** | **5.9ms** | 16ms | 125ms |
| **vs Baseline** | **2.7x faster** | baseline | 7.8x slower |
| **vs IPC** | **21x faster** | 7.8x faster | baseline |

### Resource Usage

```
CPU:        ~15% (single core)
RAM:        ~400 MB
GPU:        ~30% (AMD Radeon 780M)
GPU Memory: ~2 GB VRAM
GPU Temp:   ~55°C
```

## Features

### ✅ Model Warmup

Native backend automatically:
1. Loads all models on startup (3-5 min first time)
2. Runs warmup iterations:
   - Detection: 3 iterations on 1024x720 dummy image
   - Liveness: 2 iterations on 112x112 crop
   - Embedding: 2 iterations on 112x112 crop
3. Reports readiness when fully warmed up

**Startup output:**
```
INFO Initializing Native PyTorch backend...
INFO Loading ML models (this may take 3-5 minutes on first run)...
INFO ✓ Models loaded successfully
INFO Warming up models...
INFO   Warmup iteration 1/3
INFO   Warmup iteration 2/3
INFO   Warmup iteration 3/3
INFO   Warmup liveness/embedding 1/2
INFO   Warmup liveness/embedding 2/2
INFO ✓ Warmup complete in 2.34s
INFO Native PyTorch backend ready for production use
```

### ✅ Model Caching

MIGraphX automatically caches compiled models:
- Location: `~/.cache/onnxruntime/migraphx/`
- First run: 3-5 minutes (compilation)
- Subsequent runs: Instant (loads from cache)

### ✅ Zero IPC Overhead

Direct Rust → Python FFI via PyO3:
- No subprocess spawning
- No JSON serialization
- No Base64 encoding
- No IPC communication

### ✅ Async Integration

All operations use `spawn_blocking` for non-blocking:
```rust
async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
    tokio::task::spawn_blocking({
        // ... native call ...
    }).await?
}
```

## Environment Variables

### Required

```bash
# Path to libonnxruntime.so (auto-detected in backend)
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
```

### Optional

```bash
# AMD GPU override (for Radeon 780M gfx1103)
export HSA_OVERRIDE_GFX_VERSION=11.0.1

# HIP device selection
export HIP_VISIBLE_DEVICES=0

# GPU queue limit
export GPU_MAX_HW_QUEUES=1

# ONNX Runtime log level
export ORT_LOG_LEVEL=3  # 0=verbose, 3=error only
```

## Troubleshooting

### "No module named 'doorman_ml_native'"

**Solution:**
```bash
cd daemon/native_ml
maturin develop --release
```

### "libonnxruntime.so: cannot open shared object file"

**Solution 1:** Let backend auto-detect:
```bash
# Backend checks these paths automatically:
# - .venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so
# - /usr/lib/libonnxruntime.so
# - /usr/local/lib/libonnxruntime.so
```

**Solution 2:** Set manually:
```bash
export ORT_DYLIB_PATH=$(python3 -c "import onnxruntime; import os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi/libonnxruntime.so'))")
```

### Models compiling on every run

**Check cache:**
```bash
ls ~/.cache/onnxruntime/migraphx/
```

**If missing:** MIGraphX can't write to cache. Check permissions:
```bash
mkdir -p ~/.cache/onnxruntime/migraphx
chmod 755 ~/.cache/onnxruntime/migraphx
```

### Low FPS (<100)

1. **Check GPU is used:**
   ```bash
   rocm-smi --showuse
   # Should show ~30% GPU usage during inference
   ```

2. **Check Python GIL:**
   - Native backend releases GIL during inference
   - Verify no other Python code holding GIL

3. **Check warmup completed:**
   - Look for "Warmup complete" in logs
   - First few frames may be slower

## Comparison with Other Backends

### vs IPC (torch-ipc)

**Native Advantages:**
- 21x faster FPS
- No subprocess management
- No IPC errors
- No JSON/Base64 overhead

**IPC Advantages:**
- Process isolation (crashes don't affect daemon)
- Easier to swap Python code

**Recommendation:** Use native for production

### vs Direct Python (torch-direct)

**Native Advantages:**
- 2.7x faster (no PyTorch overhead)
- Direct ONNX Runtime calls
- Lower memory usage

**Direct Advantages:**
- None - native is strictly better

**Recommendation:** Use native

### vs ORT Backend (ort-rocm)

**Native Advantages:**
- Pre-warmed models
- Python ecosystem compatibility
- Easier debugging

**ORT Advantages:**
- Pure Rust (no Python dependency)
- Slightly lower memory (~50MB less)

**Recommendation:** Use native unless pure Rust required

## Production Checklist

- [ ] Native extension built (`./build.sh`)
- [ ] Daemon built with `--features backend-torch-native`
- [ ] Config set to `backend = "torch-native"`
- [ ] ORT_DYLIB_PATH set (or auto-detected)
- [ ] Models cached in `~/.cache/onnxruntime/migraphx/`
- [ ] Warmup completes successfully on startup
- [ ] FPS >150 in production (check logs)
- [ ] GPU utilization ~30% (check with `rocm-smi`)
- [ ] No "Model Compile" messages after first run

## Next Steps

1. Test with real camera:
   ```bash
   ./target/release/doormand --config doorman-torch-native.toml
   ```

2. Enroll face:
   ```bash
   doorman enroll --username yourname
   ```

3. Test authentication:
   ```bash
   # In PAM-protected app (e.g., sudo)
   sudo echo "Testing doorman auth"
   ```

4. Monitor performance:
   ```bash
   # Check logs
   journalctl -u doormand -f
   
   # Check GPU
   watch -n 1 rocm-smi --showuse
   ```

## Support

If you encounter issues:

1. Check logs: `journalctl -u doormand -f`
2. Run benchmark: `python3 tools/benchmark.py -c tools/benchmark_configs/native_only.json`
3. Verify native extension: `python3 -c "import doorman_ml_native; print('OK')"`
4. Check GPU: `rocm-smi --showuse --showmeminfo vram`

**Expected performance:** ~169 FPS, <6ms latency, 21x faster than IPC
