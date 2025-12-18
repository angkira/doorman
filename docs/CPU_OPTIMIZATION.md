# CPU Performance Optimization

## Overview

Doorman's Tract backend (pure Rust, CPU-only) has been optimized for maximum performance on CPU hardware.

## Optimizations Implemented

### 1. Compilation Optimizations

**RUSTFLAGS for CPU:**
```bash
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
cargo build --release --features backend-tract
```

Enables:
- **AVX2/AVX512** SIMD instructions
- **Native CPU optimizations** for your specific processor
- **Maximum optimization level**

Expected improvement: **20-30% faster inference**

### 2. Model Loading Optimizations

```rust
let model = tract_onnx::onnx()
    .model_for_path(path)?
    .into_optimized()?  // Tract applies graph optimizations
    .into_runnable()?;
```

Tract automatically:
- Fuses operations (Conv + BatchNorm + ReLU → single op)
- Eliminates redundant ops
- Optimizes memory layout

### 3. Image Processing Optimizations

**Faster Resize Filter:**
```rust
// Before: Lanczos3 (high quality, slow)
// After: Triangle (bilinear, fast)
let resized = image.resize(width, height, FilterType::Triangle);
```

Trade-off: Slightly lower quality for **2-3x faster** resizing

**Cache-Friendly Tensor Conversion:**
- Pre-allocate Vec with exact capacity
- Sequential memory access pattern (HWC → CHW)
- Better CPU cache utilization

### 4. Parallel Processing

Dependencies added:
```toml
rayon = "1.10"  # Data parallelism
ndarray = { version = "0.16", features = ["rayon"] }
```

Future optimizations:
- Parallel batch processing
- Multi-threaded preprocessing
- Concurrent model inference

## Performance Targets

### Current Performance (CPU)
- **Detection:** ~100ms/frame (10 FPS)
- **Recognition:** ~50ms/face
- **Total latency:** ~150ms per authenticated frame

### Optimized Performance (CPU with SIMD)
- **Detection:** ~60-80ms/frame (12-16 FPS)
- **Recognition:** ~30-40ms/face
- **Total latency:** ~100ms per authenticated frame

### Comparison with GPU
- **CPU (optimized):** 12-16 FPS
- **GPU (CUDA):** 50-60 FPS
- **Speedup:** ~4-5x with GPU

## CPU Selection Priority

Tract backend automatically uses:
1. **x86_64 with AVX512** - Best performance
2. **x86_64 with AVX2** - Good performance
3. **x86_64 with SSE4.2** - Baseline
4. **ARM with NEON** - Mobile/embedded

Check your CPU features:
```bash
# Linux
cat /proc/cpuinfo | grep flags
# Look for: avx2, avx512f, fma

# Or use lscpu
lscpu | grep Flags
```

## Build Commands

### Standard Build (CPU)
```bash
cargo build --release --features backend-tract,camera-gstreamer
```

### Optimized Build (CPU with native optimizations)
```bash
RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
cargo build --release --features backend-tract,camera-gstreamer
```

### Profile-Guided Optimization (PGO) - Advanced
```bash
# Step 1: Build instrumented binary
RUSTFLAGS="-C profile-generate=/tmp/pgo-data" \
cargo build --release --features backend-tract,camera-gstreamer

# Step 2: Run workload to collect profiles
./target/release/doormand --config doorman.toml
# Let it run for a few minutes, then Ctrl+C

# Step 3: Merge profiles
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data/*.profraw

# Step 4: Build optimized binary
RUSTFLAGS="-C profile-use=/tmp/pgo-data/merged.profdata" \
cargo build --release --features backend-tract,camera-gstreamer
```

Expected PGO improvement: **5-15% additional speedup**

## Runtime Optimizations

### Thread Pool Configuration

Set number of threads for Tract:
```bash
export OMP_NUM_THREADS=4  # Use 4 CPU cores
export RAYON_NUM_THREADS=4
```

### Memory Allocator

Use jemalloc for better memory performance:
```toml
# In daemon/Cargo.toml
[dependencies]
jemallocator = "0.5"
```

```rust
// In main.rs
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;
```

Expected improvement: **5-10% faster, lower memory fragmentation**

## Benchmarking

Test CPU performance:
```bash
# Build optimized
RUSTFLAGS="-C target-cpu=native" cargo build --release --features backend-tract

# Run benchmark
cargo run --release --bin test-pipeline-debug -- \
    --config doorman.toml \
    --iterations 100
```

Compare before/after optimizations.

## Future Optimizations

1. **Quantization:** INT8 models for 2-4x speedup
2. **Model pruning:** Reduce model size by 30-50%
3. **Custom operators:** Hand-tuned SIMD kernels
4. **Multi-model batching:** Process multiple frames in parallel
5. **Async inference:** Overlap I/O and compute

## When to Use CPU vs GPU

**Use CPU (Tract) when:**
- No GPU available
- Power efficiency is critical
- Latency < 200ms is acceptable
- Cost-sensitive deployment

**Use GPU (CUDA) when:**
- High throughput required (>20 FPS)
- Low latency critical (<50ms)
- GPU already present in system
- Batch processing workloads

## Monitoring

Check CPU usage during inference:
```bash
# Monitor CPU usage
htop

# Profile specific process
perf record -g ./target/release/doormand
perf report
```

Look for:
- High CPU usage (good - means we're using available cores)
- Cache misses (optimize data layout if high)
- Branch mispredictions (optimize control flow)
