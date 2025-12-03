# Quick Start: Native Extension

## 1. Install maturin

```bash
pip install maturin
```

## 2. Build Native Extension

```bash
cd daemon/native_ml
./build.sh
```

**Первая сборка:** ~2-3 минуты (компиляция зависимостей)  
**Последующие:** ~30 секунд

## 3. Verify

```bash
python3 -c "from doorman_ml import DoormanML; print('✓ Native extension loaded')"
```

## 4. Run Benchmark

```bash
# Compare all 3 backends
cd ../..
python3 tools/benchmark.py -c tools/benchmark_configs/native_comparison.json
```

## Expected Output

```
=== Benchmark: PyTorch Direct (Baseline) ===
Backend: torch-direct
Warmup: 10 iterations...
Running: 50 iterations...
[##########] 50/50
✓ Mean FPS: 61.5

=== Benchmark: PyTorch Native (PyO3) ===
Backend: torch-native
Warmup: 10 iterations...
Running: 50 iterations...
[##########] 50/50
✓ Mean FPS: 58.3  ← Should be close to baseline!

=== Benchmark: PyTorch IPC (JSON+Base64) ===
Backend: torch-ipc
Warmup: 10 iterations...
Running: 50 iterations...
[##########] 50/50
✓ Mean FPS: 9.2  ← Much slower due to IPC overhead
```

## Troubleshooting

### "doorman_ml_native not installed"

```bash
cd daemon/native_ml
maturin develop --release
```

### "ort crate compilation error"

Убедитесь, что установлен ONNX Runtime:
```bash
# Check
python3 -c "import onnxruntime; print(onnxruntime.__version__)"

# Install if missing
pip install onnxruntime-gpu  # or onnxruntime
```

### "ROCm execution provider not found"

Native extension будет работать с CPU provider. Для GPU:
1. Убедитесь, что ONNX Runtime собран с ROCm support
2. Или используйте PyTorch backend (torch-direct/torch-ipc)

## Files Created

```
daemon/native_ml/
├── Cargo.toml              # Rust extension config
├── pyproject.toml          # Python package config
├── build.sh                # Build script
├── README.md               # Module docs
├── src/
│   ├── lib.rs              # PyO3 bindings
│   ├── detector.rs         # BlazeFace
│   ├── liveness.rs         # Liveness checker
│   └── embedder.rs         # MobileFaceNet
└── python/doorman_ml/
    └── __init__.py         # Python API

tools/benchmark_configs/
└── native_comparison.json  # Benchmark config

NATIVE_EXTENSION.md         # Full documentation
IPC_OVERHEAD_ANALYSIS.md    # Analysis (updated)
```

## Next Steps

After confirming Native Extension works:

1. Integrate into daemon: `daemon/src/ml/torch_backend_native.rs`
2. Update config: `ml.backend = "torch-native"`
3. Measure real-world performance with camera
4. Compare FPS: IPC (~7-10) vs Native (~55-60)
