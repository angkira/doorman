# Current Status: iGPU Support (2025-12-03)

## ✅ ЧТО РАБОТАЕТ

### 1. GTT Memory - SUCCESS!
- ✅ GTT увеличен до **51.5GB** (было 32GB)
- ✅ VRAM: 1.9GB/2GB (desktop compositor)
- ✅ GTT: 1.4GB/51GB используется
- ✅ Нет OOM ошибок (HIP failure 719)

### 2. Model Compilation - SUCCESS!
Все 3 модели скомпилированы и в кэше:
- ✅ BlazeFace (240x320): 1.0s
- ✅ Liveness (96x96): 0.0s  
- ✅ MobileFaceNet (112x112): 2.3s

### 3. BlazeFace Decoder - FIXED!
- ✅ Input size: 240x320 (fixed from 128x128)
- ✅ Liveness size: 96x96 (fixed from 80x80)
- ✅ Letterboxing implemented
- ✅ Coordinate transformation working
- ✅ Real detections (no more mock data)

### 4. Daemon Startup - SUCCESS!
- ✅ Fast startup with cached models
- ✅ MIGraphX backend active
- ✅ No GPU errors

## ⚠️  PROBLEM: FPS Degradation

### Observed FPS:
```
Time     Camera  Detection
0:00     30 fps  7.8 fps   ✓
0:10     20 fps  4.7 fps   ↓
0:20     13 fps  2.9 fps   ↓↓
0:30     9.6 fps 2.1 fps   ↓↓↓
0:40     8.5 fps  ?        ↓↓↓↓
```

**FPS degrading over time instead of stabilizing!**

### Possible Causes:

1. **Python JSON-RPC Overhead**
   - IPC via stdin/stdout slow
   - JSON serialization overhead
   - Subprocess blocking

2. **BlazeFace Decoder Performance**
   - Letterboxing + transforms in Python
   - May be bottleneck

3. **MIGraphX Latency on iGPU**
   - Desktop compositor stealing GPU time
   - Shared memory bandwidth limits

4. **Pipeline Backpressure**
   - Queue filling up
   - No parallelism

## 📊 NEXT STEPS

### Immediate: Run Benchmark
```bash
./tools/benchmark_python_backend.sh
```

This will show:
- Pure Python FPS (no IPC overhead)
- IPC overhead percentage
- Per-model latency

### Options After Benchmark:

**Option A: Optimize Python Backend**
- Profile torch_inference.py
- Optimize coordinate transforms
- Add model batching

**Option B: Switch to Tract (Rust native)**
- No Python subprocess
- No IPC overhead
- BlazeFace decoder already exists (daemon/src/ml/blazeface_decoder.rs)
- BUT: CPU only (no ROCm in Tract)

**Option C: MIGraphX Tuning**
- Enable FP16 mode
- Tune provider options
- Optimize batch size

## 📁 KEY FILES

**Scripts:**
- `tools/run_torch.sh` - GPU backend launcher
- `tools/precompile_models.py` - Model compilation
- `tools/benchmark_python_backend.sh` - Performance test
- `tools/enable_gtt_memory.sh` - GTT config (done)

**Code:**
- `daemon/src/ml/torch_inference.py` - Python backend (BlazeFace decoder)
- `daemon/src/ml/torch_backend.rs` - Rust IPC wrapper
- `daemon/src/ml/tract_backend.rs` - Alternative backend (CPU)

**Configs:**
- `tools/configs/doorman-torch.toml` - GPU config
- `/etc/modprobe.d/ttm.conf` - GTT memory config

## 🎯 ACTION NEEDED

**Stop daemon and run benchmark:**
```bash
pkill doormand
./tools/benchmark_python_backend.sh
```

This will show where the bottleneck is and guide next steps.
