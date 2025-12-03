# 🎯 ML Backends Overview

Doorman supports multiple ML inference backends, each optimized for different use cases.

## Available Backends

### 1. **Socket (Unix Domain Socket)** ⭐ RECOMMENDED
- **Feature**: `backend-socket`
- **Config**: `backend = "socket"`
- **Performance**: **50-55 FPS** on AMD Radeon 780M iGPU
- **Latency**: ~10-20μs communication overhead
- **Use case**: Production deployment with Docker

**Pros:**
✅ Zero-copy binary protocol  
✅ No HTTP/JSON/Base64 overhead  
✅ Ultra-low latency (~10-20μs)  
✅ Works through Docker  
✅ Stable and reliable  

**Cons:**
❌ Requires Python server running  
❌ Need Docker setup  

**Setup:**
```bash
# Build container
cd docker && docker compose build && docker compose up -d

# Build daemon
cargo build --release --features backend-socket,camera-gstreamer

# Run
./target/release/doormand --user --config doorman-socket.toml
```

---

### 2. **Tract (Pure Rust)** 📦 DEFAULT
- **Feature**: `backend-tract`  
- **Config**: `backend = "tract"`
- **Performance**: **8-10 FPS** on CPU, **15-20 FPS** on GPU
- **Use case**: Simple deployment, no dependencies

**Pros:**
✅ No external dependencies  
✅ Pure Rust (safe)  
✅ Cross-platform  
✅ Easy to build  

**Cons:**
❌ Slower than GPU-accelerated backends  
❌ Limited GPU support  

**Setup:**
```bash
cargo build --release --features backend-tract,camera-gstreamer
```

---

### 3. **ONNX Runtime (CPU)** 🖥️
- **Feature**: `backend-ort-cpu`
- **Config**: `backend = "onnx"`
- **Performance**: **15-20 FPS** on CPU
- **Use case**: Server deployment without GPU

**Pros:**
✅ Faster than Tract on CPU  
✅ Mature and stable  
✅ Good CPU optimizations  

**Cons:**
❌ No GPU acceleration  
❌ External dependency (auto-downloaded)  

**Setup:**
```bash
cargo build --release --features backend-ort-cpu,camera-gstreamer
```

---

### 4. **ONNX Runtime (ROCm)** 🚀 FAST
- **Feature**: `backend-ort-rocm`
- **Config**: `backend = "onnx", device = "cuda"`
- **Performance**: **40-50 FPS** on AMD Radeon 780M iGPU
- **Use case**: Native deployment with iGPU

**Pros:**
✅ Direct GPU access  
✅ No IPC overhead  
✅ Fast inference  

**Cons:**
❌ Requires ROCm installation  
❌ Complex setup  
❌ AMD-specific  

**Setup:**
```bash
# Install ROCm first
# Then build
cargo build --release --features backend-ort-rocm,camera-gstreamer
```

---

### 5. **PyTorch (IPC)** 🐍
- **Feature**: `backend-torch`
- **Config**: `backend = "torch"`
- **Performance**: **7-10 FPS** (IPC bottleneck)
- **Use case**: Development/testing only

**Pros:**
✅ Python ecosystem  
✅ Easy model conversion  
✅ Good for prototyping  

**Cons:**
❌ Slow (JSON + Base64 overhead)  
❌ IPC communication bottleneck  
❌ Not recommended for production  

---

### 6. **PyTorch Native (PyO3)** ⚡ EXPERIMENTAL
- **Feature**: `backend-torch-native`
- **Config**: `backend = "torch-native"`
- **Performance**: **55-60 FPS** (theoretical)
- **Use case**: Maximum performance

**Pros:**
✅ No IPC overhead  
✅ Direct Python calls  
✅ Maximum performance  

**Cons:**
❌ Complex build (PyO3 + venv)  
❌ Python version dependencies  
❌ Experimental  

---

### 7. **Docker (HTTP)** 🐳
- **Feature**: `backend-docker`
- **Config**: `backend = "docker"`
- **Performance**: **10-15 FPS** (HTTP overhead)
- **Use case**: Legacy, use Socket instead

**Pros:**
✅ Isolated environment  
✅ Easy deployment  

**Cons:**
❌ Slow (HTTP + JSON + Base64)  
❌ 5-7ms overhead per frame  
❌ Superseded by Socket backend  

---

## Performance Comparison

| Backend | FPS | Latency | GPU | Setup | Production |
|---------|-----|---------|-----|-------|------------|
| **Socket** | **50-55** | **~10μs** | ✅ | Medium | ✅ **BEST** |
| Tract | 8-10 | ~100ms | ❌ | Easy | ✅ Simple |
| ORT-CPU | 15-20 | ~50ms | ❌ | Easy | ✅ Server |
| ORT-ROCm | 40-50 | ~20ms | ✅ | Hard | ✅ Native |
| PyTorch IPC | 7-10 | ~140ms | ✅ | Easy | ❌ Dev only |
| PyTorch Native | 55-60 | ~1μs | ✅ | Hard | ⚠️ Experimental |
| Docker HTTP | 10-15 | ~5ms | ✅ | Medium | ❌ Legacy |

## Recommendations

### Development
- **Tract** - Easy setup, no dependencies
- **PyTorch IPC** - Python prototyping

### Production
- **Socket** - Best balance (performance + stability)
- **ORT-ROCm** - Maximum native performance (if ROCm works)

### Server (no GPU)
- **ORT-CPU** - Best CPU performance
- **Tract** - Simplest deployment

## Build Commands

```bash
# Socket (recommended)
cargo build --release --features backend-socket,camera-gstreamer

# Tract (simple)
cargo build --release --features backend-tract,camera-gstreamer

# ONNX Runtime CPU
cargo build --release --features backend-ort-cpu,camera-gstreamer

# ONNX Runtime ROCm
cargo build --release --features backend-ort-rocm,camera-gstreamer
```

## Configuration

All backends use the same config format:

```toml
[ml]
models_dir = "~/.local/share/doorman/models"
backend = "socket"  # or "tract", "onnx", "torch", etc.
device = "cuda"     # or "cpu"
socket_path = "/tmp/doorman-ml.sock"  # for socket backend
```

## See Also

- [SOCKET_BACKEND.md](./SOCKET_BACKEND.md) - Socket backend details
- [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md) - Docker setup guide
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Overall architecture
