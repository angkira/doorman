# doorman

Fast face unlock for Linux. Replaces howdy with proper architecture.

**Status**: 🟡 **BBox fix applied, awaiting testing** → See [QUICK_TEST.md](QUICK_TEST.md)

**3 components**: PAM module (Rust) → Auth daemon (Rust) → CLI (Python)

**Key**: Daemon owns camera + models. PAM just sends IPC requests. No blocking, no crashes.

**NEW**: 4-stage non-blocking pipeline for high-performance face recognition.

## 🚀 Quick Start

```bash
# Test the latest bbox fix:
./test_bbox.sh                # Terminal 1: Start daemon
doorman preview --debug       # Terminal 2: View preview
```

See [MORNING_REPORT.md](MORNING_REPORT.md) for full status and [TODO.md](TODO.md) for priorities.

## Install

```bash
# Dependencies
sudo apt install build-essential libpam0g-dev pkg-config
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh  # Rust
curl -LsSf https://astral.sh/uv/install.sh | sh  # uv

# For high-performance camera (recommended)
./install_gstreamer.sh

# Build & install
cd doorman
uv pip install -e .
sudo doorman setup
```

## Models

Need 3 ONNX files in `/var/lib/doorman/models/`:
1. `blazeface.onnx` - Face detection
2. `liveness.onnx` - Anti-spoofing  
3. `mobilefacenet.onnx` - Recognition (512-d embeddings)

Download from:
- PINTO Model Zoo: https://github.com/PINTO0309/PINTO_model_zoo
- InsightFace: https://github.com/deepinsight/insightface/tree/master/model_zoo
- See MODELS.md for details

## Usage

```bash
# Daemon management (requires sudo)
sudo doorman start               # Start daemon
sudo doorman stop                # Stop daemon
sudo doorman restart             # Restart daemon
doorman status                   # Check status (no sudo needed)

# User management
doorman enroll                   # Enroll yourself
doorman test                     # Test authentication BEFORE enabling in PAM!
doorman preview                  # Live camera preview (OPTIONAL - see PREVIEW_BUILD.md)
doorman list                     # Show enrolled users
sudo doorman enroll <username>   # Enroll another user
sudo doorman remove <username>   # Remove user

# Model management
doorman models list              # Show model status
doorman models download          # Download missing models
doorman models verify            # Verify models
```

**⚠️ IMPORTANT:** Run `doorman test` to verify face recognition works BEFORE using it in PAM!

Lock screen (Meta+L) to test face unlock in production.

## GPU Acceleration (Radeon 780M)

```toml
# /etc/doorman/doorman.toml
[ml]
device = "rocm"  # or "cuda" for NVIDIA
gpu_device_id = 0

[authentication]
auth_frames = 7  # Fewer frames needed with GPU
```

Install ROCm, rebuild with `cargo build --release --features gpu`. See GPU_SETUP.md.

## Performance Optimization

### Shared Memory IPC (40-50 FPS)

For best performance with PyTorch backend, use shared memory to eliminate IPC overhead:

```bash
# Build with shared memory optimization
cargo build --release --features backend-torch-shm,camera-gstreamer

# Test performance
./test-torch-shm-daemon.sh

# Benchmark comparison
python3 tools/benchmark.py -c tools/benchmark_configs/ipc_optimization_comparison.json
```

**Performance improvement:**
- torch-ipc (base): ~7 FPS (JSON+Base64 overhead)
- torch-shm (optimized): ~45 FPS (zero-copy frames)

See [SHARED_MEMORY_QUICKSTART.md](SHARED_MEMORY_QUICKSTART.md) for details.

## Config

Edit `/etc/doorman/doorman.toml`:

```toml
[authentication]
similarity_threshold = 0.65  # 0.55-0.75 (lower = more lenient)
auth_frames = 10

[ml]
device = "cpu"  # or "rocm", "cuda"
```

Restart: `sudo systemctl restart doormand`

## Troubleshooting

```bash
sudo journalctl -u doormand -f    # Check logs
sudo doorman status               # Check daemon health
grep doorman /etc/pam.d/kde      # Verify PAM config
```

**Face not recognized**: Re-enroll with better lighting or lower threshold in config.  
**Camera busy**: Close other apps (Zoom, Cheese, etc.)  
**No models**: Download .onnx files to `/var/lib/doorman/models/`

## Security

**Good for**: Personal workstation convenience  
**Not for**: High-security servers, shared machines

Password fallback always available. Embeddings are root-only (0600).

## Testing

```bash
make test                        # Unit tests
make test-video                  # With video support
cargo test --test e2e_test      # Integration tests
pytest src/doorman/test_cli.py  # Python tests
```

See TESTING.md for details.

## Camera Backends

Doorman supports multiple camera backends with automatic fallback:

| Backend | Speed | Integration | When to Use |
|---------|-------|-------------|-------------|
| **GStreamer** | ⚡ 20-30 fps | PipeWire | **Production (recommended)** |
| FFmpeg | 🐌 1-10 fps | V4L2 | Fallback/compatibility |

**Use GStreamer for production:**
```bash
./install_gstreamer.sh
cargo build --release --features camera-gstreamer
```

See [GSTREAMER_BACKEND.md](GSTREAMER_BACKEND.md) for details.

## Architecture

The daemon uses a 4-stage non-blocking pipeline:

```
Camera Producer (20-30fps) → Frame Fanout → Detection (5fps) → Recognition
                                 ↓
                            Preview (15fps)
```

**Benefits**: No blocking, parallel processing, consistent frame rates, graceful degradation.

**Documentation**:
- [GSTREAMER_BACKEND.md](GSTREAMER_BACKEND.md) - High-performance camera setup
- [PIPELINE_QUICK_START.md](PIPELINE_QUICK_START.md) - Quick reference
- [ARCHITECTURE.md](ARCHITECTURE.md) - Full design specification
- [PIPELINE_IMPLEMENTATION.md](PIPELINE_IMPLEMENTATION.md) - Implementation details

## License

MIT

