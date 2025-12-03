# Installation Guide

## Quick Start

```bash
git clone https://github.com/yourusername/doorman.git
cd doorman
./install.sh
```

The interactive installer will:
1. Detect your hardware (GPU, CPU)
2. Let you choose ML backend
3. Let you choose camera backend
4. Install to user or system directory
5. Create systemd service
6. Configure everything automatically

## Requirements

### System Dependencies

**Arch Linux:**
```bash
sudo pacman -S rust python python-pip pkg-config gstreamer gst-plugins-base
```

**Ubuntu/Debian:**
```bash
sudo apt install rustc cargo python3 python3-pip pkg-config libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

### For AMD iGPU (ROCm):
```bash
# Install ROCm (Arch Linux)
yay -S rocm-hip-sdk rocm-opencl-runtime

# Verify
rocm-smi
```

### For NVIDIA GPU:
```bash
# Install CUDA toolkit
# Follow: https://developer.nvidia.com/cuda-downloads
```

## Backend Selection

### 1. Tract (Pure Rust, CPU-only)
- **Performance:** ~15-20 FPS
- **Pros:** No external dependencies, simple setup
- **Cons:** CPU-only, slower
- **Use case:** Embedded systems, simple setups

### 2. PyTorch + ONNX Runtime (Recommended)
- **Performance:** 50-60 FPS (GPU) / 8-10 FPS (CPU)
- **Pros:** GPU acceleration, good performance
- **Cons:** Requires Python environment, IPC overhead
- **Use case:** Desktop systems with GPU

**Dependencies:**
- AMD: `onnxruntime-rocm` (with MIGraphX EP)
- NVIDIA: `onnxruntime-gpu`
- CPU: `onnxruntime`

### 3. Native PyO3 Extension (Experimental)
- **Performance:** 169 FPS (theoretical)
- **Pros:** Maximum performance, no IPC overhead
- **Cons:** Complex setup, requires manual library path configuration
- **Use case:** High-performance deployments

## Manual Installation

If you prefer manual installation:

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate

# For AMD iGPU
pip install onnxruntime-rocm

# For NVIDIA
pip install onnxruntime-gpu

# For CPU
pip install onnxruntime
```

### 2. Build Daemon

```bash
# Tract backend + GStreamer
cargo build --release --features backend-tract,camera-gstreamer

# PyTorch backend + GStreamer
cargo build --release --features backend-torch,camera-gstreamer

# PyTorch Native + GStreamer
cargo build --release --features backend-torch-native,camera-gstreamer
```

### 3. Download Models

```bash
mkdir -p ~/.local/share/doorman/models
cd ~/.local/share/doorman/models

# Download ONNX models
# TODO: Add download links
```

### 4. Create Config

Create `~/.config/doorman/doorman.toml`:

```toml
[ml]
backend = "torch"  # or "tract", "torch-native"
device = "cuda"    # or "cpu"
models_dir = "/home/yourusername/.local/share/doorman/models"
confidence_threshold = 0.7
liveness_threshold = 0.5
similarity_threshold = 0.6

[camera]
backend = "gstreamer"  # or "v4l"
device = "/dev/video0"
width = 1280
height = 720
fps = 30

[daemon]
socket_path = "/run/user/1000/doorman.sock"
data_dir = "/home/yourusername/.local/share/doorman"
user_mode = true

[storage]
embeddings_path = "/home/yourusername/.local/share/doorman/embeddings"

[logging]
level = "info"
```

### 5. Create Systemd Service

Create `~/.config/systemd/user/doormand.service`:

```ini
[Unit]
Description=Doorman Face Authentication Daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart=/home/yourusername/.local/bin/doormand --user --config /home/yourusername/.config/doorman/doorman.toml
Environment="VIRTUAL_ENV=/path/to/doorman/.venv"
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.1"  # For AMD gfx1103
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=default.target
```

### 6. Start Daemon

```bash
systemctl --user daemon-reload
systemctl --user start doormand
systemctl --user enable doormand  # Start on login
```

## Troubleshooting

### Low FPS on AMD iGPU

If you get ~8 FPS instead of 50-60 FPS:

1. **Check GPU usage:**
   ```bash
   rocm-smi --showmeminfo vram --showuse
   ```
   GPU usage should be >50%

2. **Check Python version:**
   ```bash
   which python3
   python3 --version
   ```
   Daemon should use venv Python (3.12), not system Python (3.13)

3. **Check ONNX Runtime providers:**
   ```bash
   source .venv/bin/activate
   python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
   ```
   Should include `MIGraphXExecutionProvider`

4. **Set environment variables:**
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=11.0.1  # For Radeon 780M (gfx1103)
   export HIP_VISIBLE_DEVICES=0
   export VIRTUAL_ENV=/path/to/doorman/.venv
   ```

### Models Not Found

```bash
# Check models directory
ls -lh ~/.local/share/doorman/models/

# Should contain:
# - blazeface.onnx
# - liveness.onnx
# - mobilefacenet.onnx
```

### Camera Not Detected

```bash
# List available cameras
v4l2-ctl --list-devices

# Test GStreamer
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink
```

### Permission Denied

```bash
# Add user to video group
sudo usermod -a -G video $USER

# Logout and login again
```

## Performance Tuning

### AMD iGPU (Radeon 780M)

```bash
# Set GPU to performance mode
sudo sh -c 'echo performance > /sys/class/drm/card0/device/power_dpm_force_performance_level'

# Set power limit (optional)
sudo sh -c 'echo 30000000 > /sys/class/drm/card0/device/hwmon/hwmon*/power1_cap'
```

### Precompile Models

```bash
# Precompile models to avoid startup delay
source .venv/bin/activate
python3 tools/precompile_models.py
```

This caches compiled models in `~/.cache/onnxruntime/migraphx/`

## Uninstallation

```bash
# Stop and disable service
systemctl --user stop doormand
systemctl --user disable doormand

# Remove files
rm ~/.local/bin/doormand
rm ~/.config/systemd/user/doormand.service
rm -rf ~/.local/share/doorman
rm -rf ~/.config/doorman

# Reload systemd
systemctl --user daemon-reload
```

## Development Setup

```bash
# Install uv for fast Python package management
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv with uv
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install onnxruntime-rocm

# Build in debug mode
cargo build --features backend-torch,camera-gstreamer

# Run tests
cargo test

# Run benchmark
python3 tools/benchmark.py --backend torch-direct --iterations 50
```

## License

MIT License - see [LICENSE](LICENSE)
