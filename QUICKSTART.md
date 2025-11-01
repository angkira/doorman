# Quick Start

## 1. Install

```bash
sudo apt install build-essential libpam0g-dev pkg-config
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

cd doorman
uv pip install -e .
sudo doorman setup
```

## 2. Get Models

Download to `/var/lib/doorman/models/`:
- `blazeface.onnx`
- `liveness.onnx`
- `mobilefacenet.onnx`

See MODELS.md for links.

## 3. Enroll & Test

```bash
sudo doorman enroll
# Lock screen (Meta+L) to test
```

## GPU (Radeon 780M)

```bash
# Install ROCm
sudo apt install rocm-hip-libraries

# Configure
sudo mkdir -p /etc/doorman
sudo tee /etc/doorman/doorman.toml << EOF
[ml]
device = "rocm"
gpu_device_id = 0
EOF

# Rebuild with GPU
cargo build --release --features gpu
sudo systemctl restart doormand
```

See GPU_SETUP.md for full instructions.
