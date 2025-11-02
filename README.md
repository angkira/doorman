# doorman

Fast face unlock for Linux. Replaces howdy with proper architecture.

**3 components**: PAM module (Rust) → Auth daemon (Rust) → CLI (Python)

**Key**: Daemon owns camera + models. PAM just sends IPC requests. No blocking, no crashes.

## Install

```bash
# Dependencies
sudo apt install build-essential libpam0g-dev pkg-config
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh  # Rust
curl -LsSf https://astral.sh/uv/install.sh | sh  # uv

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
doorman list                     # Show enrolled users
sudo doorman enroll <username>   # Enroll another user
sudo doorman remove <username>   # Remove user

# Model management
doorman models list              # Show model status
doorman models download          # Download missing models
doorman models verify            # Verify models
```

Lock screen (Meta+L) to test face unlock.

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

## License

MIT

