# 🎮 Command Reference

Quick reference for common doorman commands.

---

## 🧪 Testing

### Test BBox Fix (Start Here!)
```bash
# Terminal 1: Start daemon with preview
./test_bbox.sh

# Terminal 2: Run preview client
doorman preview --debug
```

### Debug Output
```bash
# Full debug logs
RUST_LOG=debug ./target/release/doormand --user --preview

# Watch bbox values only
./target/release/doormand --user --preview 2>&1 | grep "bbox="

# Watch detection events
./target/release/doormand --user --preview 2>&1 | grep -E "(Detection|Face)"
```

---

## 🔨 Building

### Build Daemon
```bash
# Release build (optimized)
cargo build --release

# Debug build (with symbols)
cargo build

# Build specific binary
cargo build --release --bin doormand
```

### Build Python CLI
```bash
# Install in development mode
uv pip install -e .

# Reinstall after changes
uv pip install --force-reinstall -e .
```

---

## 🎯 Running

### Daemon
```bash
# User mode with preview
./target/release/doormand --user --preview

# System mode (requires root)
sudo ./target/release/doormand --system

# With custom config
./target/release/doormand --user --config custom.toml
```

### Preview Client
```bash
# Normal preview
doorman preview

# With debug info
doorman preview --debug

# Save screenshots
doorman preview --screenshot-dir ~/doorman-shots
```

---

## 👤 User Management (TODO)

```bash
# Enroll new user
doorman enroll <username>

# List enrolled users
doorman list

# Delete user
doorman delete <username>

# Test recognition
doorman test
```

---

## 🔍 Diagnostics

### Check Status
```bash
# Is daemon running?
ps aux | grep doormand

# Check sockets
ls -la /run/user/1000/doorman*.sock

# Test IPC connection
doorman status
```

### Camera Backends
```bash
# List available cameras
v4l2-ctl --list-devices

# Test FFmpeg capture
ffmpeg -f v4l2 -i /dev/video0 -vframes 1 test.jpg

# Check GStreamer
gst-inspect-1.0 pipewiresrc
```

### Models
```bash
# Check models directory
ls -lh ~/.local/share/doorman/models/

# Verify ONNX files
file ~/.local/share/doorman/models/*.onnx
```

---

## 🧹 Cleanup

### Stop Everything
```bash
# Kill daemon
pkill doormand

# Remove sockets
rm /run/user/1000/doorman*.sock

# Clear logs (if using systemd)
journalctl --user -u doormand --vacuum-time=1d
```

### Reset Data
```bash
# Remove all user data
rm -rf ~/.local/share/doorman/users/

# Reset configuration
rm ~/.config/doorman/config.toml

# Full reset
rm -rf ~/.local/share/doorman/ ~/.config/doorman/
```

---

## 🐛 Debugging

### Common Issues

**"No faces detected"**
```bash
# Lower detection threshold
# Edit daemon/src/ml/tract_backend.rs
# Change confidence_threshold from 0.4 to 0.3
cargo build --release
```

**"Camera failed"**
```bash
# Check permissions
groups | grep video

# Test camera directly
ffmpeg -f v4l2 -i /dev/video0 -vframes 1 test.jpg

# Try different device
./target/release/doormand --user --config <(echo 'camera.device_index = 1')
```

**"Preview not working"**
```bash
# Check socket exists
ls -la /run/user/1000/doorman-frames.sock

# Test with debug
doorman preview --debug

# Check Python deps
uv pip list | grep -E "(opencv|numpy)"
```

---

## 📊 Performance

### Benchmarking
```bash
# Monitor FPS
./target/release/doormand --user --preview 2>&1 | grep "fps"

# CPU usage
top -p $(pgrep doormand)

# Memory usage
ps aux | grep doormand | awk '{print $6/1024 " MB"}'
```

### Profiling
```bash
# Build with profiling
cargo build --release --features=profiling

# Run with perf
perf record ./target/release/doormand --user
perf report
```

---

## 🧪 Testing

### Unit Tests
```bash
# All tests
cargo test

# Specific test
cargo test test_coordinate_transform

# With output
cargo test -- --nocapture

# Release mode (faster)
cargo test --release
```

### Integration Tests
```bash
# TODO: Not yet implemented
cargo test --test integration
```

---

## 📦 Installation

### System-wide
```bash
# Build release
cargo build --release

# Install daemon
sudo cp target/release/doormand /usr/local/bin/

# Install CLI
uv pip install .

# Setup models
sudo mkdir -p /var/lib/doorman/models
sudo cp models/*.onnx /var/lib/doorman/models/
```

### Systemd Service
```bash
# User service
cp doormand-user.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable doormand
systemctl --user start doormand

# System service (requires root)
sudo cp doormand.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable doormand
sudo systemctl start doormand
```

---

## 📝 Logs

### View Logs
```bash
# User service
journalctl --user -u doormand -f

# System service
sudo journalctl -u doormand -f

# Direct run (to file)
./target/release/doormand --user --preview 2>&1 | tee doorman.log
```

### Log Levels
```bash
# Error only
RUST_LOG=error ./target/release/doormand

# Info (default)
RUST_LOG=info ./target/release/doormand

# Debug (verbose)
RUST_LOG=debug ./target/release/doormand

# Trace (very verbose)
RUST_LOG=trace ./target/release/doormand

# Module-specific
RUST_LOG=doormand::ml=debug ./target/release/doormand
```

---

## 🔐 Security

### Check Permissions
```bash
# Socket permissions
ls -la /run/user/1000/doorman*.sock

# Data directory
ls -la ~/.local/share/doorman/

# Models directory
ls -la ~/.local/share/doorman/models/
```

### Audit
```bash
# TODO: Not yet implemented
doorman audit
```

---

**Quick Links**:
- [README.md](README.md) - Project overview
- [MORNING_REPORT.md](MORNING_REPORT.md) - Current status
- [TODO.md](TODO.md) - Task priorities
- [QUICK_TEST.md](QUICK_TEST.md) - Test instructions
