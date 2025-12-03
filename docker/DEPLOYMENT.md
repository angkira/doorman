# Docker Deployment Guide

Complete guide for deploying Doorman with GPU-accelerated ML inference in Docker.

## Quick Start

```bash
# One-command deployment
./deploy-docker.sh
```

This will:
1. Build Docker inference container
2. Download/verify ONNX models
3. Build Rust daemon with Docker backend
4. Install binaries and configs
5. Create systemd services
6. Start everything

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Host System                                    │
│                                                 │
│  ┌──────────────────┐    HTTP API              │
│  │  Rust Daemon     │◄──────────────┐          │
│  │  (doormand)      │               │          │
│  │                  │               │          │
│  │  • Camera        │               │          │
│  │  • IPC Server    │               │          │
│  │  • Storage       │               │          │
│  └──────────────────┘               │          │
│                                     │          │
│  ┌──────────────────────────────────┴────────┐ │
│  │  Docker Container                         │ │
│  │  ┌────────────────────────────────────┐   │ │
│  │  │  ONNX Runtime + ROCm               │   │ │
│  │  │  Python Flask Server               │   │ │
│  │  │                                    │   │ │
│  │  │  • BlazeFace (detection)           │   │ │
│  │  │  • MiniFASNet (liveness)           │   │ │
│  │  │  • MobileFaceNet (recognition)     │   │ │
│  │  │                                    │   │ │
│  │  │  Port: 5000                        │   │ │
│  │  │  GPU: AMD Radeon 780M (ROCm)       │   │ │
│  │  └────────────────────────────────────┘   │ │
│  └───────────────────────────────────────────┘ │
│           ▲                                     │
│           │ GPU Passthrough                    │
│           │ /dev/kfd, /dev/dri                 │
└───────────┴─────────────────────────────────────┘
```

## Manual Setup

### 1. Build Container

```bash
cd docker
docker compose build
```

### 2. Start Container

```bash
docker compose up -d
```

### 3. Verify Container

```bash
# Check health
curl http://localhost:5000/health

# Expected output:
{
  "status": "healthy",
  "providers": ["ROCMExecutionProvider", "CPUExecutionProvider"],
  "models_loaded": true
}
```

### 4. Build Daemon

```bash
cargo build --release --features backend-docker,camera-gstreamer
```

### 5. Configure

Edit `doorman-docker.toml`:

```toml
[ml]
backend = "docker"
docker_endpoint = "http://localhost:5000"
```

### 6. Run Daemon

```bash
./target/release/doormand --user --config doorman-docker.toml
```

## Systemd Integration

### Container Service

```bash
systemctl --user enable doorman-inference.service
systemctl --user start doorman-inference.service
```

### Daemon Service

```bash
systemctl --user enable doormand.service
systemctl --user start doormand.service
```

### Check Status

```bash
# Container
systemctl --user status doorman-inference
docker ps | grep doorman

# Daemon  
systemctl --user status doormand
journalctl --user -u doormand -f
```

## Performance

### Expected FPS

| Component | FPS | Latency |
|-----------|-----|---------|
| Detection | 60+ | 10ms |
| Liveness | 60+ | 8ms |
| Recognition | 55+ | 9ms |
| **Pipeline** | **50-55** | **~18ms** |

### Monitoring

```bash
# Container logs
docker compose logs -f

# GPU usage
watch -n 1 rocm-smi

# HTTP requests
curl http://localhost:5000/health
```

## Troubleshooting

### Container won't start

```bash
# Check Docker
docker ps -a

# Check logs
docker compose logs

# Rebuild
docker compose down
docker compose build --no-cache
docker compose up -d
```

### GPU not detected

```bash
# Verify ROCm
rocminfo

# Check devices
ls -la /dev/kfd /dev/dri

# Check permissions
groups | grep video

# Add to video group
sudo usermod -aG video $USER
```

### Daemon can't connect

```bash
# Check endpoint
curl http://localhost:5000/health

# Check container
docker ps | grep doorman

# Check firewall
sudo ufw status
```

### Low performance

```bash
# Check GPU usage
rocm-smi

# Check container resources
docker stats doorman-onnx-rocm

# Check HSA override
docker exec doorman-onnx-rocm env | grep HSA
```

## Configuration

### Container Environment

Edit `docker/docker-compose.yml`:

```yaml
environment:
  - HSA_OVERRIDE_GFX_VERSION=11.0.1  # For Radeon 780M
  - HIP_VISIBLE_DEVICES=0
  - MODELS_DIR=/app/models
```

### Daemon Config

Edit `doorman-docker.toml`:

```toml
[ml]
backend = "docker"
docker_endpoint = "http://localhost:5000"  # Container endpoint
device = "rocm"  # Just for logging, runs in container
```

## Updating

### Update Container

```bash
cd docker
git pull
docker compose build
docker compose up -d
```

### Update Daemon

```bash
git pull
cargo build --release --features backend-docker,camera-gstreamer
systemctl --user restart doormand
```

## Uninstall

```bash
# Stop services
systemctl --user stop doormand doorman-inference
systemctl --user disable doormand doorman-inference

# Remove container
docker compose down
docker rmi doorman-onnx-rocm

# Remove files
rm -rf ~/.local/doorman
rm -rf ~/.local/share/doorman
rm ~/.config/systemd/user/doorman*.service
```

## Benefits

✅ **Isolated Environment** - No system pollution  
✅ **Pre-built Binaries** - No ONNX Runtime compilation  
✅ **GPU Acceleration** - Direct ROCm passthrough  
✅ **Easy Updates** - Pull and restart  
✅ **Reproducible** - Same setup everywhere  
✅ **Production Ready** - Battle-tested stack  

## Performance Tips

1. **Warm Models on Start** - Container pre-loads and warms up models
2. **HTTP Keep-Alive** - Daemon reuses connections
3. **Async Processing** - Non-blocking inference calls
4. **Pipeline Throttling** - 10 FPS detection, 2 FPS recognition
5. **GPU Memory** - Container manages VRAM efficiently

## Next Steps

- [ ] Add Unix socket support for lower latency
- [ ] Implement shared memory for zero-copy frames
- [ ] Add Prometheus metrics endpoint
- [ ] Multi-GPU support
- [ ] Batch inference support

Docker solution provides best balance of performance and maintainability! 🐳
