# Docker-based ONNX Runtime with ROCm

## Quick Start

```bash
# Build
cd docker && docker-compose build

# Run
docker-compose up -d

# Test
curl http://localhost:5000/health
```

## Expected: 50-60 FPS on iGPU! 🚀
