# Doorman ML Native Extension

Native Python extension for high-performance face recognition inference.

## Features

- **Zero IPC overhead** - Direct Rust → Python calls via PyO3
- **ONNX Runtime** - Uses same models as daemon (blazeface, liveness, mobilefacenet)
- **GPU support** - ROCm/CUDA execution providers
- **~55-60 FPS** - Near-native performance (vs ~7-10 FPS with IPC)

## Architecture

```
Python (benchmark.py)
    ↓ (PyO3 - no overhead)
Rust (doorman_ml_native.so)
    ↓ (direct calls)
ONNX Runtime
    ↓ (MIGraphX/ROCm)
AMD iGPU
```

## Build

```bash
# Install maturin
pip install maturin

# Build in development mode (creates .so in venv)
maturin develop --release

# Or build wheel
maturin build --release
pip install target/wheels/*.whl
```

## Usage

```python
from doorman_ml import DoormanML
import numpy as np

# Initialize
ml = DoormanML(models_dir="./models", device="cuda")

# Detect faces
image_rgb = ...  # RGB bytes (H * W * 3)
detections = ml.detect_faces(image_rgb, width=1024, height=720)

for det in detections:
    print(f"Face at {det.bbox}, confidence={det.confidence:.3f}")

# Check liveness
face_crop = ...  # 112x112x3 RGB bytes
liveness = ml.check_liveness(face_crop)
print(f"Is live: {liveness.is_live}, confidence={liveness.confidence:.3f}")

# Extract embedding
embedding_bytes = ml.extract_embedding(face_crop)
embedding = np.frombuffer(embedding_bytes, dtype=np.float32)  # 512-dim vector
```

## Benchmark Integration

```python
# tools/benchmark.py
from doorman_ml import DoormanML

class TorchNativeBackend:
    def __init__(self, models_dir, device="cuda"):
        self.ml = DoormanML(models_dir, device)
    
    def detect_faces(self, image_data, width, height):
        return self.ml.detect_faces(image_data, width, height)
```

## Performance

| Backend | FPS | Overhead | Notes |
|---------|-----|----------|-------|
| torch-direct (Python) | ~60 | 0ms | Baseline |
| **torch-native (PyO3)** | **~55-60** | **<1ms** | **This module** |
| torch-ipc-shmem | ~40-50 | 5-10ms | Shared memory |
| torch-ipc | ~7-10 | 50-80ms | JSON+Base64 |
