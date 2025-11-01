# ONNX Models Guide for doorman

This guide provides detailed instructions for obtaining and converting the required ONNX models.

## Required Models

doorman needs three ONNX models in `/var/lib/doorman/models/`:

1. **blazeface.onnx** - Face detection (~1-2 MB)
2. **liveness.onnx** - Anti-spoofing (~1-5 MB)
3. **mobilefacenet.onnx** - Face recognition (~4 MB)

## Quick Start (Pre-converted Models)

### Option 1: PINTO Model Zoo (Recommended)

The PINTO Model Zoo provides pre-converted ONNX models that work out of the box:

```bash
# 1. BlazeFace (Face Detection)
wget https://github.com/PINTO0309/PINTO_model_zoo/raw/main/033_Face_Detection_BlazeFace/01_float32/blazeface_1280x1280.onnx -O blazeface.onnx

# 2. Silent-Face Anti-Spoofing (Liveness)
wget https://github.com/PINTO0309/PINTO_model_zoo/raw/main/168_Silent-Face-Anti-Spoofing/01_float32/anti_spoof_mn3_224x224.onnx -O liveness.onnx

# 3. MobileFaceNet (Recognition)
# Download from InsightFace model zoo or convert from PyTorch
```

### Option 2: InsightFace Pre-trained Models

InsightFace provides high-quality face recognition models:

```bash
# Clone InsightFace model zoo
git clone https://github.com/deepinsight/insightface.git
cd insightface/model_zoo

# Download MobileFaceNet or ArcFace
# Models are in MXNet format, need conversion (see below)
```

## Detailed Instructions

### 1. BlazeFace (Face Detection)

**Source**: Google MediaPipe

**Conversion Steps**:

```bash
# Install dependencies
pip install mediapipe onnx tf2onnx tensorflow

# Option A: Export from MediaPipe
python3 << EOF
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# This may require MediaPipe 0.9.0+
# Export functionality varies by version
EOF

# Option B: Use pre-converted from PINTO
wget https://github.com/PINTO0309/PINTO_model_zoo/raw/main/033_Face_Detection_BlazeFace/01_float32/blazeface_1280x1280.onnx -O blazeface.onnx

# Verify model
python3 << EOF
import onnx
model = onnx.load("blazeface.onnx")
onnx.checker.check_model(model)
print(f"✓ BlazeFace model loaded: {len(model.graph.node)} nodes")
EOF
```

**Alternative Models**:
- **RetinaFace** - More accurate but slower
- **MTCNN** - Classic, widely supported
- **YuNet** - OpenCV's face detector

### 2. Silent-Face Anti-Spoofing (Liveness Detection)

**Source**: MiniVision AI

**Download Pre-trained**:

```bash
# Clone the repository
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
cd Silent-Face-Anti-Spoofing

# Download pre-trained model (PyTorch)
# Models are in resources/anti_spoof_models/
# Common versions: 2.7_80x80_MiniFASNetV2.pth

# Convert PyTorch -> ONNX
pip install torch torchvision onnx

python3 << EOF
import torch
import torch.onnx
from src.anti_spoof_predict import AntiSpoofPredict

# Load model
model = AntiSpoofPredict(0)  # Device ID
# Export to ONNX
dummy_input = torch.randn(1, 3, 80, 80)
torch.onnx.export(
    model.model,
    dummy_input,
    "liveness.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("✓ Liveness model exported")
EOF
```

**Pre-converted Option**:

```bash
# From PINTO Model Zoo
wget https://github.com/PINTO0309/PINTO_model_zoo/raw/main/168_Silent-Face-Anti-Spoofing/01_float32/anti_spoof_mn3_224x224.onnx -O liveness.onnx
```

**Alternative Models**:
- **FaceNet Anti-spoofing**
- **FAS-CNN** (Lightweight CNN)
- **FeatherNet** (Mobile-optimized)

### 3. MobileFaceNet (Face Recognition)

**Source**: InsightFace / DeepInsight

**Option A: Convert from InsightFace (MXNet)**

```bash
# Install dependencies
pip install mxnet onnx

# Clone InsightFace
git clone https://github.com/deepinsight/insightface.git
cd insightface/recognition/arcface_torch

# Download pre-trained MobileFaceNet
# Available at: https://github.com/deepinsight/insightface/tree/master/model_zoo

# Convert MXNet -> ONNX
python3 << EOF
import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

# Load MXNet model
sym = 'mobilefacenet-symbol.json'
params = 'mobilefacenet-0000.params'

# Convert
onnx_file = 'mobilefacenet.onnx'
onnx_mxnet.export_model(sym, params, [(1, 3, 112, 112)], np.float32, onnx_file)
print(f"✓ Exported to {onnx_file}")
EOF
```

**Option B: Use Pre-trained ONNX**

```bash
# From ONNX Model Zoo (if available)
wget https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx -O mobilefacenet.onnx

# Or from community repositories
git clone https://github.com/onnx/models.git
cd models/vision/body_analysis/arcface/
# Follow instructions to download
```

**Option C: Convert PyTorch -> ONNX**

```bash
pip install torch torchvision onnx facenet-pytorch

python3 << EOF
import torch
import torch.onnx
from facenet_pytorch import InceptionResnetV1

# Load pre-trained model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 160, 160)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "mobilefacenet.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("✓ Face recognition model exported")
EOF
```

**Alternative Models**:
- **ArcFace** - Higher accuracy (ResNet100)
- **FaceNet** - Google's classic model
- **CosFace** - Good balance of speed/accuracy

## Installation

Once you have the three ONNX files, install them:

```bash
sudo mkdir -p /var/lib/doorman/models
sudo cp blazeface.onnx /var/lib/doorman/models/
sudo cp liveness.onnx /var/lib/doorman/models/
sudo cp mobilefacenet.onnx /var/lib/doorman/models/
sudo chmod 644 /var/lib/doorman/models/*.onnx
```

## Verification

Test that models load correctly:

```bash
# Start daemon in debug mode
sudo RUST_LOG=debug /usr/local/bin/doormand

# Check logs for model loading
sudo journalctl -u doormand -f

# You should see:
# "Loaded 3/3 models"
```

Or use the status command:

```bash
sudo doorman status
# Should show "Models: ✓"
```

## Model Requirements

### Input/Output Specifications

**BlazeFace**:
- Input: `[1, 3, H, W]` (RGB image, any resolution, will be scaled)
- Output: Bounding boxes + confidence scores

**Liveness**:
- Input: `[1, 3, 80, 80]` or `[1, 3, 224, 224]` (face crop)
- Output: `[1, 2]` (scores for [fake, real])

**MobileFaceNet**:
- Input: `[1, 3, 112, 112]` (aligned face)
- Output: `[1, 512]` (embedding vector)

### Performance Notes

- **Inference time**: ~50-200ms per frame on CPU
- **Memory**: ~100-300 MB total for all models
- **Disk space**: ~10-20 MB total

## Troubleshooting

### Model Not Loading

```bash
# Check file exists and permissions
ls -lh /var/lib/doorman/models/

# Test manually with Python
python3 << EOF
import onnxruntime as ort
sess = ort.InferenceSession("/var/lib/doorman/models/blazeface.onnx")
print("✓ BlazeFace loads")
EOF
```

### Wrong Input Shape

If models expect different input shapes, update the preprocessing in `daemon/src/ml.rs`:

```rust
// Change downscale size to match your model
let small_img = image.resize_exact(
    112,  // Match your model's expected width
    112,  // Match your model's expected height
    image::imageops::FilterType::Lanczos3,
);
```

### Performance Issues

- Use smaller models (MobileFaceNet vs ArcFace-ResNet100)
- Reduce `AUTH_FRAMES` in `shared/src/lib.rs`
- Consider GPU acceleration (requires ONNX Runtime with CUDA)

## Advanced: Custom Models

To use your own trained models:

1. Ensure ONNX opset version 11+ for compatibility
2. Match input dimensions in `ml.rs`
3. Update embedding size if not 512-d
4. Test thoroughly before using in production

## Resources

- [ONNX Model Zoo](https://github.com/onnx/models)
- [PINTO Model Zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Silent-Face Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- [MediaPipe](https://google.github.io/mediapipe/)

## License Notes

Ensure you comply with the licenses of the models you use:
- **MediaPipe/BlazeFace**: Apache 2.0
- **Silent-Face**: MIT
- **InsightFace**: Varies by model (check model_zoo)

---

**Need help?** Open an issue with your model conversion logs and we'll assist!

