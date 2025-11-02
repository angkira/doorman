# Obtaining a Real Liveness Detection Model

Liveness detection (anti-spoofing) models are more restricted than face detection or recognition models due to security concerns. Here are real, PRODUCTION-GRADE options:

## Option 1: InsightFace Buffalo Models (RECOMMENDED)

**Most popular - 18k+ GitHub stars, production-ready**

InsightFace is THE industry-standard face analysis toolkit. Their Buffalo models include anti-spoofing.

### Installation:
```bash
pip install insightface onnxruntime
```

### Download models:
```python
import insightface
from insightface.app import FaceAnalysis

# This will auto-download buffalo models including liveness
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Models will be downloaded to: ~/.insightface/models/buffalo_l/
# Copy the liveness detector:
# ~/.insightface/models/buffalo_l/w600k_r50.onnx
```

### For Doorman:
```bash
# Find the model
ls ~/.insightface/models/buffalo_l/

# Copy to doorman (look for anti-spoofing/liveness model)
cp ~/.insightface/models/buffalo_l/*liveness*.onnx /path/to/doorman/data/models/liveness.onnx
# OR
cp ~/.insightface/models/buffalo_l/w600k_r50.onnx /path/to/doorman/data/models/liveness.onnx
```

**Model Specs:**
- Input: (1, 3, 112, 112) RGB
- Output: (1, 2) [fake, real] or (1, 1) score
- Normalization: mean/std or -1 to 1
- **Best accuracy and speed**

**Links:**
- GitHub: https://github.com/deepinsight/insightface (18k+ stars)
- Models: https://github.com/deepinsight/insightface/tree/master/model_zoo
- Docs: https://insightface.ai/

## Option 2: Silent Face Anti-Spoofing (MiniFASNet)

**Good alternative - 3.5k+ stars**

1. Clone the repository:
```bash
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
cd Silent-Face-Anti-Spoofing
```

2. Download pretrained checkpoints:
- Go to: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/master/resources/anti_spoof_models
- Download `2.7_80x80_MiniFASNetV2.pth` or similar

3. Convert to ONNX:
```bash
pip install torch onnx
python convert_to_onnx.py --model 2.7_80x80_MiniFASNetV2 --output liveness.onnx
```

4. Place in doorman:
```bash
cp liveness.onnx /path/to/doorman/data/models/
```

**Model Specs:**
- Input: (1, 3, 80, 80) RGB
- Output: (1, 3) [real, print, replay]
- Normalization: divide by 255

## Option 3: Train Your Own

Use the OULU-NPU, CASIA-FASD, or Replay-Attack datasets:

```bash
# Using Silent-Face-Anti-Spoofing
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
cd Silent-Face-Anti-Spoofing

# Download dataset (requires registration)
# - OULU-NPU: https://sites.google.com/site/oulunpudatabase/
# - CASIA-FASD: http://www.cbsr.ia.ac.cn/english/FASDB_Agreement/Agreement.pdf

# Train
python train.py --dataset oulu --model MiniFASNetV2

# Convert to ONNX
python export_onnx.py --checkpoint ./checkpoints/best.pth --output liveness.onnx
```

## Option 4: Commercial Solutions

If you need production-ready models with support:

1. **KBY-AI Face Liveness Detection SDK**
   - https://github.com/kby-ai/Face-Liveness-Detection-SDK
   - Provides ONNX models
   - May require license for commercial use

2. **Recognito Face Liveness SDK**
   - https://recognito.vision/face-liveness-detection-sdk/
   - iBeta Level 1 & 2 certified
   - Commercial licensing

3. **Luxand FaceSDK**
   - https://www.luxand.com/facesdk/liveness-download/
   - Includes liveness detection
   - Paid license required

## Option 5: Alternative Open-Source Models

### FaceAntiSpoofing by hanxuanliang
```bash
git clone https://github.com/hanxuanliang/FaceAntiSpoofing.git
cd FaceAntiSpoofing
# Follow their README for pretrained models
```

### FaceAntiSpoofing by clks-wzz
```bash
git clone https://github.com/clks-wzz/FaceAntiSpoofing.git
# Check releases for pretrained models
```

## Recommended Approach for Doorman

**⭐ BEST: Use InsightFace (Option 1)**
1. Install insightface: `pip install insightface onnxruntime`
2. Download buffalo models (auto-download on first run)
3. Copy liveness model to `data/models/liveness.onnx`
4. **18k+ stars, production-proven, best accuracy**

**Alternative: Silent-Face-Anti-Spoofing (Option 2)**
1. Clone Silent-Face-Anti-Spoofing (3.5k+ stars)
2. Use their pretrained MiniFASNetV2 checkpoint
3. Convert to ONNX
4. Place in `data/models/liveness.onnx`

**For Production:**
1. **InsightFace is already production-ready** (used by thousands)
2. Or evaluate commercial SDKs for compliance/support
3. Ensure proper testing against various attack types

## Converting PyTorch to ONNX

Generic conversion script:

```python
import torch
import torch.onnx

# Load your trained model
model = YourLivenessModel()
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 80, 80)  # Adjust size as needed

# Export
torch.onnx.export(
    model,
    dummy_input,
    'liveness.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

## Verification

Once you have `liveness.onnx`:

```bash
# Verify it's in the right place
doorman models list

# Check the model specs
doorman models verify liveness

# Test with real camera
doorman enroll testuser
```

## Important Notes

- **Security**: Anti-spoofing models are security-critical. Use only well-tested models.
- **Dataset Quality**: Train on diverse datasets including print attacks, replay attacks, and 3D masks.
- **Regular Updates**: Spoofing techniques evolve; keep models updated.
- **False Positives**: Tune thresholds to balance security vs user experience.

