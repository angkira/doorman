#!/bin/bash
# Quick BlazeFace-only test (skip liveness/recognition to avoid long compile)

set -e

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ORT_LOG_LEVEL=3

source .venv/bin/activate

echo "=== Quick BlazeFace Test ==="
echo ""
echo "Testing only BlazeFace detector (skip liveness/recognition)"
echo "This avoids long MobileFaceNet compilation (249MB)"
echo ""

timeout 120 python3 -c "
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import time

models_dir = '$HOME/.local/share/doorman/models'
blazeface = models_dir + '/blazeface.onnx'

print('Loading BlazeFace on MIGraphX...')
print('(First run compiles for GPU - takes ~30s)')
print('')

providers = ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(blazeface, providers=providers)

print(f'✓ Model on: {session.get_providers()[0]}')
print('')

# Test 1: Black image (no face)
print('Test 1: Black image (no face)')
img = Image.new('RGB', (640, 480), 'black')

# Letterbox to 240x320
scale = min(320/640, 240/480)
resized_w, resized_h = int(640*scale), int(480*scale)
resized = img.resize((resized_w, resized_h))
letterboxed = Image.new('RGB', (320, 240), (0,0,0))
offset_x = (320 - resized_w) // 2
offset_y = (240 - resized_h) // 2
letterboxed.paste(resized, (offset_x, offset_y))

# Preprocess
arr = np.array(letterboxed).astype(np.float32) / 255.0
arr = (arr - 0.5) / 0.5
arr = arr.transpose(2,0,1)[np.newaxis, ...]

# Inference
outputs = session.run(None, {'input': arr})
scores = outputs[0][0]
boxes = outputs[1][0]

# Count detections
detections = sum(1 for i in range(len(scores)) if scores[i][1] > 0.5)
print(f'  Detections: {detections}')
if detections == 0:
    print('  ✓ Correct (no face)')
else:
    print(f'  ⚠ Unexpected: found {detections} faces in black image')

print('')

# Test 2: Performance
print('Test 2: Performance (10 iterations)')
iterations = 10
start = time.perf_counter()
for _ in range(iterations):
    session.run(None, {'input': arr})
end = time.perf_counter()

avg_ms = ((end - start) / iterations) * 1000
fps = 1000 / avg_ms

print(f'  Avg time: {avg_ms:.2f}ms')
print(f'  FPS: {fps:.1f}')

if fps > 100:
    print('  ✓ Excellent (>100 FPS)')
elif fps > 50:
    print('  ✓ Good (>50 FPS)')
elif fps > 20:
    print('  ⚠ Acceptable (>20 FPS)')
else:
    print('  ✗ Low (<20 FPS) - GPU issue?')

print('')
print('=== Test Complete ===')
" 2>&1
