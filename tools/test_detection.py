#!/usr/bin/env python3
"""Quick test of BlazeFace detection with real image"""

import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Add daemon/src/ml to path
sys.path.insert(0, str(Path(__file__).parent.parent / "daemon" / "src" / "ml"))

from torch_inference import TorchInferenceBackend

def create_test_image_with_face():
    """Create simple test image with a face-like shape"""
    img = Image.new('RGB', (640, 480), color='white')
    draw = ImageDraw.Draw(img)

    # Draw simple face (circle + eyes + mouth)
    draw.ellipse([220, 140, 420, 340], fill='beige', outline='black')
    draw.ellipse([270, 190, 300, 220], fill='black')  # left eye
    draw.ellipse([340, 190, 370, 220], fill='black')  # right eye
    draw.arc([270, 240, 370, 290], 0, 180, fill='black', width=3)  # smile

    return img

def main():
    models_dir = os.path.expanduser("~/.local/share/doorman/models")

    print("=== BlazeFace Detection Test ===\n")
    print(f"Models dir: {models_dir}")
    print("Initializing backend with ROCm...")

    backend = TorchInferenceBackend(models_dir, device="cuda")

    print("\n--- Test 1: Black image (no face) ---")
    black_img = Image.new('RGB', (640, 480), color='black')
    import io
    buf = io.BytesIO()
    black_img.save(buf, format='JPEG')
    result = backend.detect_faces(buf.getvalue(), 640, 480)
    print(f"Detections: {len(result['detections'])}")
    if result['detections']:
        print(f"ERROR: Should not detect face in black image!")
        print(f"Got: {result['detections'][0]}")
    else:
        print("✓ Correctly detected no faces")

    print("\n--- Test 2: Simple face drawing ---")
    face_img = create_test_image_with_face()
    buf = io.BytesIO()
    face_img.save(buf, format='JPEG')
    result = backend.detect_faces(buf.getvalue(), 640, 480)
    print(f"Detections: {len(result['detections'])}")

    if result['detections']:
        det = result['detections'][0]
        print(f"✓ Detected face!")
        print(f"  Position: x={det['x']:.1f}, y={det['y']:.1f}")
        print(f"  Size: w={det['width']:.1f}, h={det['height']:.1f}")
        print(f"  Confidence: {det['confidence']:.3f}")

        # Check if coordinates are reasonable (not mock data like 100, 100)
        if abs(det['x'] - 100) < 1 and abs(det['y'] - 100) < 1:
            print("⚠ WARNING: Looks like mock data (x=100, y=100)")
        else:
            print("✓ Coordinates look real (not mock data)")

        # Visualize
        draw = ImageDraw.Draw(face_img)
        x, y, w, h = det['x'], det['y'], det['width'], det['height']
        draw.rectangle([x, y, x+w, y+h], outline='red', width=3)
        face_img.save('/tmp/doorman_detection_test.jpg')
        print(f"\n✓ Saved visualization to /tmp/doorman_detection_test.jpg")
    else:
        print("✗ No face detected (might be too simple drawing)")

    print("\n--- Test 3: Performance check ---")
    import time
    iterations = 10
    start = time.perf_counter()
    for _ in range(iterations):
        backend.detect_faces(buf.getvalue(), 640, 480)
    end = time.perf_counter()

    avg_time = ((end - start) / iterations) * 1000
    fps = 1000 / avg_time
    print(f"Average time: {avg_time:.2f}ms")
    print(f"FPS: {fps:.1f}")

    if fps > 50:
        print("✓ Performance is good (>50 FPS)")
    elif fps > 20:
        print("⚠ Performance is acceptable (>20 FPS)")
    else:
        print("✗ Performance is low (<20 FPS) - check GPU usage")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
