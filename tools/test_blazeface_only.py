#!/usr/bin/env python3
"""Quick test of BlazeFace detection only"""

import sys
import os
import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import io

# Setup environment
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
os.environ['ORT_LOG_LEVEL'] = '3'

import onnxruntime as ort

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

def detect_faces(session, image_data, width, height):
    """Detect faces - simplified version"""
    # Decode image
    orig_image = Image.open(io.BytesIO(image_data)).convert('RGB')
    orig_width, orig_height = orig_image.size

    # Resize to 128x128 with letterboxing
    target_size = 128
    scale = min(target_size / orig_width, target_size / orig_height)
    resized_w = int(orig_width * scale)
    resized_h = int(orig_height * scale)

    resized_image = orig_image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

    # Create black canvas and paste resized image
    letterboxed = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    offset_x = (target_size - resized_w) // 2
    offset_y = (target_size - resized_h) // 2
    letterboxed.paste(resized_image, (offset_x, offset_y))

    # Preprocess
    img_array = np.array(letterboxed).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
    img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

    # Inference
    outputs = session.run(None, {'input': img_array})

    # Decode BlazeFace output
    scores_tensor = outputs[0]
    boxes_tensor = outputs[1]

    scores = scores_tensor[0]
    boxes = boxes_tensor[0]

    # Find best detection
    confidence_threshold = 0.5
    detections = []

    for i in range(len(scores)):
        face_score = scores[i][1]

        if face_score > confidence_threshold:
            top_y = boxes[i][0]
            top_x = boxes[i][1]
            bot_y = boxes[i][2]
            bot_x = boxes[i][3]

            # Convert to pixel coordinates in original image
            x_letterbox = top_x * target_size
            y_letterbox = top_y * target_size
            x2_letterbox = bot_x * target_size
            y2_letterbox = bot_y * target_size

            x_resized = x_letterbox - offset_x
            y_resized = y_letterbox - offset_y
            x2_resized = x2_letterbox - offset_x
            y2_resized = y2_letterbox - offset_y

            x_orig = (x_resized / resized_w) * orig_width
            y_orig = (y_resized / resized_h) * orig_height
            x2_orig = (x2_resized / resized_w) * orig_width
            y2_orig = (y2_resized / resized_h) * orig_height

            x = float(max(0, x_orig))
            y = float(max(0, y_orig))
            w = float(max(1, abs(x2_orig - x_orig)))
            h = float(max(1, abs(y2_orig - y_orig)))

            detections.append({
                "x": x, "y": y, "width": w, "height": h,
                "confidence": float(face_score)
            })

    if detections:
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections[0]
    return None

def main():
    models_dir = os.path.expanduser("~/.local/share/doorman/models")
    blazeface_path = os.path.join(models_dir, "blazeface.onnx")

    print("=== BlazeFace Detection Test ===\n")
    print(f"Model: {blazeface_path}")
    print("Initializing ONNX Runtime with MIGraphX...")
    print("(First run will compile model for GPU - takes ~30s)\n")

    # Load model
    providers = ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(blazeface_path, providers=providers)

    print(f"✓ Model loaded on: {session.get_providers()[0]}\n")

    # Test 1: Black image
    print("--- Test 1: Black image (no face) ---")
    black_img = Image.new('RGB', (640, 480), color='black')
    buf = io.BytesIO()
    black_img.save(buf, format='JPEG')

    det = detect_faces(session, buf.getvalue(), 640, 480)
    if det is None:
        print("✓ Correctly detected no faces\n")
    else:
        print(f"⚠ Unexpected detection: {det}\n")

    # Test 2: Face drawing
    print("--- Test 2: Simple face drawing ---")
    face_img = create_test_image_with_face()
    buf = io.BytesIO()
    face_img.save(buf, format='JPEG')

    det = detect_faces(session, buf.getvalue(), 640, 480)
    if det:
        print(f"✓ Detected face!")
        print(f"  Position: x={det['x']:.1f}, y={det['y']:.1f}")
        print(f"  Size: w={det['width']:.1f}, h={det['height']:.1f}")
        print(f"  Confidence: {det['confidence']:.3f}")

        # Check if mock data
        if abs(det['x'] - 100) < 1 and abs(det['y'] - 100) < 1:
            print("  ⚠ WARNING: Looks like mock data!")
        else:
            print("  ✓ Real coordinates (not mock)")

        # Visualize
        draw = ImageDraw.Draw(face_img)
        x, y, w, h = det['x'], det['y'], det['width'], det['height']
        draw.rectangle([x, y, x+w, y+h], outline='red', width=3)
        face_img.save('/tmp/doorman_blazeface_test.jpg')
        print(f"\n✓ Saved visualization to /tmp/doorman_blazeface_test.jpg\n")
    else:
        print("✗ No face detected\n")

    # Test 3: Performance
    print("--- Test 3: Performance (10 iterations) ---")
    iterations = 10
    start = time.perf_counter()
    for _ in range(iterations):
        detect_faces(session, buf.getvalue(), 640, 480)
    end = time.perf_counter()

    avg_time = ((end - start) / iterations) * 1000
    fps = 1000 / avg_time
    print(f"Average time: {avg_time:.2f}ms")
    print(f"FPS: {fps:.1f}")

    if fps > 50:
        print("✓ Excellent performance (>50 FPS)\n")
    elif fps > 20:
        print("✓ Good performance (>20 FPS)\n")
    else:
        print("⚠ Low performance (<20 FPS)\n")

    print("=== Test Complete ===")

if __name__ == "__main__":
    main()
