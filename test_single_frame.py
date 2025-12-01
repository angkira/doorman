#!/usr/bin/env python3
"""Test face detection on a single frame"""

import cv2
import numpy as np

def test_frame():
    # Load frame
    img = cv2.imread('test_frame.jpg')
    if img is None:
        print("❌ Failed to load test_frame.jpg")
        return
    
    h, w = img.shape[:2]
    print(f"📷 Frame loaded: {w}x{h}")
    
    # Simple face detection using OpenCV (for comparison)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print(f"\n👤 OpenCV detected {len(faces)} faces")
    
    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            print(f"  Face {i+1}: x={x}, y={y}, w={w}, h={h}")
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save annotated image
        cv2.imwrite('test_frame_annotated.jpg', img)
        print("\n✅ Saved annotated image to test_frame_annotated.jpg")
    else:
        print("\n⚠️  No faces detected by OpenCV")
        print("This might explain why BlazeFace also fails")
        print("Possible reasons:")
        print("  - Face too small in 4K frame")
        print("  - Face at angle/profile")
        print("  - Poor lighting")
        print("  - Face occluded")

if __name__ == '__main__':
    test_frame()
