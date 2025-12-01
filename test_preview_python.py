#!/usr/bin/env python3
"""
Pure Python Preview - Test Face Detection Pipeline
Bypasses Rust daemon entirely to test camera + ML pipeline
"""
import cv2
import numpy as np
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def draw_fps(frame, fps):
    """Draw FPS counter on frame"""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def draw_detection(frame, bbox, confidence, color):
    """Draw bounding box with confidence"""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    # Draw confidence
    text = f"{confidence:.2f}"
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def test_opencv_camera():
    """Test 1: Pure OpenCV camera capture (no ML)"""
    print("\n" + "="*60)
    print("TEST 1: OpenCV Camera Capture (30 FPS target)")
    print("="*60)
    print("Press 'q' to quit\n")

    cap = cv2.VideoCapture(0)

    # Set camera to 1280x720 @ 30fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Verify settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS)

    print(f"✓ Camera opened: {width}x{height} @ {fps_cap} fps")

    frame_times = []
    frame_count = 0

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        # Calculate FPS
        frame_times.append(time.time() - start)
        if len(frame_times) > 30:
            frame_times.pop(0)

        avg_time = sum(frame_times) / len(frame_times)
        current_fps = 1.0 / avg_time if avg_time > 0 else 0

        # Draw FPS
        draw_fps(frame, current_fps)

        # Show frame
        cv2.imshow('OpenCV Camera Test', frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frames: {frame_count}, FPS: {current_fps:.1f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Test complete. Average FPS: {current_fps:.1f}\n")

def test_face_detection():
    """Test 2: Camera + Face Detection with BlazeFace"""
    print("\n" + "="*60)
    print("TEST 2: OpenCV Camera + Face Detection")
    print("="*60)
    print("Loading BlazeFace model...")

    # Import InsightFace
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✓ InsightFace loaded (using buffalo_l models)")
    except Exception as e:
        print(f"❌ Failed to load InsightFace: {e}")
        print("\nTrying OpenCV Haar Cascade as fallback...")
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        app = None

    print("Press 'q' to quit\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_times = []
    detect_times = []
    frame_count = 0

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        # Detect faces
        detect_start = time.time()

        if app is not None:
            # InsightFace detection
            faces = app.get(frame)
            detect_time = (time.time() - detect_start) * 1000  # ms

            for face in faces:
                bbox = face.bbox.astype(int)
                confidence = face.det_score

                # Color based on confidence
                if confidence > 0.95:
                    color = (0, 255, 0)  # Green - high confidence
                elif confidence > 0.8:
                    color = (0, 255, 255)  # Yellow - medium
                else:
                    color = (0, 165, 255)  # Orange - low

                draw_detection(frame, bbox, confidence, color)
        else:
            # Haar Cascade fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            detect_time = (time.time() - detect_start) * 1000

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        detect_times.append(detect_time)
        if len(detect_times) > 30:
            detect_times.pop(0)

        # Calculate FPS
        frame_times.append(time.time() - start)
        if len(frame_times) > 30:
            frame_times.pop(0)

        avg_time = sum(frame_times) / len(frame_times)
        current_fps = 1.0 / avg_time if avg_time > 0 else 0
        avg_detect = sum(detect_times) / len(detect_times)

        # Draw stats
        draw_fps(frame, current_fps)
        cv2.putText(frame, f"Detect: {avg_detect:.1f}ms", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Show frame
        cv2.imshow('Face Detection Test', frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frames: {frame_count}, FPS: {current_fps:.1f}, "
                  f"Detect: {avg_detect:.1f}ms, Faces: {len(faces)}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Test complete. Average FPS: {current_fps:.1f}, "
          f"Detection time: {avg_detect:.1f}ms\n")

def test_full_pipeline():
    """Test 3: Full pipeline with recognition"""
    print("\n" + "="*60)
    print("TEST 3: Full Pipeline (Detection + Recognition)")
    print("="*60)
    print("Loading models...")

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✓ InsightFace loaded with recognition")
    print("Press 'q' to quit\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_times = []
    pipeline_times = []
    frame_count = 0

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Full pipeline
        pipeline_start = time.time()
        faces = app.get(frame)
        pipeline_time = (time.time() - pipeline_start) * 1000

        pipeline_times.append(pipeline_time)
        if len(pipeline_times) > 30:
            pipeline_times.pop(0)

        # Draw results
        for face in faces:
            bbox = face.bbox.astype(int)
            confidence = face.det_score

            # Simulate recognition check
            # TODO: Load embeddings and compare
            has_embedding = hasattr(face, 'embedding')

            if has_embedding:
                # Green if we have embedding (recognized)
                color = (0, 255, 0)
                text = f"OK {confidence:.2f}"
            else:
                # Red if no embedding
                color = (0, 0, 255)
                text = f"? {confidence:.2f}"

            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Calculate FPS
        frame_times.append(time.time() - start)
        if len(frame_times) > 30:
            frame_times.pop(0)

        avg_time = sum(frame_times) / len(frame_times)
        current_fps = 1.0 / avg_time if avg_time > 0 else 0
        avg_pipeline = sum(pipeline_times) / len(pipeline_times)

        # Draw stats
        draw_fps(frame, current_fps)
        cv2.putText(frame, f"Pipeline: {avg_pipeline:.1f}ms", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow('Full Pipeline Test', frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frames: {frame_count}, FPS: {current_fps:.1f}, "
                  f"Pipeline: {avg_pipeline:.1f}ms, Faces: {len(faces)}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Test complete. Average FPS: {current_fps:.1f}, "
          f"Pipeline: {avg_pipeline:.1f}ms\n")

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║        DOORMAN PREVIEW DEBUG - PURE PYTHON TESTS             ║
║                                                              ║
║  This bypasses the Rust daemon to test camera + ML          ║
║  Run each test to isolate issues:                           ║
║                                                              ║
║  1. OpenCV camera capture (baseline 30 FPS)                 ║
║  2. Camera + face detection (with bboxes)                   ║
║  3. Full pipeline (detection + recognition)                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    while True:
        print("\nSelect test:")
        print("  1 - OpenCV camera only (baseline)")
        print("  2 - Camera + face detection")
        print("  3 - Full pipeline (detection + recognition)")
        print("  q - Quit")

        choice = input("\nYour choice: ").strip()

        if choice == '1':
            test_opencv_camera()
        elif choice == '2':
            test_face_detection()
        elif choice == '3':
            test_full_pipeline()
        elif choice.lower() == 'q':
            print("\n✓ Done!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
