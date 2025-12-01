#!/usr/bin/env python3
"""
Simple camera test - NO ML dependencies
Just tests if we can capture frames at 30 FPS
"""
import subprocess
import sys
import time

def test_ffmpeg_direct():
    """Test 1: Direct FFmpeg capture to files"""
    print("\n" + "="*60)
    print("TEST 1: FFmpeg Direct Capture (10 frames)")
    print("="*60)

    cmd = [
        "ffmpeg",
        "-f", "v4l2",
        "-video_size", "1280x720",
        "-framerate", "30",
        "-i", "/dev/video0",
        "-frames:v", "10",
        "-f", "image2",
        "frame_%03d.jpg"
    ]

    print(f"Running: {' '.join(cmd)}")
    start = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"✅ SUCCESS: Captured 10 frames in {elapsed:.2f}s")
        print(f"   FPS: {10/elapsed:.1f}")
        print("\nCheck current directory for frame_001.jpg, frame_002.jpg, etc.")
        return True
    else:
        print(f"❌ FAILED")
        print(f"stderr: {result.stderr}")
        return False

def test_ffmpeg_stream():
    """Test 2: FFmpeg continuous streaming (raw RGB)"""
    print("\n" + "="*60)
    print("TEST 2: FFmpeg Continuous Stream (30 frames)")
    print("="*60)

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-f", "v4l2",
        "-video_size", "1280x720",
        "-framerate", "30",
        "-i", "/dev/video0",
        "-frames:v", "30",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-"
    ]

    print(f"Running: ffmpeg ... (streaming RGB)")
    start = time.time()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_size = 1280 * 720 * 3  # RGB24
    frames_read = 0

    try:
        while frames_read < 30:
            data = proc.stdout.read(frame_size)
            if len(data) != frame_size:
                print(f"⚠️  Frame {frames_read+1}: Got {len(data)} bytes, expected {frame_size}")
                break
            frames_read += 1
            if frames_read % 10 == 0:
                print(f"   Read {frames_read} frames...")
    except Exception as e:
        print(f"❌ Error reading frames: {e}")

    proc.terminate()
    proc.wait()

    elapsed = time.time() - start

    if frames_read > 0:
        print(f"✅ Read {frames_read}/30 frames in {elapsed:.2f}s")
        print(f"   FPS: {frames_read/elapsed:.1f}")
        return frames_read == 30
    else:
        print(f"❌ FAILED to read any frames")
        stderr = proc.stderr.read().decode()
        print(f"stderr: {stderr}")
        return False

def test_v4l2_info():
    """Test 3: Check V4L2 camera capabilities"""
    print("\n" + "="*60)
    print("TEST 3: V4L2 Camera Info")
    print("="*60)

    result = subprocess.run(
        ["v4l2-ctl", "--device=/dev/video0", "--all"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ Camera information:")
        for line in result.stdout.split('\n')[:30]:  # First 30 lines
            print(f"   {line}")
        return True
    else:
        print("❌ v4l2-ctl not found. Install with: sudo apt-get install v4l-utils")
        return False

def test_formats():
    """Test 4: List supported formats"""
    print("\n" + "="*60)
    print("TEST 4: Supported Formats")
    print("="*60)

    result = subprocess.run(
        ["v4l2-ctl", "--device=/dev/video0", "--list-formats-ext"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ Supported formats:")
        print(result.stdout)
        return True
    else:
        print("❌ Failed to list formats")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          SIMPLE CAMERA TEST - NO ML DEPENDENCIES             ║
║                                                              ║
║  Tests camera access with ffmpeg and v4l2                   ║
╚══════════════════════════════════════════════════════════════╝
""")

    tests = [
        ("V4L2 Info", test_v4l2_info),
        ("Supported Formats", test_formats),
        ("FFmpeg Direct Capture", test_ffmpeg_direct),
        ("FFmpeg Stream", test_ffmpeg_stream),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results[name] = False

        input("\nPress Enter to continue to next test...")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n{passed}/{total} tests passed")

if __name__ == "__main__":
    main()
