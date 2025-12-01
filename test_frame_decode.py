#!/usr/bin/env python3
"""
Test frame decoding from daemon
Saves first frame as test_output.jpg to verify JPEG encoding
"""
import socket
import struct
import sys

SOCKET_PATH = "/run/user/1000/doorman-frames.sock"

def recv_exactly(sock, n):
    """Receive exactly n bytes"""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def main():
    print(f"Connecting to {SOCKET_PATH}...")

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCKET_PATH)
        print("✓ Connected")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return 1

    # Read first frame
    print("Reading frame size...")
    size_bytes = recv_exactly(sock, 4)
    if not size_bytes:
        print("❌ Failed to read size")
        return 1

    frame_size = struct.unpack('>I', size_bytes)[0]
    print(f"✓ Frame size: {frame_size} bytes ({frame_size/1024:.1f} KB)")

    if frame_size > 10*1024*1024:  # 10MB sanity check
        print(f"❌ Frame size too large: {frame_size}")
        return 1

    print("Reading JPEG data...")
    jpeg_data = recv_exactly(sock, frame_size)
    if not jpeg_data:
        print("❌ Failed to read JPEG data")
        return 1

    print(f"✓ Received {len(jpeg_data)} bytes")

    # Check JPEG magic bytes (FFD8 at start, FFD9 at end)
    if jpeg_data[:2] != b'\xff\xd8':
        print(f"❌ Invalid JPEG header: {jpeg_data[:4].hex()}")
        print(f"   Expected: ffd8xxxx")
        return 1

    if jpeg_data[-2:] != b'\xff\xd9':
        print(f"❌ Invalid JPEG footer: {jpeg_data[-4:].hex()}")
        print(f"   Expected: xxxxffd9")
        return 1

    print("✓ Valid JPEG magic bytes")

    # Save to file
    output_path = "test_output.jpg"
    with open(output_path, 'wb') as f:
        f.write(jpeg_data)

    print(f"✓ Saved to {output_path}")

    # Try to decode with OpenCV if available
    try:
        import cv2
        import numpy as np

        nparr = np.frombuffer(jpeg_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("❌ OpenCV failed to decode JPEG")
            return 1

        h, w, c = img.shape
        print(f"✓ OpenCV decoded: {w}x{h}, {c} channels")

        # Save decoded version
        cv2.imwrite("test_decoded.jpg", img)
        print(f"✓ Saved decoded to test_decoded.jpg")

    except ImportError:
        print("⚠ OpenCV not available, skipping decode test")
    except Exception as e:
        print(f"❌ Decode error: {e}")
        return 1

    sock.close()
    print("\n✅ Frame test PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
