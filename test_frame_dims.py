#!/usr/bin/env python3
import socket
import struct

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/run/user/1000/doorman-frames.sock")

# Read one frame
size_bytes = sock.recv(4)
if len(size_bytes) == 4:
    size = struct.unpack('>I', size_bytes)[0]
    print(f"Frame size: {size} bytes ({size/1024:.1f} KB)")
    
    jpeg_data = b""
    while len(jpeg_data) < size:
        jpeg_data += sock.recv(size - len(jpeg_data))
    
    # Try to decode and check dimensions
    import cv2
    import numpy as np
    arr = np.frombuffer(jpeg_data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        print(f"Decoded image shape: {img.shape}")
        print(f"Dimensions: {img.shape[1]}x{img.shape[0]} (WxH)")
    else:
        print("Failed to decode JPEG")
        # Check JPEG header
        if jpeg_data[:2] == b'\xff\xd8':
            print("Valid JPEG start marker")
        else:
            print(f"Invalid JPEG marker: {jpeg_data[:10].hex()}")

sock.close()
