#!/usr/bin/env python3
import socket
import struct
from PIL import Image
import io

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
    
    # Decode with PIL
    try:
        img = Image.open(io.BytesIO(jpeg_data))
        print(f"Image format: {img.format}")
        print(f"Image mode: {img.mode}")
        print(f"Image size: {img.size} (width x height)")
        print(f"Expected RGB bytes: {img.size[0] * img.size[1] * 3}")
    except Exception as e:
        print(f"Failed to decode: {e}")
        print(f"JPEG header: {jpeg_data[:20].hex()}")

sock.close()
