#!/usr/bin/env python3
"""
Simple test to check what debug stream sends
"""
import socket
import json
import os

def main():
    user_runtime_dir = os.getenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    socket_path = f"{user_runtime_dir}/doorman-debug.sock"
    
    print(f"Connecting to {socket_path}...")
    
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        print("✓ Connected!")
        print("Waiting for messages (Ctrl+C to exit)...\n")
        
        buffer = b""
        count = 0
        
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                print("Connection closed")
                break
            
            buffer += chunk
            
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line:
                    try:
                        msg = json.loads(line.decode())
                        count += 1
                        detection = msg.get("detection", {})
                        bbox = detection.get("bbox")
                        confidence = detection.get("confidence")
                        frame_size = detection.get("frame_size")
                        
                        print(f"Message #{count}:")
                        print(f"  bbox: {bbox}")
                        print(f"  confidence: {confidence}")
                        print(f"  frame_size: {frame_size}")
                        print()
                        
                    except json.JSONDecodeError as e:
                        print(f"Parse error: {e}")
                        
    except FileNotFoundError:
        print(f"✗ Socket not found: {socket_path}")
        print("  Make sure daemon is running with --preview flag")
    except KeyboardInterrupt:
        print("\n✓ Done")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    main()
