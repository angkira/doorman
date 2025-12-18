#!/usr/bin/env python3
"""
PyTorch inference server with Shared Memory optimization.
Zero-copy frame transfer via shared memory + Unix socket control.
"""

import os
import sys
import json
import socket
import numpy as np
from pathlib import Path
from posix_ipc import SharedMemory

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
tools_path = project_root / "tools"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(tools_path))

from torch_models import load_models, detect_faces, check_liveness, extract_embedding


class InferenceServer:
    def __init__(self, models_dir: str, device: str, shm_name: str, socket_path: str):
        self.models_dir = models_dir
        self.device = device
        self.shm_name = shm_name
        self.socket_path = socket_path
        
        # Open shared memory
        self.shm = SharedMemory(shm_name)
        print(f"Opened shared memory: {shm_name}", file=sys.stderr, flush=True)
        
        # Load models
        print(f"Loading models from {models_dir} on {device}...", file=sys.stderr, flush=True)
        self.models = load_models(models_dir, device)
        print("✓ Models loaded", file=sys.stderr, flush=True)
        
        # Warmup
        print("Warming up models...", file=sys.stderr, flush=True)
        self._warmup()
        print("✓ Models warmed up", file=sys.stderr, flush=True)
        
        # Create Unix socket server
        if os.path.exists(socket_path):
            os.remove(socket_path)
        
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(socket_path)
        self.server_socket.listen(1)
        print(f"✓ Server listening on {socket_path}", file=sys.stderr, flush=True)
    
    def _warmup(self):
        """Warm up models with dummy data."""
        # Dummy 640x480 RGB image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warmup detection
        _ = detect_faces(self.models['detector'], dummy_image, self.device)
        
        # Warmup liveness (112x112 face)
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        _ = check_liveness(self.models['liveness'], dummy_face, self.device)
        
        # Warmup recognition
        _ = extract_embedding(self.models['recognizer'], dummy_face, self.device)
    
    def read_frame_from_shm(self, width: int, height: int) -> np.ndarray:
        """Read frame from shared memory."""
        size = height * width * 3
        
        # Memory-map shared memory
        import mmap
        mm = mmap.mmap(self.shm.fd, size)
        
        # Read frame data
        data = np.frombuffer(mm, dtype=np.uint8, count=size)
        frame = data.reshape((height, width, 3))
        
        mm.close()
        return frame
    
    def handle_detect(self, width: int, height: int) -> dict:
        """Handle face detection request."""
        try:
            # Read frame from shared memory (zero-copy!)
            frame = self.read_frame_from_shm(width, height)
            
            # Detect faces
            detections = detect_faces(self.models['detector'], frame, self.device)
            
            return {
                "detections": [
                    {
                        "bbox": [float(x) for x in det['bbox']],
                        "confidence": float(det['confidence'])
                    }
                    for det in detections
                ]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def handle_liveness(self, width: int, height: int) -> dict:
        """Handle liveness check request."""
        try:
            # Read face from shared memory
            face = self.read_frame_from_shm(width, height)
            
            # Check liveness
            score = check_liveness(self.models['liveness'], face, self.device)
            
            return {"score": float(score)}
        except Exception as e:
            return {"error": str(e)}
    
    def handle_embedding(self, width: int, height: int) -> dict:
        """Handle embedding extraction request."""
        try:
            # Read face from shared memory
            face = self.read_frame_from_shm(width, height)
            
            # Extract embedding
            embedding = extract_embedding(self.models['recognizer'], face, self.device)
            
            return {"embedding": embedding.tolist()}
        except Exception as e:
            return {"error": str(e)}
    
    def handle_warmup(self) -> dict:
        """Handle warmup request."""
        try:
            self._warmup()
            return {"status": "ok"}
        except Exception as e:
            return {"error": str(e)}
    
    def run(self):
        """Run server loop."""
        print("✓ Server ready", file=sys.stderr, flush=True)
        
        conn, _ = self.server_socket.accept()
        print("✓ Client connected", file=sys.stderr, flush=True)
        
        try:
            while True:
                # Read command: "command width height\n"
                line = b""
                while True:
                    byte = conn.recv(1)
                    if not byte or byte == b'\n':
                        break
                    line += byte
                
                if not line:
                    break
                
                parts = line.decode('utf-8').strip().split()
                if not parts:
                    continue
                
                command = parts[0]
                
                # Handle command
                if command == "detect":
                    width, height = int(parts[1]), int(parts[2])
                    response = self.handle_detect(width, height)
                elif command == "liveness":
                    width, height = int(parts[1]), int(parts[2])
                    response = self.handle_liveness(width, height)
                elif command == "embedding":
                    width, height = int(parts[1]), int(parts[2])
                    response = self.handle_embedding(width, height)
                elif command == "warmup":
                    response = self.handle_warmup()
                elif command == "shutdown":
                    response = {"status": "shutting down"}
                    conn.sendall(json.dumps(response).encode('utf-8') + b'\n')
                    break
                else:
                    response = {"error": f"Unknown command: {command}"}
                
                # Send response
                conn.sendall(json.dumps(response).encode('utf-8') + b'\n')
        
        finally:
            conn.close()
            self.server_socket.close()
            if os.path.exists(self.socket_path):
                os.remove(self.socket_path)
            print("✓ Server shut down", file=sys.stderr, flush=True)


def main():
    models_dir = os.environ.get('DOORMAN_MODELS_DIR', '~/.local/share/doorman/models')
    device = os.environ.get('DOORMAN_DEVICE', 'cpu')
    shm_name = os.environ.get('DOORMAN_SHM_NAME', 'doorman_shm')
    socket_path = os.environ.get('DOORMAN_SOCKET_PATH', '/tmp/doorman-inference.sock')
    
    # Expand home directory
    models_dir = os.path.expanduser(models_dir)
    
    print(f"Starting inference server...", file=sys.stderr, flush=True)
    print(f"Models: {models_dir}", file=sys.stderr, flush=True)
    print(f"Device: {device}", file=sys.stderr, flush=True)
    print(f"Shared memory: {shm_name}", file=sys.stderr, flush=True)
    print(f"Socket: {socket_path}", file=sys.stderr, flush=True)
    
    try:
        server = InferenceServer(models_dir, device, shm_name, socket_path)
        server.run()
    except KeyboardInterrupt:
        print("\n✓ Server stopped", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"✗ Server error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
