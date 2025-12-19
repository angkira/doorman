#!/usr/bin/env python3
"""
PyTorch inference server with Shared Memory optimization.
Zero-copy frame transfer via shared memory + Unix socket control.
"""

import os
import sys
import json
import socket
from pathlib import Path

# Import core dependencies BEFORE modifying sys.path
try:
    import numpy as np
    from posix_ipc import SharedMemory
except ImportError as e:
    print(json.dumps({"error": f"Import failed: {e}. Python: {sys.version}, Path: {sys.path}"}), file=sys.stderr, flush=True)
    sys.exit(1)

# Add project root to path (for torch_models)
project_root = Path(__file__).parent.parent.parent.parent
tools_path = project_root / "tools"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(tools_path))

try:
    from torch_models import load_models, detect_faces, check_liveness, extract_embedding
except ImportError as e:
    print(json.dumps({"error": f"Import failed: {e}. Python: {sys.version}, Path: {sys.path}"}), file=sys.stderr, flush=True)
    sys.exit(1)


class InferenceServer:
    def __init__(self, models_dir: str, device: str, shm_name_0: str, shm_name_1: str, socket_path: str):
        self.models_dir = models_dir
        self.device = device
        self.socket_path = socket_path
        
        # Open both shared memory buffers for double buffering
        import mmap
        self.shm_buffers = [
            SharedMemory(shm_name_0),
            SharedMemory(shm_name_1)
        ]
        # Create persistent mmaps for each buffer (reuse, don't recreate)
        self.mmaps = [
            mmap.mmap(self.shm_buffers[0].fd, 1920 * 1080 * 3),
            mmap.mmap(self.shm_buffers[1].fd, 1920 * 1080 * 3)
        ]
        print(f"Opened shared memory buffers: {shm_name_0}, {shm_name_1}", file=sys.stderr, flush=True)
        
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
        _ = detect_faces(self.models, dummy_image, 640, 480)
        
        # TODO: Re-enable when liveness/recognition models are available
        # Warmup liveness (112x112 face)
        # dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        # _ = check_liveness(self.models, dummy_face)
        
        # Warmup recognition
        # _ = extract_embedding(self.models, dummy_face)
    
    def read_frame_from_shm(self, width: int, height: int, buffer_index: int) -> np.ndarray:
        """Read frame from shared memory - COPY data to avoid holding references."""
        size = height * width * 3
        
        # Use persistent mmap (don't create/destroy on every call)
        mm = self.mmaps[buffer_index]
        
        # Read frame data and COPY to new array (don't hold reference to shm buffer)
        data = np.frombuffer(mm, dtype=np.uint8, count=size)
        frame = data.reshape((height, width, 3)).copy()  # COPY to release shm reference
        return frame
    
    def handle_detect(self, width: int, height: int, buffer_index: int) -> dict:
        """Handle face detection request."""
        frame = None
        try:
            # Read frame from shared memory (zero-copy!)
            frame = self.read_frame_from_shm(width, height, buffer_index)
            
            # Detect faces
            detections = detect_faces(self.models, frame, width, height)
            
            result = {
                "detections": [
                    {
                        "bbox": [float(x) for x in det['bbox']],
                        "confidence": float(det['confidence'])
                    }
                    for det in detections
                ]
            }
            
            # Explicitly release memory
            del frame
            import gc
            gc.collect()
            
            return result
        except Exception as e:
            if frame is not None:
                del frame
                import gc
                gc.collect()
            return {"error": str(e)}
    
    def handle_liveness(self, width: int, height: int, buffer_index: int) -> dict:
        """Handle liveness check request."""
        face = None
        try:
            # Read face from shared memory
            face = self.read_frame_from_shm(width, height, buffer_index)
            
            # Check liveness
            score = check_liveness(self.models, face)
            
            result = {"score": float(score)}
            
            # Explicitly release memory
            del face
            import gc
            gc.collect()
            
            return result
        except Exception as e:
            if face is not None:
                del face
                import gc
                gc.collect()
            return {"error": str(e)}
    
    def handle_embedding(self, width: int, height: int, buffer_index: int) -> dict:
        """Handle embedding extraction request."""
        face = None
        try:
            # Read face from shared memory
            face = self.read_frame_from_shm(width, height, buffer_index)
            
            # Extract embedding
            embedding = extract_embedding(self.models, face)
            
            result = {"embedding": embedding.tolist()}
            
            # Explicitly release memory
            del face, embedding
            import gc
            gc.collect()
            
            return result
        except Exception as e:
            if face is not None:
                del face
                import gc
                gc.collect()
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
                # Read command: "command width height buffer_index\n"
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
                    width, height, buffer_index = int(parts[1]), int(parts[2]), int(parts[3])
                    response = self.handle_detect(width, height, buffer_index)
                elif command == "liveness":
                    width, height, buffer_index = int(parts[1]), int(parts[2]), int(parts[3])
                    response = self.handle_liveness(width, height, buffer_index)
                elif command == "embedding":
                    width, height, buffer_index = int(parts[1]), int(parts[2]), int(parts[3])
                    response = self.handle_embedding(width, height, buffer_index)
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
            # Clean up mmaps and shared memory
            for mm in self.mmaps:
                mm.close()
            for shm in self.shm_buffers:
                shm.close_fd()
            if os.path.exists(self.socket_path):
                os.remove(self.socket_path)
            print("✓ Server shut down", file=sys.stderr, flush=True)


def main():
    models_dir = os.environ.get('DOORMAN_MODELS_DIR', '~/.local/share/doorman/models')
    device = os.environ.get('DOORMAN_DEVICE', 'cpu')
    shm_name_0 = os.environ.get('DOORMAN_SHM_NAME_0', 'doorman_shm_0')
    shm_name_1 = os.environ.get('DOORMAN_SHM_NAME_1', 'doorman_shm_1')
    socket_path = os.environ.get('DOORMAN_SOCKET_PATH', '/tmp/doorman-inference.sock')
    
    # Expand home directory
    models_dir = os.path.expanduser(models_dir)
    
    print(f"Starting inference server...", file=sys.stderr, flush=True)
    print(f"Models: {models_dir}", file=sys.stderr, flush=True)
    print(f"Device: {device}", file=sys.stderr, flush=True)
    print(f"Shared memory buffers: {shm_name_0}, {shm_name_1}", file=sys.stderr, flush=True)
    print(f"Socket: {socket_path}", file=sys.stderr, flush=True)
    
    try:
        server = InferenceServer(models_dir, device, shm_name_0, shm_name_1, socket_path)
        server.run()
    except KeyboardInterrupt:
        print("\n✓ Server stopped", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"✗ Server error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
