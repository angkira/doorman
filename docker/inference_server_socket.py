#!/usr/bin/env python3
"""
ONNX Runtime Inference Server with Unix Domain Socket
Zero-copy binary protocol for maximum performance
"""

import os
import sys
import socket
import struct
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import onnxruntime as ort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """ONNX Runtime inference with ROCm"""
    
    def __init__(self, models_dir: str, device: str = "cuda"):
        self.models_dir = Path(models_dir)
        self.device = device
        self.detector = None
        self.liveness = None
        self.embedder = None
        
        self._setup_onnxruntime()
        self._load_models()
        self._warmup()
    
    def _setup_onnxruntime(self):
        """Configure ONNX Runtime providers"""
        providers = ort.get_available_providers()
        logger.info(f"Available providers: {providers}")
        
        if "ROCMExecutionProvider" in providers:
            logger.info("✓ Using ROCMExecutionProvider")
            self.providers = [
                ('ROCMExecutionProvider', {'device_id': 0}),
                'CPUExecutionProvider'
            ]
        elif "CUDAExecutionProvider" in providers:
            logger.info("✓ Using CUDAExecutionProvider")
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            logger.warning("✗ GPU not available, using CPU")
            self.providers = ['CPUExecutionProvider']
    
    def _load_models(self):
        """Load ONNX models"""
        logger.info("Loading models...")
        
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.detector = ort.InferenceSession(
                str(self.models_dir / "blazeface.onnx"),
                sess_options=sess_opts,
                providers=self.providers
            )
            logger.info("✓ BlazeFace detector loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load detector: {e}")
        
        try:
            self.liveness = ort.InferenceSession(
                str(self.models_dir / "liveness.onnx"),
                sess_options=sess_opts,
                providers=self.providers
            )
            logger.info("✓ Liveness detector loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load liveness: {e}")
        
        try:
            self.embedder = ort.InferenceSession(
                str(self.models_dir / "mobilefacenet.onnx"),
                sess_options=sess_opts,
                providers=self.providers
            )
            logger.info("✓ MobileFaceNet embedder loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load embedder: {e}")
    
    def _warmup(self):
        """Warmup models with dummy data"""
        logger.info("Warming up models...")
        
        if self.detector:
            dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
            self.detector.run(None, {'input': dummy})
        
        if self.liveness:
            dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
            self.liveness.run(None, {'input': dummy})
        
        if self.embedder:
            dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)
            self.embedder.run(None, {'input': dummy})
        
        logger.info("✓ Models warmed up!")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in frame"""
        if self.detector is None:
            return []
        
        # Preprocess (simplified - add proper preprocessing)
        input_tensor = self._preprocess_detection(frame)
        
        # Run inference
        outputs = self.detector.run(None, {'input': input_tensor})
        
        # Postprocess
        detections = self._postprocess_detection(outputs, frame.shape)
        return detections
    
    def check_liveness(self, face: np.ndarray) -> Tuple[bool, float]:
        """Check if face is live"""
        if self.liveness is None:
            return True, 1.0
        
        # Preprocess
        input_tensor = self._preprocess_liveness(face)
        
        # Run inference
        outputs = self.liveness.run(None, {'input': input_tensor})
        
        # Postprocess
        score = float(outputs[0][0][0])
        is_live = score > 0.5
        return is_live, score
    
    def extract_embedding(self, face: np.ndarray) -> np.ndarray:
        """Extract face embedding"""
        if self.embedder is None:
            return np.zeros(128, dtype=np.float32)
        
        # Preprocess
        input_tensor = self._preprocess_embedding(face)
        
        # Run inference
        outputs = self.embedder.run(None, {'input': input_tensor})
        
        # Return embedding
        return outputs[0][0]
    
    def _preprocess_detection(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for detection"""
        # TODO: Add proper BlazeFace preprocessing
        # For now, simple resize and normalize
        import cv2
        resized = cv2.resize(frame, (128, 128))
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        return batched.astype(np.float32)
    
    def _preprocess_liveness(self, face: np.ndarray) -> np.ndarray:
        """Preprocess face for liveness check"""
        import cv2
        resized = cv2.resize(face, (128, 128))
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        return batched.astype(np.float32)
    
    def _preprocess_embedding(self, face: np.ndarray) -> np.ndarray:
        """Preprocess face for embedding extraction"""
        import cv2
        resized = cv2.resize(face, (112, 112))
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        return batched.astype(np.float32)
    
    def _postprocess_detection(self, outputs, frame_shape) -> List[Dict]:
        """Postprocess detection outputs"""
        # TODO: Add proper BlazeFace postprocessing
        # For now, return empty list
        return []


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket"""
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data.extend(chunk)
    return bytes(data)


def recv_frame(sock: socket.socket) -> Optional[np.ndarray]:
    """Receive frame from Unix socket using binary protocol
    
    Protocol:
        [width:u32][height:u32][channels:u32][data:bytes]
    """
    try:
        # Read header
        header = recv_exact(sock, 12)
        width, height, channels = struct.unpack('III', header)
        
        # Read frame data
        frame_size = width * height * channels
        frame_data = recv_exact(sock, frame_size)
        
        # Convert to numpy array
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, channels))
        return frame
    except Exception as e:
        logger.error(f"Error receiving frame: {e}")
        return None


def send_json_response(sock: socket.socket, response: Dict):
    """Send JSON response through socket
    
    Protocol:
        [type:u8=1][len:u32][json_data:bytes]
    """
    try:
        data = json.dumps(response).encode('utf-8')
        msg = struct.pack('BI', 1, len(data)) + data
        sock.sendall(msg)
    except Exception as e:
        logger.error(f"Error sending response: {e}")


def send_binary_response(sock: socket.socket, data: bytes):
    """Send binary response through socket
    
    Protocol:
        [type:u8=2][len:u32][binary_data:bytes]
    """
    try:
        msg = struct.pack('BI', 2, len(data)) + data
        sock.sendall(msg)
    except Exception as e:
        logger.error(f"Error sending response: {e}")


def handle_client(engine: InferenceEngine, client_sock: socket.socket):
    """Handle inference requests from a client
    
    Request types:
        0 - Ping (health check)
        1 - Detect faces
        2 - Check liveness
        3 - Extract embedding
    """
    logger.info("Client connected")
    
    try:
        while True:
            # Receive request type
            req_type_data = client_sock.recv(1)
            if not req_type_data:
                break
            
            req_type = struct.unpack('B', req_type_data)[0]
            
            if req_type == 0:  # Ping
                send_json_response(client_sock, {"status": "ok"})
            
            elif req_type == 1:  # Detect faces
                frame = recv_frame(client_sock)
                if frame is None:
                    break
                
                detections = engine.detect_faces(frame)
                send_json_response(client_sock, {"detections": detections})
            
            elif req_type == 2:  # Check liveness
                frame = recv_frame(client_sock)
                if frame is None:
                    break
                
                is_live, score = engine.check_liveness(frame)
                send_json_response(client_sock, {
                    "is_live": is_live,
                    "score": float(score)
                })
            
            elif req_type == 3:  # Extract embedding
                frame = recv_frame(client_sock)
                if frame is None:
                    break
                
                embedding = engine.extract_embedding(frame)
                # Send as binary
                send_binary_response(client_sock, embedding.tobytes())
            
            else:
                logger.warning(f"Unknown request type: {req_type}")
                break
    
    except Exception as e:
        logger.error(f"Error handling client: {e}")
    finally:
        client_sock.close()
        logger.info("Client disconnected")


def run_server(engine: InferenceEngine, socket_path: str):
    """Run Unix Domain Socket server"""
    # Remove old socket if exists
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    
    # Create socket
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(5)
    
    # Set permissions
    os.chmod(socket_path, 0o666)
    
    logger.info("=" * 60)
    logger.info("ONNX Runtime Inference Server Ready!")
    logger.info(f"Socket: {socket_path}")
    logger.info(f"Device: {engine.device}")
    logger.info(f"Providers: {engine.providers}")
    logger.info("=" * 60)
    
    try:
        while True:
            client_sock, _ = server.accept()
            # Handle in same thread (simple for now)
            handle_client(engine, client_sock)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


if __name__ == "__main__":
    # Get configuration
    models_dir = os.getenv("MODELS_DIR", "/app/models")
    device = os.getenv("DEVICE", "cuda")
    socket_path = os.getenv("DOORMAN_SOCKET", "/tmp/doorman-ml.sock")
    
    # Initialize and run
    try:
        engine = InferenceEngine(models_dir, device)
        run_server(engine, socket_path)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
