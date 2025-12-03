#!/usr/bin/env python3
"""
PyTorch inference backend for Doorman
Runs models on ROCm iGPU via ONNX Runtime + ROCm
"""

import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io
import base64

try:
    import onnxruntime as ort
except ImportError:
    print("[PyTorch] ERROR: onnxruntime not installed. Run: uv pip install onnxruntime-rocm", file=sys.stderr)
    sys.exit(1)

class TorchInferenceBackend:
    def __init__(self, models_dir: str, device: str = "cuda"):
        self.models_dir = Path(models_dir)

        # Setup ONNX Runtime with ROCm/MIGraphX
        # NOTE: For iGPU with desktop compositor, CPU may be more stable due to VRAM constraints
        if device == "cuda":
            providers = ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        print(f"[PyTorch] Initializing ONNX Runtime with providers: {providers}", file=sys.stderr)
        print(f"[PyTorch] Available providers: {ort.get_available_providers()}", file=sys.stderr)
        
        # Load ONNX models
        self.face_detector = self._load_onnx_model("blazeface.onnx", providers)
        self.liveness_detector = self._load_onnx_model("liveness.onnx", providers)
        self.face_recognizer = self._load_onnx_model("mobilefacenet.onnx", providers)
        
        print("[PyTorch] All models loaded successfully", file=sys.stderr)
    
    def _load_onnx_model(self, filename: str, providers):
        """Load ONNX model via ONNX Runtime"""
        model_path = self.models_dir / filename
        print(f"[PyTorch] Loading {filename}...", file=sys.stderr)
        
        session = ort.InferenceSession(
            str(model_path),
            providers=providers
        )
        
        print(f"[PyTorch]   Model on device: {session.get_providers()[0]}", file=sys.stderr)
        return session
    
    def detect_faces(self, image_data: bytes, width: int, height: int) -> dict:
        """Detect faces in image"""
        # Decode image
        orig_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        orig_width, orig_height = orig_image.size

        # Resize to BlazeFace input size (240x320) with letterboxing to preserve aspect ratio
        target_width = 320
        target_height = 240
        scale = min(target_width / orig_width, target_height / orig_height)
        resized_w = int(orig_width * scale)
        resized_h = int(orig_height * scale)

        # Resize image
        resized_image = orig_image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

        # Create black canvas and paste resized image (letterboxing)
        letterboxed = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        offset_x = (target_width - resized_w) // 2
        offset_y = (target_height - resized_h) // 2
        letterboxed.paste(resized_image, (offset_x, offset_y))

        # Preprocess
        img_array = np.array(letterboxed).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

        # Inference
        outputs = self.face_detector.run(None, {'input': img_array})

        # Decode BlazeFace output to bboxes
        # BlazeFace outputs (from PINTO Model Zoo):
        # - scores: [1, N, 2] - class probabilities (background, face)
        # - boxes: [1, N, 4] - bounding boxes [top_y, top_x, bot_y, bot_x] normalized [0,1]
        scores_tensor = outputs[0]  # Shape: [1, N, 2]
        boxes_tensor = outputs[1]   # Shape: [1, N, 4]

        # Reshape to [N, 2] and [N, 4]
        scores = scores_tensor[0]  # [N, 2]
        boxes = boxes_tensor[0]    # [N, 4]

        # Find best face detection
        confidence_threshold = 0.5
        detections = []

        for i in range(len(scores)):
            face_score = scores[i][1]  # Index 1 is face class

            if face_score > confidence_threshold:
                # Extract bbox in format [top_y, top_x, bot_y, bot_x] normalized to letterboxed image
                top_y = boxes[i][0]
                top_x = boxes[i][1]
                bot_y = boxes[i][2]
                bot_x = boxes[i][3]

                # Convert from normalized [0,1] to letterboxed pixel coordinates
                x_letterbox = top_x * target_width
                y_letterbox = top_y * target_height
                x2_letterbox = bot_x * target_width
                y2_letterbox = bot_y * target_height

                # Remove letterbox offsets to get coordinates in resized image
                x_resized = x_letterbox - offset_x
                y_resized = y_letterbox - offset_y
                x2_resized = x2_letterbox - offset_x
                y2_resized = y2_letterbox - offset_y

                # Scale back to original image dimensions (in pixels)
                x_orig = (x_resized / resized_w) * orig_width
                y_orig = (y_resized / resized_h) * orig_height
                x2_orig = (x2_resized / resized_w) * orig_width
                y2_orig = (y2_resized / resized_h) * orig_height

                # Convert to (x, y, width, height) in pixels
                x = float(max(0, x_orig))
                y = float(max(0, y_orig))
                w = float(max(1, abs(x2_orig - x_orig)))
                h = float(max(1, abs(y2_orig - y_orig)))

                detections.append({
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "confidence": float(face_score)
                })

        # Sort by confidence and return top detection
        if detections:
            detections.sort(key=lambda d: d["confidence"], reverse=True)
            return {"detections": [detections[0]]}
        else:
            return {"detections": []}
    
    def check_liveness(self, face_crop: bytes) -> dict:
        """Check if face is live"""
        image = Image.open(io.BytesIO(face_crop)).convert('RGB')
        image = image.resize((96, 96))  # Liveness input size (model expects 96x96)
        
        # Preprocess
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Inference
        outputs = self.liveness_detector.run(None, {'input': img_array})
        scores = outputs[0][0]
        
        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores))
        scores = exp_scores / exp_scores.sum()
        
        is_live = scores[2] > 0.5  # Class 2 is "live"
        
        return {
            "is_live": bool(is_live),
            "confidence": float(scores[2])
        }
    
    def extract_embedding(self, face_crop: bytes) -> dict:
        """Extract face embedding"""
        image = Image.open(io.BytesIO(face_crop)).convert('RGB')
        image = image.resize((112, 112))  # MobileFaceNet input size
        
        # Preprocess
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Inference
        outputs = self.face_recognizer.run(None, {'input': img_array})
        embedding = outputs[0][0]
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return {
            "embedding": embedding.tolist()
        }

def main():
    """JSON-RPC interface for Rust"""
    if len(sys.argv) < 2:
        print("Usage: torch_inference.py <models_dir>", file=sys.stderr)
        sys.exit(1)
    
    models_dir = sys.argv[1]
    backend = TorchInferenceBackend(models_dir, device="cuda")
    
    print("[PyTorch] Ready for requests", file=sys.stderr)
    
    # Read JSON-RPC requests from stdin
    for line in sys.stdin:
        try:
            request = json.loads(line)
            method = request["method"]
            params = request.get("params", {})
            
            if method == "detect_faces":
                image_data = base64.b64decode(params["image_data"])
                result = backend.detect_faces(
                    image_data,
                    params["width"],
                    params["height"]
                )
            elif method == "check_liveness":
                face_crop = base64.b64decode(params["face_crop"])
                result = backend.check_liveness(face_crop)
            elif method == "extract_embedding":
                face_crop = base64.b64decode(params["face_crop"])
                result = backend.extract_embedding(face_crop)
            else:
                result = {"error": f"Unknown method: {method}"}
            
            response = {
                "id": request.get("id"),
                "result": result
            }
            print(json.dumps(response), flush=True)
            
        except Exception as e:
            error_response = {
                "id": request.get("id") if "request" in locals() else None,
                "error": str(e)
            }
            print(json.dumps(error_response), flush=True)
            print(f"[PyTorch] Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
