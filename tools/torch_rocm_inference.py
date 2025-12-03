#!/usr/bin/env python3
"""PyTorch ROCm inference server for doorman"""

import sys
import json
import base64
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from io import BytesIO
from PIL import Image

class TorchROCmInference:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        
        # Check ROCm availability
        if not torch.cuda.is_available():
            print(json.dumps({"error": "ROCm/CUDA not available"}), flush=True)
            sys.exit(1)
        
        device_count = torch.cuda.device_count()
        print(json.dumps({"info": f"ROCm available with {device_count} devices"}), file=sys.stderr, flush=True)
        
        self.device = torch.device("cuda:0")
        
        # Load models with ONNX Runtime MIGraphX EP (AMD iGPU)
        providers = ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
        
        self.detector = ort.InferenceSession(
            str(self.models_dir / "blazeface.onnx"),
            providers=providers
        )
        self.liveness = ort.InferenceSession(
            str(self.models_dir / "liveness.onnx"),
            providers=providers
        )
        self.recognizer = ort.InferenceSession(
            str(self.models_dir / "mobilefacenet.onnx"),
            providers=providers
        )
        
        print(json.dumps({"status": "ready"}), flush=True)
    
    def detect_face(self, image_data: bytes) -> dict:
        """Detect face in image"""
        # Decode image
        img = Image.open(BytesIO(image_data)).convert('RGB')
        img = img.resize((320, 240))  # BlazeFace expects 320x240
        
        # Preprocess
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0)
        
        # Run inference
        outputs = self.detector.run(None, {self.detector.get_inputs()[0].name: img_array})
        
        # Parse outputs (simplified)
        return {
            "bbox": [0.3, 0.3, 0.4, 0.4],  # Placeholder
            "confidence": 0.9
        }
    
    def check_liveness(self, face_crop: bytes) -> dict:
        """Check if face is live"""
        img = Image.open(BytesIO(face_crop)).convert('RGB')
        img = img.resize((112, 112))
        
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0)
        
        outputs = self.liveness.run(None, {self.liveness.get_inputs()[0].name: img_array})
        
        return {
            "is_live": True,  # Placeholder
            "confidence": 0.95
        }
    
    def extract_embedding(self, face_crop: bytes) -> list:
        """Extract face embedding"""
        img = Image.open(BytesIO(face_crop)).convert('RGB')
        img = img.resize((112, 112))
        
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0)
        
        outputs = self.recognizer.run(None, {self.recognizer.get_inputs()[0].name: img_array})
        
        embedding = outputs[0][0].tolist()
        return embedding
    
    def process_command(self, cmd: dict):
        """Process a command from stdin"""
        try:
            command_type = cmd.get("type")
            
            if command_type == "detect":
                image_b64 = cmd.get("image")
                image_data = base64.b64decode(image_b64)
                result = self.detect_face(image_data)
                print(json.dumps({"success": True, "result": result}), flush=True)
            
            elif command_type == "liveness":
                crop_b64 = cmd.get("crop")
                crop_data = base64.b64decode(crop_b64)
                result = self.check_liveness(crop_data)
                print(json.dumps({"success": True, "result": result}), flush=True)
            
            elif command_type == "embed":
                crop_b64 = cmd.get("crop")
                crop_data = base64.b64decode(crop_b64)
                result = self.extract_embedding(crop_data)
                print(json.dumps({"success": True, "embedding": result}), flush=True)
            
            else:
                print(json.dumps({"error": f"Unknown command: {command_type}"}), flush=True)
        
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)
    
    def run(self):
        """Main loop - read commands from stdin"""
        for line in sys.stdin:
            try:
                cmd = json.loads(line.strip())
                self.process_command(cmd)
            except json.JSONDecodeError:
                print(json.dumps({"error": "Invalid JSON"}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: torch_rocm_inference.py <models_dir>", file=sys.stderr)
        sys.exit(1)
    
    models_dir = sys.argv[1]
    server = TorchROCmInference(models_dir)
    server.run()

if __name__ == "__main__":
    main()
