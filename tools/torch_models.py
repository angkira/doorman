#!/usr/bin/env python3
"""
PyTorch/ONNX model loading and inference functions for Doorman.
Supports both CPU and CUDA/ROCm backends.
"""

import torch
import onnxruntime as ort
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, Dict, List

class DoormanModels:
    """Container for loaded ML models"""
    def __init__(self, detector, liveness, recognizer, device: str):
        self.detector = detector
        self.liveness = liveness
        self.recognizer = recognizer
        self.device = device

def load_models(models_dir: str, device: str = "cuda") -> DoormanModels:
    """
    Load all doorman models (BlazeFace, Liveness, MobileFaceNet).
    
    Args:
        models_dir: Path to directory containing .onnx models
        device: "cuda" or "cpu"
    
    Returns:
        DoormanModels object containing all loaded models
    """
    models_path = Path(models_dir).expanduser()
    
    # Setup ONNX Runtime execution providers
    if device == "cuda":
        if torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            print(f"WARNING: CUDA requested but not available, falling back to CPU", flush=True)
            providers = ['CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    print(f"Loading models from {models_path} with providers: {providers}", flush=True)
    
    # Load models
    detector = ort.InferenceSession(
        str(models_path / "blazeface.onnx"),
        providers=providers
    )
    liveness = ort.InferenceSession(
        str(models_path / "liveness.onnx"),
        providers=providers
    )
    recognizer = ort.InferenceSession(
        str(models_path / "mobilefacenet.onnx"),
        providers=providers
    )
    
    print(f"✓ Models loaded on {device}", flush=True)
    
    return DoormanModels(detector, liveness, recognizer, device)

def detect_faces(models: DoormanModels, image_rgb: np.ndarray, width: int, height: int) -> List[Dict]:
    """
    Detect faces in image using BlazeFace.
    
    Args:
        models: Loaded models
        image_rgb: RGB image as numpy array (HxWx3)
        width: Original image width
        height: Original image height
    
    Returns:
        List of detected faces with bbox and confidence
    """
    # Resize to BlazeFace input size (320x240)
    img = Image.fromarray(image_rgb).resize((320, 240))
    img_array = np.array(img).astype(np.float32)
    
    # Normalize to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    
    # CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, 0)
    
    # Run inference
    outputs = models.detector.run(None, {models.detector.get_inputs()[0].name: img_array})
    
    # Parse BlazeFace outputs (simplified - just return raw for now)
    # TODO: Proper decoding with anchors
    detections = []
    
    # Placeholder: return single face in center if confidence > threshold
    if len(outputs) >= 2:
        scores = outputs[0][0]  # [N,] or [1, N]
        boxes = outputs[1][0]   # [N, 4] or [1, N, 4]
        
        # Ensure proper shapes
        if scores.ndim > 1:
            scores = scores.flatten()
        if boxes.ndim == 3:
            boxes = boxes[0]  # [1, N, 4] -> [N, 4]
        
        # Ensure boxes is [N, 4]
        if boxes.shape[1] != 4:
            # Invalid format, return empty
            return detections
        
        for i in range(min(len(scores), len(boxes))):
            score = float(scores[i])
            if score > 0.5:  # Threshold
                box = boxes[i]
                detections.append({
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "confidence": score
                })
    
    return detections

def check_liveness(models: DoormanModels, face_crop_rgb: np.ndarray) -> Dict:
    """
    Check if face crop is live (not a photo/video).
    
    Args:
        models: Loaded models
        face_crop_rgb: Face crop as RGB numpy array
    
    Returns:
        Dict with is_live and confidence
    """
    # Resize to liveness model input (96x96)
    img = Image.fromarray(face_crop_rgb).resize((96, 96))
    img_array = np.array(img).astype(np.float32)
    
    # Normalize
    img_array = (img_array / 127.5) - 1.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, 0)
    
    # Run inference
    outputs = models.liveness.run(None, {models.liveness.get_inputs()[0].name: img_array})
    
    # Parse output (binary classification)
    score = outputs[0][0][0] if len(outputs[0].shape) > 1 else outputs[0][0]
    is_live = score > 0.5
    
    return {
        "is_live": bool(is_live),
        "confidence": float(score)
    }

def extract_embedding(models: DoormanModels, face_crop_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 512-dim face embedding using MobileFaceNet.
    
    Args:
        models: Loaded models
        face_crop_rgb: Face crop as RGB numpy array
    
    Returns:
        512-dim embedding as numpy array
    """
    # Resize to recognizer input (112x112)
    img = Image.fromarray(face_crop_rgb).resize((112, 112))
    img_array = np.array(img).astype(np.float32)
    
    # Normalize
    img_array = (img_array / 127.5) - 1.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, 0)
    
    # Run inference
    outputs = models.recognizer.run(None, {models.recognizer.get_inputs()[0].name: img_array})
    
    # Return embedding
    embedding = outputs[0][0]  # [512,]
    return embedding
