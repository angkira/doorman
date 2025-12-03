#!/usr/bin/env python3
"""ONNX Runtime Inference Server with ROCm"""
import os, json, base64, logging
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

detector = liveness = embedder = None

def init_models(models_dir="/app/models"):
    global detector, liveness, embedder
    logger.info("Initializing ONNX Runtime with ROCm...")
    providers = [('ROCMExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    
    detector = ort.InferenceSession(str(Path(models_dir) / "blazeface.onnx"), providers=providers)
    liveness = ort.InferenceSession(str(Path(models_dir) / "liveness.onnx"), providers=providers)
    embedder = ort.InferenceSession(str(Path(models_dir) / "mobilefacenet.onnx"), providers=providers)
    logger.info(f"✓ Models loaded on: {detector.get_providers()}")
    
    # Warmup
    dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
    detector.run(None, {'input': dummy})
    liveness.run(None, {'input': dummy})
    embedder.run(None, {'input': dummy})
    logger.info("✓ Warmed up!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'providers': ort.get_available_providers()})

@app.route('/detect', methods=['POST'])
def detect():
    # Simplified - add proper preprocessing
    return jsonify({'boxes': [], 'scores': []})

if __name__ == '__main__':
    init_models(os.getenv('MODELS_DIR', '/app/models'))
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), threaded=True)
