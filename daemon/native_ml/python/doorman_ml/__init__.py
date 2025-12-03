"""
Doorman ML Native Extension

High-performance face recognition using native Rust + ONNX Runtime.
No IPC overhead, direct inference from Python.
"""

try:
    from doorman_ml_native import DoormanML, DetectionResult, LivenessResult
    __all__ = ['DoormanML', 'DetectionResult', 'LivenessResult']
except ImportError:
    # Not built yet
    pass
