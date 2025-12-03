"""Doorman ML Native - wrapper to setup library paths"""
import os
import sys

# Setup must happen BEFORE first import
try:
    import onnxruntime
    ort_dir = os.path.dirname(onnxruntime.__file__)
    capi_dir = os.path.join(ort_dir, 'capi')
    ort_lib = os.path.join(capi_dir, 'libonnxruntime.so')
    
    if os.path.exists(ort_lib):
        os.environ['ORT_DYLIB_PATH'] = ort_lib
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if capi_dir not in ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{capi_dir}:{ld_path}" if ld_path else capi_dir
except ImportError:
    pass

# Import native module
from .doorman_ml_native import *
