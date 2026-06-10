#!/usr/bin/env python3
"""Inspect BlazeFace ONNX model to understand output format"""

import onnx
import os
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "~/.local/share/doorman/models/blazeface.onnx"
model_path = os.path.expanduser(model_path)

print(f"Loading model: {model_path}")
model = onnx.load(model_path)

print("\n=== Model Inputs ===")
for input in model.graph.input:
    print(f"  {input.name}: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")

print("\n=== Model Outputs ===")
for output in model.graph.output:
    print(f"  {output.name}: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")

print("\n=== Metadata ===")
for prop in model.metadata_props:
    print(f"  {prop.key}: {prop.value}")
