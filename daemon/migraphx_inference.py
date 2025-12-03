#!/usr/bin/env python3
"""
MIGraphX inference server - runs models on AMD iGPU
"""
import sys
import json
import struct
import numpy as np

# Add ROCm Python path
sys.path.insert(0, '/opt/rocm/lib')

try:
    import migraphx
except ImportError:
    print(json.dumps({"error": "MIGraphX not found"}), flush=True)
    sys.exit(1)

class MIGraphXInference:
    def __init__(self):
        self.models = {}
        
    def load_model(self, name: str, path: str):
        """Load ONNX model and compile for ROCm"""
        try:
            model = migraphx.parse_onnx(path)
            model.compile(migraphx.get_target("gpu"))
            self.models[name] = model
            return {"status": "ok", "model": name}
        except Exception as e:
            return {"error": str(e)}
    
    def infer(self, model_name: str, input_data: dict):
        """Run inference"""
        try:
            model = self.models[model_name]
            # Convert input dict to migraphx parameters
            params = {}
            for key, value in input_data.items():
                arr = np.array(value["data"]).reshape(value["shape"]).astype(np.float32)
                params[key] = migraphx.argument(arr)
            
            result = model.run(params)
            
            # Convert output to list
            output = {}
            for i, res in enumerate(result):
                output[f"output_{i}"] = res.tolist()
            
            return {"status": "ok", "output": output}
        except Exception as e:
            return {"error": str(e)}

def main():
    """JSON RPC server over stdin/stdout"""
    server = MIGraphXInference()
    
    print(json.dumps({"status": "ready"}), flush=True)
    
    for line in sys.stdin:
        try:
            req = json.loads(line)
            cmd = req["cmd"]
            
            if cmd == "load":
                resp = server.load_model(req["name"], req["path"])
            elif cmd == "infer":
                resp = server.infer(req["model"], req["input"])
            elif cmd == "exit":
                break
            else:
                resp = {"error": f"Unknown command: {cmd}"}
            
            print(json.dumps(resp), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    main()
