#!/bin/bash
# GPU Performance Check - verify if MIGraphX actually works and how fast

set -e

echo "=================================="
echo "GPU Performance Verification"
echo "=================================="
echo ""

# Activate venv
if [ ! -d ".venv" ]; then
    echo "❌ Error: .venv not found"
    exit 1
fi

source .venv/bin/activate

# Set ROCm env
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ORT_LOG_LEVEL=3  # Suppress warnings

MODELS_DIR="$HOME/.local/share/doorman/models"

echo "Step 1/3: Checking which Execution Provider loads..."
echo "---"

python3 << 'EOF'
import onnxruntime as ort
from pathlib import Path

ort.set_default_logger_severity(3)
models_dir = Path.home() / ".local/share/doorman/models"
providers = ['MIGraphXExecutionProvider', 'CPUExecutionProvider']

print("Requested providers:", providers)
print("Available providers:", ort.get_available_providers())
print("")
print("Loading blazeface.onnx (MIGraphX compiles on first load, may take 30+ sec)...")

sess = ort.InferenceSession(str(models_dir / "blazeface.onnx"), providers=providers)

print("")
print("✓ Model loaded!")
print(f"Active provider: {sess.get_providers()[0]}")
print(f"All providers in session: {sess.get_providers()}")
EOF

echo ""
echo "=================================="
echo "Step 2/3: Benchmarking MIGraphX..."
echo "---"

python3 << 'EOF'
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

ort.set_default_logger_severity(3)

models_dir = Path.home() / ".local/share/doorman/models"
providers = ['MIGraphXExecutionProvider', 'CPUExecutionProvider']

sess = ort.InferenceSession(str(models_dir / "blazeface.onnx"), providers=providers)
print(f"Provider: {sess.get_providers()[0]}")

# Prepare input
dummy_input = np.random.randn(1, 3, 240, 320).astype(np.float32)
input_name = sess.get_inputs()[0].name

# Warmup
print("Warming up (10 runs)...")
for _ in range(10):
    sess.run(None, {input_name: dummy_input})

# Benchmark
print("Benchmarking (100 runs)...")
start = time.perf_counter()
for _ in range(100):
    sess.run(None, {input_name: dummy_input})
elapsed = time.perf_counter() - start

avg_ms = elapsed / 100 * 1000
fps = 100 / elapsed

print("")
print(f"Results:")
print(f"  Average inference: {avg_ms:.2f} ms")
print(f"  Throughput: {fps:.1f} FPS")
print("")

if sess.get_providers()[0] == 'MIGraphXExecutionProvider':
    if fps > 500:
        print("✅ GPU is working! Performance looks good.")
    elif fps > 100:
        print("⚠️  GPU is active but slower than expected")
    else:
        print("❌ GPU is active but suspiciously slow (may be fallback)")
else:
    print("❌ WARNING: Running on CPU, not GPU!")
EOF

echo ""
echo "=================================="
echo "Step 3/3: CPU vs MIGraphX comparison..."
echo "---"

python3 << 'EOF'
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

ort.set_default_logger_severity(3)
models_dir = Path.home() / ".local/share/doorman/models"
model_path = str(models_dir / "blazeface.onnx")
dummy = np.random.randn(1, 3, 240, 320).astype(np.float32)

results = {}

for backend_name, providers in [("CPU", ['CPUExecutionProvider']),
                                 ("MIGraphX (GPU)", ['MIGraphXExecutionProvider', 'CPUExecutionProvider'])]:
    print(f"\nTesting: {backend_name}")
    print("-" * 40)

    sess = ort.InferenceSession(model_path, providers=providers)
    actual_provider = sess.get_providers()[0]
    print(f"Active provider: {actual_provider}")

    input_name = sess.get_inputs()[0].name

    # Warmup
    for _ in range(5):
        sess.run(None, {input_name: dummy})

    # Benchmark
    start = time.perf_counter()
    num_runs = 50
    for _ in range(num_runs):
        sess.run(None, {input_name: dummy})
    elapsed = time.perf_counter() - start

    fps = num_runs / elapsed
    avg_ms = elapsed / num_runs * 1000

    results[backend_name] = {'fps': fps, 'ms': avg_ms, 'provider': actual_provider}

    print(f"Results: {fps:.1f} FPS ({avg_ms:.2f} ms)")

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

for name, data in results.items():
    print(f"{name:20} {data['fps']:6.1f} FPS  ({data['ms']:5.2f} ms)  [{data['provider']}]")

cpu_fps = results['CPU']['fps']
gpu_fps = results['MIGraphX (GPU)']['fps']
speedup = gpu_fps / cpu_fps

print("")
print(f"Speedup: {speedup:.2f}x")
print("")

if results['MIGraphX (GPU)']['provider'] == 'CPUExecutionProvider':
    print("❌ PROBLEM: MIGraphX request fell back to CPU!")
    print("   Check:")
    print("   - ROCm installation")
    print("   - HSA_OVERRIDE_GFX_VERSION env var")
    print("   - onnxruntime-rocm package")
elif speedup < 1.5:
    print("❌ PROBLEM: GPU is not faster than CPU!")
    print("   Model might be too small to benefit from GPU.")
elif speedup < 5:
    print("⚠️  GPU works but speedup is modest ({:.1f}x)".format(speedup))
    print("   This is normal for small models.")
else:
    print(f"✅ SUCCESS: GPU is {speedup:.1f}x faster than CPU!")

print("")
print("Expected for Radeon 780M with MIGraphX:")
print("  BlazeFace (240x320): 500-1000 FPS")
print("  Liveness (96x96): 2000-3000 FPS")
print("  MobileFaceNet (112x112): 1000-2000 FPS")
print("")
EOF

echo ""
echo "=================================="
echo "Done!"
echo "=================================="
