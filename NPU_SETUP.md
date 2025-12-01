# AMD Ryzen AI NPU Setup Guide

This guide explains how to enable AMD Ryzen AI NPU support in doorman using the VitisAI execution provider.

## Hardware Requirements

- **AMD Ryzen 7 8700G** (or other Ryzen AI processors with XDNA NPU)
- The NPU runs at 1.6GHz for AI workloads

## Software Requirements

- **Ubuntu 22.04/24.04**
- **Python 3.10**
- **AMD Ryzen AI Software 1.6.1** (latest as of Nov 2025)

## Installation Steps

### 1. Install System Dependencies

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv libboost-filesystem1.74.0
```

### 2. Install AMD NPU Drivers (XRT)

Download and install XRT packages from AMD:

```bash
# Install XRT base, base-dev, NPU, and AMDXDNA plugin packages
# These must be downloaded from AMD's Ryzen AI Early Access Lounge
# https://www.amd.com/en/products/processors/consumer/ryzen-ai.html

sudo dpkg -i xrt_202420.*.deb
sudo dpkg -i xrt-dev_202420.*.deb
sudo dpkg -i xrt_plugin.amdxdna_*.deb
```

### 3. Install Ryzen AI Software

```bash
# Download ryzen_ai-1.6.1.tgz from AMD Ryzen AI Early Access Lounge
tar -xzf ryzen_ai-1.6.1.tgz
cd ryzen_ai-1.6.1

# Run installation script
./install_ryzen_ai.sh
```

### 4. Install ONNX Runtime with VitisAI Support

The Ryzen AI package includes a custom ONNX Runtime with VitisAI execution provider:

```bash
# Activate the Ryzen AI environment
source ~/ryzen-ai-1.6.1/venv/bin/activate

# The installation script already installed onnxruntime with VitisAI support
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Should show: ['VitisAIExecutionProvider', 'CPUExecutionProvider']
```

### 5. Configure Doorman for NPU

Edit `~/.config/doorman/doorman.toml`:

```toml
[ml]
models_dir = "/home/YOUR_USERNAME/.local/share/doorman/models"
backend = "onnx"  # Use ONNX Runtime backend
device = "npu"    # Use NPU/VitisAI execution provider
cpu_threads = 4
```

### 6. Set Environment Variables

The Ryzen AI ONNX Runtime library needs to be in your library path:

```bash
# Add to ~/.bashrc or run before starting doorman
export LD_LIBRARY_PATH=~/ryzen-ai-1.6.1/venv/lib/python3.10/site-packages/onnxruntime/capi:$LD_LIBRARY_PATH
```

### 7. Rebuild Doorman with ORT Backend

```bash
cd ~/Home/doorman
cargo build --release --features backend-ort --bin doormand
```

### 8. Test NPU Acceleration

```bash
# Start daemon
~/bin/doormand --user --preview

# In another terminal, test with preview
doorman preview --console
```

## Verification

Check the daemon logs to verify NPU is being used:

```bash
journalctl --user -u doormand -f
```

Look for:
```
INFO doorman::ml::ort_backend: Using VitisAI execution provider (AMD Ryzen AI NPU)
```

## Performance Expectations

- **CPU (Tract)**: ~6-7 FPS
- **NPU (VitisAI)**: Expected 15-25 FPS (varies by model)
- **CUDA (NVIDIA)**: 20-30+ FPS

## Troubleshooting

### NPU Not Detected

```bash
# Check if XRT drivers are installed
dpkg -l | grep xrt

# Check NPU device
ls -la /dev/accel/

# Check XRT version
/opt/xilinx/xrt/bin/xbutil examine
```

### Falls Back to CPU

If you see "Using CPU execution provider" instead of VitisAI, check:

1. **Driver version compatibility**: VitisAI EP requires specific XRT driver versions
2. **Model quantization**: Some models may need quantization for NPU
3. **Library path**: Ensure `LD_LIBRARY_PATH` includes the Ryzen AI ONNX Runtime

### Model Quantization for NPU

NPU works best with quantized INT8 models. You may need to quantize your models:

```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    model_input='blazeface.onnx',
    model_output='blazeface_quant.onnx',
    weight_type=QuantType.QUInt8
)
```

## References

- [AMD Ryzen AI Software Documentation](https://ryzenai.docs.amd.com/en/latest/)
- [Linux Installation Guide](https://ryzenai.docs.amd.com/en/latest/linux.html)
- [VitisAI Execution Provider Docs](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
- [Model Quantization Guide](https://ryzenai.docs.amd.com/en/latest/modelrun.html)

## Current Status

✅ **Code implementation complete** - NPU support is already in the ORT backend
⚠️ **Drivers required** - You need to install AMD Ryzen AI software
🧪 **Experimental** - Linux NPU support is newer than Windows

To use NPU, simply:
1. Install AMD Ryzen AI drivers (steps above)
2. Set `device = "npu"` in your config
3. Use the Ryzen AI ONNX Runtime library

**Alternative**: For now, your **NVIDIA RTX 4060 with CUDA** will give you better performance and is more mature on Linux.
