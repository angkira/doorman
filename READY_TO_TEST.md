# Ready to Test - GTT Memory Solution

## Current Status ✓

**System Configuration:**
- ✅ Kernel 6.17.0 (GTT support available)
- ✅ AMD Radeon 780M iGPU (gfx1100)
- ✅ ROCm 7.0.2 installed
- ✅ ONNX Runtime 1.23.0 with MIGraphX
- ✅ 32GB GTT available (currently: 2.8GB used)

**Code Fixes Completed:**
- ✅ BlazeFace decoder implemented (240x320 input)
- ✅ Letterboxing with coordinate transformation
- ✅ Model input dimensions corrected
- ✅ HSA_OVERRIDE_GFX_VERSION=11.0.1 configured

**Current Problem:**
- ❌ VRAM: 2GB (1.9GB used by desktop compositor = 89%)
- ⚠️ MIGraphX tries to allocate in VRAM → Out of Memory
- ⚠️ GTT 32GB exists but MIGraphX not using it

## Solution: Apply GTT Configuration

### Step 1: Enable GTT Memory (requires sudo + reboot)

```bash
cd ~/Home/doorman

# Configure kernel TTM module for 48GB GTT
./tools/enable_gtt_memory.sh

# Reboot to apply changes
sudo reboot
```

**What this does:**
- Creates `/etc/modprobe.d/ttm.conf` with `pages_limit=12582912` (48GB)
- Updates initramfs to load TTM config on boot
- Allows ROCm/MIGraphX to use system RAM for GPU workloads

### Step 2: Verify GTT After Reboot

```bash
# Check GTT increased to ~48GB
rocm-smi --showmeminfo all | grep GTT

# Expected output:
# GTT Total Memory: ~48GB (instead of 32GB)

# Run full verification
./tools/verify_gtt_setup.sh
```

### Step 3: Test Face Detection

```bash
# Test with GPU backend
./tools/run_torch.sh --preview

# Watch for in logs:
# [PyTorch] Model on device: MIGraphXExecutionProvider
# Detection FPS: 50-120 (not 0.5!)
# Bounding boxes: real coordinates (not {x: 100, y: 100})
```

**Expected Results:**
- ✅ Models load without OOM errors
- ✅ Detection FPS: 50-120 (GPU-accelerated)
- ✅ Real bounding boxes from BlazeFace decoder
- ✅ Preview shows faces with green rectangles

### Step 4: Run Benchmark

```bash
# Measure real performance
./tools/benchmark_python_backend.sh

# Expected:
# Face Detection: 50-120 FPS
# Full Pipeline: 20-50 FPS
```

## Alternative: CPU Backend (no reboot needed)

If you want to test the BlazeFace decoder fix **now** without rebooting:

```bash
# Use CPU backend (slower but stable)
./tools/run_torch_cpu.sh --preview

# Expected:
# Detection: 10-20 FPS
# Real bounding boxes: ✓
# No GPU memory errors: ✓
```

## Files Reference

**Scripts:**
- `tools/enable_gtt_memory.sh` - Configure GTT (run once, needs reboot)
- `tools/verify_gtt_setup.sh` - Check system status
- `tools/run_torch.sh` - Launch daemon with GPU backend
- `tools/run_torch_cpu.sh` - Launch daemon with CPU backend
- `tools/benchmark_python_backend.sh` - Measure performance

**Configs:**
- `tools/configs/doorman-torch.toml` - GPU backend config
- `tools/configs/doorman-torch-cpu.toml` - CPU backend config

**Documentation:**
- `GTT_MEMORY_FIX.md` - Full GTT solution explanation
- `GPU_MEMORY_ISSUE.md` - iGPU VRAM limitations analysis
- `FIXES_SUMMARY.md` - All code fixes applied

## Troubleshooting

### If still OOM after GTT config:

1. **Check GTT actually increased:**
   ```bash
   rocm-smi --showmeminfo all | grep "GTT Total"
   # Should show ~48GB, not 32GB
   ```

2. **Try FP16 (half precision):**
   Edit `daemon/python/torch_inference.py`:
   ```python
   provider_options = {
       'device_id': 0,
       'migraphx_fp16_enable': '1',  # Use half precision
   }
   ```

3. **Fallback to CPU:**
   ```bash
   ./tools/run_torch_cpu.sh --preview
   ```

## Next Steps After Testing

Once GPU backend works:

1. **E2E Testing:**
   ```bash
   # Enroll a face
   doorman enroll

   # Test authentication
   doorman status
   ```

2. **Integration Testing:**
   ```bash
   # Run enrollment test
   cargo test --test enrollment_e2e_test
   ```

3. **Production Deployment:**
   - Configure as systemd service
   - Set up lock integration
   - Add logging/monitoring

## Sources

- [Ollama AMD iGPU GTT fix](https://github.com/ollama/ollama/pull/6282)
- [ROCm Radeon Prerequisites](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/prerequisites.html)
- [Linux Kernel 6.10+ TTM changes](https://github.com/ollama/ollama/issues/5471)
