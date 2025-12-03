#!/bin/bash
# Verify GTT memory setup for AMD Radeon 780M

echo "=== GTT Memory Setup Verification ==="
echo ""

# Check kernel version (need 6.10+)
KERNEL_VERSION=$(uname -r | cut -d. -f1,2)
echo "1. Kernel Version: $(uname -r)"
if (( $(echo "$KERNEL_VERSION >= 6.10" | bc -l) )); then
    echo "   ✓ Kernel 6.10+ detected (GTT support available)"
else
    echo "   ⚠ Kernel < 6.10 (GTT support may be limited)"
fi
echo ""

# Check TTM configuration
echo "2. TTM Configuration:"
if [ -f /etc/modprobe.d/ttm.conf ]; then
    echo "   ✓ /etc/modprobe.d/ttm.conf exists"
    cat /etc/modprobe.d/ttm.conf | grep -v "^#"
else
    echo "   ✗ /etc/modprobe.d/ttm.conf NOT found"
    echo "   Run: ./tools/enable_gtt_memory.sh"
fi
echo ""

# Check current GTT memory
echo "3. Current GTT Memory:"
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showmeminfo all | grep -E "GTT|VRAM" | grep -E "Total|Used"
else
    echo "   ⚠ rocm-smi not found"
fi
echo ""

# Check GPU info
echo "4. GPU Information:"
if command -v rocminfo &> /dev/null; then
    rocminfo | grep -E "Marketing Name|gfx" | head -2
else
    echo "   ⚠ rocminfo not found"
fi
echo ""

# Check HSA override
echo "5. HSA Override (in run_torch.sh):"
if [ -f tools/run_torch.sh ]; then
    grep "HSA_OVERRIDE_GFX_VERSION" tools/run_torch.sh
    echo "   ✓ run_torch.sh configured"
else
    echo "   ✗ tools/run_torch.sh not found"
fi
echo ""

# Check ONNX Runtime
echo "6. ONNX Runtime with ROCm:"
if [ -d "$HOME/Home/doorman/.venv" ]; then
    source "$HOME/Home/doorman/.venv/bin/activate"
    python3 -c "import onnxruntime as ort; print('   ✓ ONNX Runtime version:', ort.__version__); print('   Available providers:', ort.get_available_providers())" 2>/dev/null || echo "   ✗ Failed to import onnxruntime"
else
    echo "   ⚠ Python venv not found"
fi
echo ""

# Summary
echo "=== Next Steps ==="
echo ""
if [ ! -f /etc/modprobe.d/ttm.conf ]; then
    echo "❌ GTT not configured yet. Run:"
    echo "   ./tools/enable_gtt_memory.sh"
    echo "   sudo reboot"
else
    echo "✓ GTT configuration exists."
    echo ""
    echo "If you just configured it, reboot now:"
    echo "   sudo reboot"
    echo ""
    echo "After reboot, verify GTT increased to ~48GB:"
    echo "   rocm-smi --showmeminfo all | grep GTT"
    echo ""
    echo "Then test the daemon:"
    echo "   ./tools/run_torch.sh --preview"
fi
echo ""
