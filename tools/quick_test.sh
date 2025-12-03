#!/bin/bash
# Quick test script - automatically chooses best backend

set -e

echo "=== Doorman Quick Test ==="
echo ""

# Check if GTT is configured
if [ -f /etc/modprobe.d/ttm.conf ]; then
    echo "✓ GTT configured"
    GTT_CONFIGURED=true
else
    echo "⚠ GTT not configured yet"
    GTT_CONFIGURED=false
fi

# Check current VRAM usage
VRAM_USED=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Used Memory" | awk -F': ' '{print $NF}' | tr -d ' ' || echo "0")
VRAM_TOTAL=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Memory (B)" | head -1 | awk -F': ' '{print $NF}' | tr -d ' ' || echo "0")

if [ "$VRAM_TOTAL" != "0" ]; then
    VRAM_PERCENT=$(awk "BEGIN {printf \"%.0f\", ($VRAM_USED/$VRAM_TOTAL)*100}")
    echo "VRAM: ${VRAM_USED}B / ${VRAM_TOTAL}B (${VRAM_PERCENT}% used)"
else
    VRAM_PERCENT=0
fi

echo ""

# Decide which backend to use
if [ "$GTT_CONFIGURED" = false ]; then
    echo "❌ GTT not configured - using CPU backend"
    echo ""
    echo "To enable GPU backend:"
    echo "  1. ./tools/enable_gtt_memory.sh"
    echo "  2. sudo reboot"
    echo "  3. Re-run this script"
    echo ""
    echo "Starting with CPU backend in 3 seconds..."
    sleep 3
    exec ./tools/run_torch_cpu.sh "$@"
elif [ "$VRAM_PERCENT" -gt 85 ]; then
    echo "⚠ VRAM usage high (${VRAM_PERCENT}%) - using CPU backend"
    echo ""
    echo "Desktop compositor is using most VRAM."
    echo "GPU backend may fail with OOM."
    echo ""
    echo "Options:"
    echo "  1. Continue with CPU backend (safe, slower)"
    echo "  2. Try GPU backend anyway (may OOM)"
    echo "  3. Close heavy apps and retry"
    echo ""
    read -p "Choice [1/2/3]: " choice
    case $choice in
        2)
            echo "Trying GPU backend..."
            exec ./tools/run_torch.sh "$@"
            ;;
        3)
            echo "Close apps and re-run: ./tools/quick_test.sh"
            exit 0
            ;;
        *)
            echo "Using CPU backend..."
            exec ./tools/run_torch_cpu.sh "$@"
            ;;
    esac
else
    echo "✓ VRAM usage OK (${VRAM_PERCENT}%) - using GPU backend"
    echo ""
    echo "Starting with GPU backend..."
    exec ./tools/run_torch.sh "$@"
fi
