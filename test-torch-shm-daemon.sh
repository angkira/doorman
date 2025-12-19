#!/bin/bash
set -e

echo "=============================================="
echo "   Testing Torch Shared Memory Backend"
echo "=============================================="
echo ""

# Ensure torch dependencies are installed
echo "Installing Python dependencies..."
uv sync --group torch --quiet
echo "✓ Dependencies installed"
echo ""

# Check models (use doorman CLI to verify)
if ! uv run doorman models list | grep -q "3/3 installed"; then
    echo "❌ Not all models installed"
    echo "Please download models first:"
    echo "  uv run doorman models download --force"
    exit 1
fi
echo "✓ All models found"

echo "[1/3] Building daemon..."
echo "------------------------------------------------------------"
cargo build --release --features backend-torch-shm,camera-gstreamer
echo "✓ Daemon built"
echo ""

echo "[2/3] Starting daemon with shared memory backend..."
echo "------------------------------------------------------------"
DAEMON_PID=""
cleanup() {
    if [ -n "$DAEMON_PID" ]; then
        echo "Stopping daemon..."
        kill $DAEMON_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

./target/release/doormand --user --preview --debug \
    --config tools/configs/doorman-torch-shm.toml &
DAEMON_PID=$!

echo "✓ Daemon started (PID: $DAEMON_PID)"
echo "Waiting for startup..."
sleep 5

echo ""
echo "[3/3] Starting preview (Press 'q' to quit)..."
echo "------------------------------------------------------------"
uv run doorman preview

echo "✓ Done"
