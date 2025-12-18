#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================="
echo "   Testing Shared Memory Backend"
echo "=============================================="
echo

# Check models
MODELS_DIR="$ROOT_DIR/data/models"
if [ ! -f "$MODELS_DIR/blazeface.onnx" ]; then
    echo "❌ Models not found in $MODELS_DIR"
    echo "Run: doorman models download"
    exit 1
fi

echo "✓ Models found"
echo

# Build Rust daemon with shared memory backend
echo "[1/3] Building daemon with shared memory backend..."
echo "------------------------------------------------------------"
cd "$ROOT_DIR"
cargo build --release --features backend-torch-shm,camera-gstreamer
echo "✓ Daemon built"
echo

# Start daemon in test mode
echo "[2/3] Starting daemon with shared memory backend..."
echo "------------------------------------------------------------"
./target/release/doormand --config tools/configs/doorman-torch-shm.toml --user --preview &
DAEMON_PID=$!
echo "✓ Daemon started (PID: $DAEMON_PID)"
echo "Waiting for startup..."
sleep 5
echo

# Run preview
echo "[3/3] Starting preview (Press 'q' to quit)..."
echo "------------------------------------------------------------"
doorman preview

# Cleanup
echo
echo "Stopping daemon..."
kill $DAEMON_PID 2>/dev/null || true
wait $DAEMON_PID 2>/dev/null || true
echo "✓ Done"
