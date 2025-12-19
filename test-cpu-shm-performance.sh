#!/bin/bash
set -e

echo "=============================================="
echo "   CPU + Shared Memory Performance Test"
echo "=============================================="
echo ""

# Build daemon
echo "[1/3] Building daemon..."
echo "------------------------------------------------------------"
cargo build --release --features backend-torch-shm,camera-gstreamer
echo "✓ Daemon built"
echo ""

# Start daemon in background
echo "[2/3] Starting daemon with CPU + Shared Memory..."
echo "------------------------------------------------------------"
./target/release/doormand --user --preview --debug --config doorman-torch-shm.toml &
DAEMON_PID=$!
echo "✓ Daemon started (PID: $DAEMON_PID)"

# Wait for startup
echo "Waiting for daemon to initialize (15s)..."
sleep 15

# Check if daemon is still running
if ! kill -0 $DAEMON_PID 2>/dev/null; then
    echo "❌ Daemon crashed during startup"
    exit 1
fi

echo "✓ Daemon running"
echo ""

# Monitor performance
echo "[3/3] Monitoring performance (30s)..."
echo "------------------------------------------------------------"
echo "Watching daemon logs for FPS info..."
echo ""

# Show logs with FPS detection for 30 seconds
timeout 30 journalctl --user -u doormand -f --since "1 minute ago" | grep -E "(FPS|fps|Detection|Recognition)" || true

echo ""
echo "------------------------------------------------------------"
echo "Stopping daemon..."
kill $DAEMON_PID
wait $DAEMON_PID 2>/dev/null || true
echo "✓ Done"
