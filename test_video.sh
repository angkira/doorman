#!/bin/bash
# Quick test script for video file input

set -e

echo "🎬 Testing Doorman with Video File"
echo "======================================"
echo ""
echo "Video: 2025-11-26-115723.webm"
echo "Config: doorman-video-test.toml"
echo ""
echo "Starting daemon..."
echo ""

# Kill any existing daemon
pkill -9 doormand 2>/dev/null || true
sleep 1

# Start daemon with video file
DOORMAN_CONFIG=doorman-video-test.toml ./target/release/doormand --user --preview &
DAEMON_PID=$!

echo "Daemon started (PID: $DAEMON_PID)"
echo ""
echo "Waiting 5 seconds for initialization..."
sleep 5

echo ""
echo "To view preview, run in another terminal:"
echo "  doorman preview --debug"
echo ""
echo "Press Ctrl+C to stop daemon"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo 'Stopping daemon...'; kill $DAEMON_PID 2>/dev/null; exit 0" INT
wait $DAEMON_PID
