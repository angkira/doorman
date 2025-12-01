#!/bin/bash
# Test face detection on video file

VIDEO_FILE="2025-11-26-115723.webm"

if [ ! -f "$VIDEO_FILE" ]; then
    echo "Error: Video file not found: $VIDEO_FILE"
    exit 1
fi

echo "Testing face detection on video: $VIDEO_FILE"
echo ""

# Kill any existing daemon
pkill -f doormand || true
sleep 1

# Start daemon in background with video file source
echo "Starting daemon with video file input..."
RUST_LOG=info ./target/release/doormand --user --preview --video-file "$VIDEO_FILE" 2>&1 | tee /tmp/video_detection.log &
DAEMON_PID=$!

# Wait for daemon to start
sleep 3

# Start preview
echo "Starting preview..."
doorman preview --debug

# Cleanup
echo ""
echo "Stopping daemon..."
kill $DAEMON_PID 2>/dev/null || true
wait $DAEMON_PID 2>/dev/null || true

echo "Test complete! Check /tmp/video_detection.log for details"
