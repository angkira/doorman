#!/bin/bash
# Test doorman with video file as input

VIDEO_FILE="${1:-2025-11-26-115723.webm}"

if [ ! -f "$VIDEO_FILE" ]; then
    echo "Error: Video file not found: $VIDEO_FILE"
    echo "Usage: $0 [video_file]"
    exit 1
fi

echo "Testing doorman with video file: $VIDEO_FILE"
echo "Building test binary..."

cargo build --release --bin test-video-file || {
    echo "Build failed"
    exit 1
}

echo "Starting test with video file..."
./target/release/test-video-file "$VIDEO_FILE"
