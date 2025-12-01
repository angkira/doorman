#!/bin/bash
# Install GStreamer dependencies for doorman camera backend

echo "Installing GStreamer and PipeWire support..."

sudo apt-get update
sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-pipewire \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    pipewire \
    wireplumber \
    libpipewire-0.3-dev \
    libspa-0.2-dev

echo "Testing GStreamer installation..."
gst-launch-1.0 --version

echo "Testing PipeWire camera access..."
# List available video sources
gst-device-monitor-1.0 Video

echo "Done! GStreamer backend is ready to use."
echo "Build with: cargo build --release --features camera-gstreamer"
