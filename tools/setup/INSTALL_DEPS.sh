#!/bin/bash
# Install build dependencies for doorman

echo "Installing build dependencies..."
sudo apt update
sudo apt install -y \
    build-essential \
    clang \
    libclang-dev \
    libpam0g-dev \
    pkg-config \
    libopencv-dev \
    opencv-data

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "Now you can build:"
echo "  cd /home/angkira/Home/doorman"
echo "  cargo build --release --features backend-tract"
echo ""
echo "Run tests:"
echo "  cargo test --features backend-tract"

