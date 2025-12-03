#!/bin/bash
set -e

echo "Building doorman_ml_native extension..."

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Build and install in development mode
echo "Building with maturin..."
maturin develop --release

echo ""
echo "✓ Build complete!"
echo ""
echo "Usage:"
echo "  python3 -c 'from doorman_ml import DoormanML; print(DoormanML)'"
echo ""
