#!/bin/bash
set -e

# CPU-only mode (stable for iGPU with desktop compositor)

# Python virtual environment
VENV_PATH="$HOME/Home/doorman/.venv"
if [ -d "$VENV_PATH" ]; then
    export VIRTUAL_ENV="$VENV_PATH"
    export PATH="$VENV_PATH/bin:$PATH"
    echo "Using Python venv: $VENV_PATH"
else
    echo "Warning: venv not found at $VENV_PATH, using system python3"
fi

export ORT_LOG_LEVEL=3  # Suppress warnings

# Models directory
export MODELS_DIR="$HOME/.local/share/doorman/models"

echo "=== Doorman PyTorch Backend (CPU) ==="
echo "Models dir: $MODELS_DIR"
echo "Using CPU (stable with desktop compositor)"
echo ""

# Build with torch backend feature
echo "Building with backend-torch feature..."
cargo build --release --features backend-torch

echo ""
echo "Starting daemon with torch backend (CPU)..."

# Use CPU device in config
exec ./target/release/doormand --user --config tools/configs/doorman-torch-cpu.toml "$@"
