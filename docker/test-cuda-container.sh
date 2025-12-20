#!/bin/bash
# Test CUDA Docker container with shared memory inference

set -e

echo "Starting CUDA Docker container test..."

# Create shared memory buffers (daemon will do this normally)
echo "Creating shared memory buffers..."

# Run container with GPU support, shared memory, and socket
docker run --rm --gpus all \
    --ipc=host \
    -v /tmp:/tmp \
    -v ~/.local/share/doorman/models:/app/models:ro \
    -e MODELS_DIR=/app/models \
    -e DEVICE=cuda \
    -e SHM_NAME_0=doorman_shm_test_0 \
    -e SHM_NAME_1=doorman_shm_test_1 \
    -e SOCKET_PATH=/tmp/doorman-inference-test.sock \
    doorman-cuda:latest python3 torch_inference_shm.py

echo "✓ Container test complete!"
