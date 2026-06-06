#!/usr/bin/env bash
# test_ubuntu.sh — build the Ubuntu test image and run the Linux validation
# harness (enroll -> genuine match -> impostor reject) inside it.
#
# Usage:
#   scripts/test_ubuntu.sh              # native arch (fast on Apple Silicon)
#   PLATFORM=linux/amd64 scripts/test_ubuntu.sh   # true x86_64 (emulated on ARM)
#
# Requires Docker. From the repo root.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE=doorman-test
PLATFORM_ARG=""
if [ -n "${PLATFORM:-}" ]; then
    PLATFORM_ARG="--platform=${PLATFORM}"
    echo "Building for platform: ${PLATFORM}"
fi

if [ ! -d data/models ] || [ -z "$(ls -A data/models/*.onnx 2>/dev/null)" ]; then
    echo "error: data/models/*.onnx not found. Run scripts/fetch_models.sh first." >&2
    exit 1
fi

echo "==> docker build -f Dockerfile.test -t ${IMAGE} ."
# shellcheck disable=SC2086
docker build ${PLATFORM_ARG} -f Dockerfile.test -t "${IMAGE}" .

echo "==> docker run --rm ${IMAGE}"
# shellcheck disable=SC2086
docker run --rm ${PLATFORM_ARG} "${IMAGE}"
