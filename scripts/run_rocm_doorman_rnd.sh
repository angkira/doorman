#!/usr/bin/env bash
set -euo pipefail

IMAGE="${DOORMAN_ROCM_IMAGE:-hars-training-rocm:latest}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${DOORMAN_DATASET_DIR:-/var/lib/doorman/dataset}"
LIVE_MODELS_DIR="${DOORMAN_LIVE_MODELS_DIR:-/var/lib/doorman/models}"
RND_ROOT="${DOORMAN_RND_ROOT:-$REPO_ROOT/target/doorman-rnd}"
OUT_SUBDIR="${DOORMAN_RND_OUT_SUBDIR:-$(date +%Y%m%d-%H%M%S)}"
ENCODER="${DOORMAN_RND_ENCODER:-inventory}"
PYTHON_BIN="${DOORMAN_RND_PYTHON:-python}"

if [ "$PYTHON_BIN" = "python" ]; then
  case "$ENCODER" in
    da3-base|depthanything3-base|depth-anything-3-base)
      PYTHON_BIN="/mnt/datasets/therock-venv/bin/python"
      ;;
  esac
fi

if [ ! -d "$DATASET_DIR" ]; then
  echo "Dataset dir not found: $DATASET_DIR" >&2
  echo "Enable collection, reinstall the current deb, restart doormand, then unlock once:" >&2
  echo "  sudo apt-get install --reinstall ./target/debian/dist/doorman_0.1.0_amd64.deb" >&2
  echo "  sudo scripts/enable_local_dataset_logging.sh" >&2
  exit 1
fi

case "$OUT_SUBDIR" in
  /*|*..*|"")
    echo "Unsafe DOORMAN_RND_OUT_SUBDIR: $OUT_SUBDIR" >&2
    exit 1
    ;;
esac

mkdir -p "$RND_ROOT/$OUT_SUBDIR"
LIVE_MODEL_MOUNT=()
if [ -d "$LIVE_MODELS_DIR" ]; then
  LIVE_MODEL_MOUNT=(-v "$LIVE_MODELS_DIR":/live-models:ro)
fi

exec docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --security-opt seccomp=unconfined \
  -e HF_HOME=/data/huggingface_cache \
  -e TRANSFORMERS_CACHE=/data/huggingface_cache \
  -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  -v "$REPO_ROOT":/workspace/doorman:ro \
  -v "$DATASET_DIR":/data/doorman-dataset:ro \
  -v "$RND_ROOT/$OUT_SUBDIR":/out:rw \
  "${LIVE_MODEL_MOUNT[@]}" \
  -v cortex_huggingface_cache:/data/huggingface_cache:rw \
  -v /mnt/datasets:/mnt/datasets:ro \
  -w /workspace/doorman \
  "$IMAGE" \
  "$PYTHON_BIN" /workspace/doorman/scripts/rnd_depth_fusion_experiment.py \
    --dataset-dir /data/doorman-dataset \
    --out-dir /out \
    --encoder "$ENCODER" \
    "$@"
