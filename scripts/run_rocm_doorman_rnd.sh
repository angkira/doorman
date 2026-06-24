#!/usr/bin/env bash
set -euo pipefail

IMAGE="${DOORMAN_ROCM_IMAGE:-hars-training-rocm:latest}"
DATASET_DIR="${DOORMAN_DATASET_DIR:-/var/lib/doorman/dataset}"
LIVE_MODELS_DIR="${DOORMAN_LIVE_MODELS_DIR:-/var/lib/doorman/models}"
RND_ROOT="${DOORMAN_RND_ROOT:-/var/lib/doorman/rnd}"
OUT_SUBDIR="${DOORMAN_RND_OUT_SUBDIR:-$(date +%Y%m%d-%H%M%S)}"
ENCODER="${DOORMAN_RND_ENCODER:-inventory}"

if [ ! -d "$DATASET_DIR" ]; then
  echo "Dataset dir not found: $DATASET_DIR" >&2
  echo "Enable collection first: sudo scripts/enable_local_dataset_logging.sh" >&2
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
  -v /home/angkira/Home/doorman:/workspace/doorman:ro \
  -v "$DATASET_DIR":/data/doorman-dataset:ro \
  -v "$RND_ROOT/$OUT_SUBDIR":/out:rw \
  "${LIVE_MODEL_MOUNT[@]}" \
  -v cortex_huggingface_cache:/data/huggingface_cache:rw \
  -v /mnt/datasets:/mnt/datasets:rw \
  -w /workspace/doorman \
  "$IMAGE" \
  python /workspace/doorman/scripts/rnd_depth_fusion_experiment.py \
    --dataset-dir /data/doorman-dataset \
    --out-dir /out \
    --encoder "$ENCODER" \
    "$@"
