#!/usr/bin/env bash
#
# fetch_models.sh — provision the ONNX models doorman needs at runtime.
#
# Models are written to BOTH:
#   - data/models/                       (repo copy)
#   - $XDG_DATA_HOME/doorman/models/      (runtime --user location; defaults to ~/.local/share)
#
# Downloaded models are PINNED to a specific upstream revision and verified by
# SHA-256. Locally-exported models (EdgeFace, MiniFASNet) are committed under
# data/models/ and verified by SHA-256 too, so the build is reproducible.
#
# Models:
#   - YuNet face detector   (OpenCV Zoo, 2023mar) — face_detection_yunet_2023mar.onnx (MIT, downloaded)
#   - EdgeFace-S recognizer (Idiap, gamma=0.5)    — edgeface_s.onnx (CC-BY-NC-SA 4.0, NON-COMMERCIAL, ~15 MB, exported)
#   - MiniFASNetV2-SE liveness (facenox)          — minifasnet_v2se.onnx (Apache-2.0, ~1.82 MB, downloaded)
#
# EXPORTED MODELS (EdgeFace) are produced once from PyTorch by:
#       scripts/export_edgeface.py     (otroshi/edgeface checkpoint -> ONNX)
#   in a torch venv (see README). The exported .onnx files are committed to
#   data/models/. This script VERIFIES them by SHA-256 and mirrors them to the
#   runtime dir; if one is missing and /tmp/edgeface-venv exists, it attempts
#   the export automatically.
#
# NOTE: EdgeFace-S weights are CC-BY-NC-SA 4.0 (non-commercial). For commercial
# use, swap in AuraFace-v1 (fal, native ONNX, commercial-OK).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_MODELS="${REPO_ROOT}/data/models"
RUNTIME_MODELS="${XDG_DATA_HOME:-${HOME}/.local/share}/doorman/models"
VENV_PY="${EDGEFACE_VENV:-/tmp/edgeface-venv}/bin/python"

mkdir -p "${DATA_MODELS}" "${RUNTIME_MODELS}"

# ---------------------------------------------------------------------------
# YuNet detector (downloaded). FIXED 640x640 input, 12 outputs
# (cls_/obj_/bbox_/kps_ at strides 8/16/32). Pinned to an opencv_zoo commit.
# ---------------------------------------------------------------------------
YUNET_NAME="face_detection_yunet_2023mar.onnx"
YUNET_URL="https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_SHA256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

# ---------------------------------------------------------------------------
# EdgeFace-S recognizer (exported, committed). Idiap. Weights CC-BY-NC-SA 4.0
# (NON-COMMERCIAL); "BSD-3" refers to the bob framework code, not the weights.
#   - Input  `input`:     [1, 3, 112, 112], RGB, normalized (x - 127.5)/127.5 -> [-1,1].
#   - Output `embedding`: [1, 512] (the daemon L2-normalizes it).
# Faces aligned to the canonical 5-point 112x112 template before embedding.
# Reproduce: scripts/export_edgeface.py (otroshi/edgeface edgeface_s_gamma_05.pt).
# ---------------------------------------------------------------------------
RECOGNIZER_NAME="edgeface_s.onnx"
RECOGNIZER_SHA256="d1fd41ae2037715a86378a80692e52138195fcf1309b79e2c469e6467c7113d0"

# ---------------------------------------------------------------------------
# MiniFASNetV2-SE liveness (downloaded). facenox/face-antispoof-onnx, Apache-2.0.
# Single model, fed a 128x128 RGB crop (face bbox -> square max(w,h)*1.5,
# reflect-101 padded), normalized /255 -> [0,1], NCHW.
# Output: [1,2] logits — index 0 == real, index 1 == spoof. Decision:
#   is_real = (real_logit - spoof_logit) >= ln(p/(1-p)) (default p=0.5 -> argmax).
# Pinned to release v1.0.0 (best_model.onnx). INT8 variant available too.
# Source: https://github.com/facenox/face-antispoof-onnx
# ---------------------------------------------------------------------------
MINIFASNET_NAME="minifasnet_v2se.onnx"
MINIFASNET_URL="https://github.com/facenox/face-antispoof-onnx/releases/download/v1.0.0/best_model.onnx"
MINIFASNET_SHA256="af2381b88f38769222ed93379e12444e2a50814575de1c46170de570c55a42b6"
# Optional INT8 (~600 KB) drop-in (best_model_quantized.onnx); not used by default.
MINIFASNET_INT8_NAME="minifasnet_v2se_int8.onnx"
MINIFASNET_INT8_URL="https://github.com/facenox/face-antispoof-onnx/releases/download/v1.0.0/best_model_quantized.onnx"
MINIFASNET_INT8_SHA256="fde20585635cae62ed1d41796f76b6f8bc4b92cd91ec1cf0f1bc6485d2d587a9"

verify_sha256() {
    local file="$1" expected="$2"
    local actual
    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "${file}" | awk '{print $1}')"
    else
        actual="$(shasum -a 256 "${file}" | awk '{print $1}')"
    fi
    if [[ "${actual}" != "${expected}" ]]; then
        echo "ERROR: SHA-256 mismatch for ${file}" >&2
        echo "  expected: ${expected}" >&2
        echo "  actual:   ${actual}" >&2
        return 1
    fi
}

# Download + verify a model, then mirror it into the runtime dir.
fetch_model() {
    local name="$1" url="$2" sha="$3"
    local dest="${DATA_MODELS}/${name}"

    if [[ -f "${dest}" ]] && verify_sha256 "${dest}" "${sha}" 2>/dev/null; then
        echo "✓ ${name} already present and verified"
    else
        echo "↓ downloading ${name}"
        echo "  ${url}"
        curl -fL --retry 3 -o "${dest}" "${url}"
        verify_sha256 "${dest}" "${sha}"
        echo "✓ ${name} downloaded and verified"
    fi

    cp -f "${dest}" "${RUNTIME_MODELS}/${name}"
    echo "  → ${RUNTIME_MODELS}/${name}"
}

# Verify a locally-committed (exported) model and mirror it to the runtime dir.
# Optionally runs an export command if the file is missing and a torch venv exists.
verify_local_model() {
    local name="$1" sha="$2" export_cmd="${3:-}"
    local dest="${DATA_MODELS}/${name}"

    if [[ ! -f "${dest}" ]]; then
        if [[ -n "${export_cmd}" && -x "${VENV_PY}" ]]; then
            echo "… ${name} missing — attempting export via ${VENV_PY}"
            ( cd "${REPO_ROOT}" && eval "${export_cmd}" )
        else
            echo "ERROR: ${name} missing from ${DATA_MODELS} and no torch venv at ${VENV_PY}." >&2
            echo "  Re-create it with the matching scripts/export_*.py in a torch env." >&2
            return 1
        fi
    fi

    # Skip strict verification if the pinned hash is still a placeholder.
    if [[ "${sha}" == __*__ ]]; then
        echo "⚠ ${name} present (sha pin not set yet — skipping strict verify)"
    else
        verify_sha256 "${dest}" "${sha}"
        echo "✓ ${name} present and verified"
    fi

    cp -f "${dest}" "${RUNTIME_MODELS}/${name}"
    echo "  → ${RUNTIME_MODELS}/${name}"
}

echo "Repo models dir:    ${DATA_MODELS}"
echo "Runtime models dir: ${RUNTIME_MODELS}"
echo

fetch_model "${YUNET_NAME}" "${YUNET_URL}" "${YUNET_SHA256}"
verify_local_model "${RECOGNIZER_NAME}" "${RECOGNIZER_SHA256}" \
    "${VENV_PY} scripts/export_edgeface.py --out ${DATA_MODELS}/${RECOGNIZER_NAME}"
fetch_model "${MINIFASNET_NAME}" "${MINIFASNET_URL}" "${MINIFASNET_SHA256}"
fetch_model "${MINIFASNET_INT8_NAME}" "${MINIFASNET_INT8_URL}" "${MINIFASNET_INT8_SHA256}"

echo
echo "Done. YuNet detector + EdgeFace-S recognizer + MiniFASNetV2-SE liveness are ready."
