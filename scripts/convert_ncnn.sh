#!/usr/bin/env bash
# =============================================================================
# convert_ncnn.sh — convert the Depth-Anything-V2 depth ONNX to ncnn via pnnx
# =============================================================================
#
# WHAT THIS PRODUCES
#   <out_dir>/depth.ncnn.param   (ncnn graph: magic 7767517, layer/blob counts)
#   <out_dir>/depth.ncnn.bin     (ncnn weights)
#   These are the artifacts NcnnHybridBackend loads for the iGPU depth path.
#   Input blob:  in0   (shape 1x3x518x518, f32)
#   Output blob: out0  (the depth map)
#
# PINNED CONVERTER
#   pip package "pnnx" version 20260526 — kept in lock-step with the ncnn
#   runtime tag built by scripts/build_ncnn.sh (also 20260526) so the emitted
#   .param/.bin and the linked libncnn share one operator/ABI generation.
#
# EXACT INVOCATION (recovered from the P1 feasibility run, /tmp/ncnn_probe)
#   pnnx depth.onnx inputshape=[1,3,518,518]
#   The source ONNX (depth_anything_v2_small_fp32.onnx) is fed to pnnx directly;
#   pnnx ingests ONNX as well as TorchScript. The 518x518 shape is the model's
#   native patch grid (37x37 patches of 14px). Producing exactly this shape is
#   what the P1 run validated.
#
# BUILD-TIME APT PACKAGES (Debian/Ubuntu)
#   python3            # 3.10-3.12 (project is pinned to 3.12)
#   python3-venv       # to create the isolated venv
#   python3-pip        # (usually pulled by python3-venv / ensurepip)
#   One of:  apt-get install -y python3 python3-venv python3-pip
#   Everything else (pnnx, torch, onnx) is pip-installed into a local venv.
#
# RUNTIME PACKAGES
#   NONE. This is a build/conversion-time step only. The resulting .param/.bin
#   are consumed by the daemon at runtime; pnnx/torch are not needed there.
#
# COST
#   First run pip-installs pnnx (+ a CPU torch for the converter): a few hundred
#   MB of wheels, a few minutes on a cold cache. Conversion itself is < 1 min.
#   Output: depth.ncnn.bin ~48 MB, depth.ncnn.param ~21 KB.
#
# USAGE
#   scripts/convert_ncnn.sh <onnx_models_dir> <out_dir>
#       <onnx_models_dir> must contain depth_anything_v2_small_fp32.onnx
#   FORCE=1 scripts/convert_ncnn.sh <onnx_models_dir> <out_dir>   # reconvert
#   PNNX_VERSION=20260113 scripts/convert_ncnn.sh ...             # override pin
#
# IDEMPOTENT: no-op if both depth.ncnn.param and depth.ncnn.bin already exist
# in <out_dir> unless FORCE=1.
# =============================================================================
set -euo pipefail

PNNX_VERSION="${PNNX_VERSION:-20260526}"
ONNX_NAME="depth_anything_v2_small_fp32.onnx"
INPUT_SHAPE="[1,3,518,518]"

die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

[ $# -ge 2 ] || die "usage: $0 <onnx_models_dir> <out_dir>"
ONNX_DIR="$1"; OUT_DIR="$2"
case "$ONNX_DIR" in /*) : ;; *) ONNX_DIR="$(pwd)/$ONNX_DIR" ;; esac
case "$OUT_DIR"  in /*) : ;; *) OUT_DIR="$(pwd)/$OUT_DIR" ;; esac

SRC_ONNX="$ONNX_DIR/$ONNX_NAME"

# --- idempotency guard ------------------------------------------------------
if [ -f "$OUT_DIR/depth.ncnn.param" ] && [ -f "$OUT_DIR/depth.ncnn.bin" ] && [ "${FORCE:-0}" != "1" ]; then
  info "depth.ncnn.{param,bin} already present in $OUT_DIR — nothing to do (set FORCE=1 to reconvert)."
  exit 0
fi

# --- dependency checks ------------------------------------------------------
command -v python3 >/dev/null 2>&1 || die "missing python3 — apt-get install -y python3"
python3 -c 'import venv' 2>/dev/null || die "python venv module missing — apt-get install -y python3-venv"
[ -f "$SRC_ONNX" ] || die "source ONNX not found: $SRC_ONNX"

mkdir -p "$OUT_DIR"

# --- venv (refreshed if pnnx/version missing) -------------------------------
VENV="$OUT_DIR/.pnnx-venv"
PNNX_BIN="$VENV/bin/pnnx"
need_setup=1
if [ -x "$PNNX_BIN" ]; then
  have="$("$VENV/bin/pip" show pnnx 2>/dev/null | awk '/^Version:/{print $2}')"
  [ "$have" = "$PNNX_VERSION" ] && need_setup=0
fi
if [ "$need_setup" = "1" ]; then
  info "creating venv + installing pnnx==$PNNX_VERSION (this pulls torch/onnx for the converter) ..."
  rm -rf "$VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip wheel >/dev/null
  # pnnx wheels bundle their own libtorch; onnx is needed to ingest the .onnx.
  "$VENV/bin/pip" install "pnnx==$PNNX_VERSION" onnx >/dev/null
fi
[ -x "$PNNX_BIN" ] || die "pnnx not available in venv after install"

# --- run pnnx in a scratch dir (it litters intermediate files) --------------
SCRATCH="$(mktemp -d /tmp/convert_ncnn.XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
cp "$SRC_ONNX" "$SCRATCH/depth.onnx"

info "running: pnnx depth.onnx inputshape=$INPUT_SHAPE  (pnnx $PNNX_VERSION)"
( cd "$SCRATCH" && "$PNNX_BIN" depth.onnx "inputshape=$INPUT_SHAPE" ) \
  || die "pnnx conversion failed"

[ -f "$SCRATCH/depth.ncnn.param" ] || die "pnnx did not emit depth.ncnn.param"
[ -f "$SCRATCH/depth.ncnn.bin" ]   || die "pnnx did not emit depth.ncnn.bin"

# --- publish only the ncnn artifacts; SCRATCH (and all the pnnx.* / *.py
#     intermediates) is wiped by the trap above. -----------------------------
mv -f "$SCRATCH/depth.ncnn.param" "$OUT_DIR/depth.ncnn.param"
mv -f "$SCRATCH/depth.ncnn.bin"   "$OUT_DIR/depth.ncnn.bin"

# --- sanity-check the .param header -----------------------------------------
read -r magic   < "$OUT_DIR/depth.ncnn.param"
header2="$(sed -n '2p' "$OUT_DIR/depth.ncnn.param")"
layers="$(echo "$header2" | awk '{print $1}')"
blobs="$(echo  "$header2" | awk '{print $2}')"
[ "$magic" = "7767517" ] || die "bad .param magic ($magic) — expected 7767517"
[ -n "$layers" ] && [ "$layers" -gt 0 ] 2>/dev/null || die "empty layer count in .param header"
[ -n "$blobs" ]  && [ "$blobs"  -gt 0 ] 2>/dev/null || die "empty blob count in .param header"

info "DONE. wrote ncnn depth model to $OUT_DIR"
echo  "      magic=$magic  layers=$layers  blobs=$blobs  (input blob in0, output out0)"
ls -la "$OUT_DIR/depth.ncnn.param" "$OUT_DIR/depth.ncnn.bin"
