#!/usr/bin/env bash
# =============================================================================
# build_ncnn.sh — reproducibly build a STATIC libncnn with Vulkan for doorman
# =============================================================================
#
# WHAT THIS PRODUCES
#   A static ncnn install at <prefix> with the exact layout the doorman daemon
#   build (daemon/build.rs, env DOORMAN_NCNN_DIR) expects:
#       <prefix>/lib/libncnn.a
#       <prefix>/lib/libglslang.a
#       <prefix>/lib/libSPIRV.a libMachineIndependent.a libGenericCodeGen.a
#       <prefix>/lib/libOSDependent.a libglslang-default-resource-limits.a
#       <prefix>/include/ncnn/*.h
#   The hybrid Rust binary links these archives statically; nothing from this
#   prefix is needed at runtime (see RUNTIME below).
#
# PINNED VERSION
#   ncnn release tag 20260526 (Tencent/ncnn), fetched as the official
#   "-full-source.zip" release tarball, which BUNDLES a known-good glslang —
#   no recursive submodule clone, no network surprises. This tag is kept in
#   lock-step with the pnnx version used by scripts/convert_ncnn.sh (also
#   20260526) so the converted .param/.bin and the linked runtime share one ABI.
#
# BUILD-TIME APT PACKAGES (Debian/Ubuntu)
#   build-essential        # C/C++ compiler + make
#   cmake                  # >= 3.10
#   libvulkan-dev          # Vulkan headers (vulkan/vulkan.h) + loader dev
#   curl  (or wget)        # download the release tarball
#   unzip                  # extract the -full-source.zip
#   One of:  apt-get install -y build-essential cmake libvulkan-dev curl unzip
#
# RUNTIME APT PACKAGES (on the deployment machine)
#   The static lib is baked into the binary. At RUNTIME you only need the
#   Vulkan loader + an ICD for the iGPU:
#       libvulkan1             # the Vulkan loader (libvulkan.so.1)
#       mesa-vulkan-drivers    # RADV/ANV ICD for AMD/Intel integrated GPUs
#   (No libncnn, no glslang, no Vulkan -dev packages at runtime.)
#
# COST
#   ~22 MB download, builds with all cores. Approx 3-6 min on a 16-core box.
#   Resulting libncnn.a is ~25-27 MB; the whole <prefix> is ~70-90 MB.
#
# USAGE
#   scripts/build_ncnn.sh <prefix>
#   FORCE=1 scripts/build_ncnn.sh <prefix>     # rebuild even if it already exists
#   NCNN_TAG=20260113 scripts/build_ncnn.sh <prefix>   # override pinned tag
#
# IDEMPOTENT: if <prefix>/lib/libncnn.a already exists it is a no-op unless
# FORCE=1 is set.
# =============================================================================
set -euo pipefail

NCNN_TAG="${NCNN_TAG:-20260526}"

die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

[ $# -ge 1 ] || die "usage: $0 <prefix>  (e.g. $0 /opt/ncnn)"
PREFIX="$1"
# Make the prefix absolute (build dirs are created relative to CWD otherwise).
case "$PREFIX" in
  /*) : ;;
  *)  PREFIX="$(pwd)/$PREFIX" ;;
esac

# --- idempotency guard ------------------------------------------------------
if [ -f "$PREFIX/lib/libncnn.a" ] && [ "${FORCE:-0}" != "1" ]; then
  info "libncnn.a already present at $PREFIX/lib — nothing to do (set FORCE=1 to rebuild)."
  exit 0
fi

# --- tool / dependency checks ----------------------------------------------
need_tool() {
  command -v "$1" >/dev/null 2>&1 || die "missing '$1' — apt-get install -y $2"
}
need_tool cmake cmake
need_tool make  build-essential
# a C++ compiler: prefer c++/g++ but accept clang++
if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
  die "no C++ compiler found — apt-get install -y build-essential"
fi
need_tool unzip unzip
# downloader: curl or wget
if command -v curl >/dev/null 2>&1; then DL="curl -fL --retry 3 -o";
elif command -v wget >/dev/null 2>&1; then DL="wget -O";
else die "need curl or wget — apt-get install -y curl"; fi
# Vulkan headers (ncnn requires them when NCNN_VULKAN=ON)
if [ ! -f /usr/include/vulkan/vulkan.h ] && [ -z "${VULKAN_SDK:-}" ]; then
  die "Vulkan headers not found (vulkan/vulkan.h) — apt-get install -y libvulkan-dev"
fi

# --- fetch + extract the pinned full-source release tarball -----------------
WORK="$(mktemp -d /tmp/build_ncnn.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
ZIP="$WORK/ncnn-${NCNN_TAG}-full-source.zip"
URL="https://github.com/Tencent/ncnn/releases/download/${NCNN_TAG}/ncnn-${NCNN_TAG}-full-source.zip"

info "downloading ncnn ${NCNN_TAG} full-source (bundles glslang) ..."
$DL "$ZIP" "$URL" || die "download failed: $URL"
[ -s "$ZIP" ] || die "downloaded tarball is empty: $ZIP"

info "extracting ..."
# The official -full-source.zip extracts the tree directly (no ncnn-* wrapper
# dir), so unpack into a dedicated subdir we control.
SRC="$WORK/src"
mkdir -p "$SRC"
unzip -q "$ZIP" -d "$SRC"
# If a future tarball ever wraps everything in a single top-level dir, descend
# into it so CMakeLists.txt is at $SRC.
if [ ! -f "$SRC/CMakeLists.txt" ]; then
  inner="$(find "$SRC" -maxdepth 1 -mindepth 1 -type d | head -1)"
  [ -n "$inner" ] && [ -f "$inner/CMakeLists.txt" ] && SRC="$inner"
fi
[ -f "$SRC/CMakeLists.txt" ] || die "could not locate ncnn CMakeLists.txt after extract"

# --- configure + build + install -------------------------------------------
BUILD="$SRC/build"
mkdir -p "$BUILD"
info "configuring (static + Vulkan, no tools/examples) ..."
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DNCNN_VULKAN=ON \
  -DNCNN_SHARED_LIB=OFF \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF \
  -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_SYSTEM_GLSLANG=OFF \
  -DNCNN_PYTHON=OFF

info "building with $(nproc) jobs (this is the slow part) ..."
cmake --build "$BUILD" --config Release -j "$(nproc)"

info "installing into $PREFIX ..."
cmake --install "$BUILD"

# --- verify the expected layout --------------------------------------------
[ -f "$PREFIX/lib/libncnn.a" ]          || die "install incomplete: $PREFIX/lib/libncnn.a missing"
[ -f "$PREFIX/include/ncnn/net.h" ]     || die "install incomplete: $PREFIX/include/ncnn/net.h missing"
[ -f "$PREFIX/lib/libglslang.a" ]       || die "install incomplete: $PREFIX/lib/libglslang.a missing (needed for static link)"

info "DONE. ncnn ${NCNN_TAG} static+Vulkan installed at: $PREFIX"
echo  "      point the daemon build at it with:  DOORMAN_NCNN_DIR=$PREFIX"
ls -la "$PREFIX/lib"/libncnn.a "$PREFIX/lib"/libglslang*.a 2>/dev/null || true
