#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "dpkg-deb is required to build the Debian package" >&2
  exit 1
fi

version="${VERSION:-$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n1)}"
if [[ -z "$version" ]]; then
  echo "could not determine package version" >&2
  exit 1
fi
version="${version#v}"

arch="${DEB_ARCH:-}"
if [[ -z "$arch" ]]; then
  if command -v dpkg >/dev/null 2>&1; then
    arch="$(dpkg --print-architecture)"
  else
    case "$(uname -m)" in
      x86_64) arch="amd64" ;;
      aarch64|arm64) arch="arm64" ;;
      *) arch="$(uname -m)" ;;
    esac
  fi
fi

multiarch="${DEB_HOST_MULTIARCH:-}"
if [[ -z "$multiarch" ]]; then
  if command -v dpkg-architecture >/dev/null 2>&1; then
    multiarch="$(dpkg-architecture -qDEB_HOST_MULTIARCH)"
  else
    case "$arch" in
      amd64) multiarch="x86_64-linux-gnu" ;;
      arm64) multiarch="aarch64-linux-gnu" ;;
      *) multiarch="$(uname -m)-linux-gnu" ;;
    esac
  fi
fi

features="${DOORMAN_DEB_FEATURES:-backend-ort,camera-mock,camera-ffmpeg,camera-v4l2}"
build_root="$root/target/debian"
mkdir -p "$build_root"
pkgdir="$(mktemp -d "$build_root/doorman_${version}_${arch}.XXXXXX")"
chmod 0755 "$pkgdir"
distdir="$build_root/dist"

mkdir -p "$distdir"

echo "Building doorman Debian package"
echo "  version: $version"
echo "  arch: $arch"
echo "  multiarch: $multiarch"
echo "  features: $features"
echo "  staging: $pkgdir"

cargo build \
  --locked \
  --release \
  -p doormand \
  --no-default-features \
  --features "$features" \
  --bin doormand \
  --bin doorman

cargo build --locked --release -p pam_doorman

install -d -m 0755 "$pkgdir/DEBIAN"
install -d -m 0755 "$pkgdir/usr/bin"
install -d -m 0755 "$pkgdir/usr/lib/$multiarch/security"
install -d -m 0755 "$pkgdir/usr/lib/doorman/adapters/sway"
install -d -m 0755 "$pkgdir/lib/systemd/system"
install -d -m 0755 "$pkgdir/usr/lib/systemd/user"
install -d -m 0755 "$pkgdir/usr/share/doc/doorman"
install -d -m 0755 "$pkgdir/usr/share/doorman"

install -m 0755 target/release/doormand "$pkgdir/usr/bin/doormand"
install -m 0755 target/release/doorman "$pkgdir/usr/bin/doorman"
install -m 0644 target/release/libpam_doorman.so \
  "$pkgdir/usr/lib/$multiarch/security/libpam_doorman.so"
ln -s libpam_doorman.so "$pkgdir/usr/lib/$multiarch/security/pam_doorman.so"

install -m 0644 packaging/doorman.toml.example "$pkgdir/usr/share/doorman/doorman.toml.example"
install -m 0755 packaging/adapters/sway/doorman-sway-lock \
  "$pkgdir/usr/lib/doorman/adapters/sway/doorman-sway-lock"
install -m 0755 packaging/adapters/sway/doorman-sway-watch \
  "$pkgdir/usr/lib/doorman/adapters/sway/doorman-sway-watch"
install -m 0644 packaging/adapters/sway/sway.conf.example \
  "$pkgdir/usr/lib/doorman/adapters/sway/sway.conf.example"
install -m 0644 packaging/systemd/doormand.service "$pkgdir/lib/systemd/system/doormand.service"
install -m 0644 packaging/systemd/doormand-user.service \
  "$pkgdir/usr/lib/systemd/user/doormand.service"
install -m 0644 packaging/deb/README.Debian "$pkgdir/usr/share/doc/doorman/README.Debian"

runtime_models=(
  face_detection_yunet_2023mar.onnx
  edgeface_s.onnx
  minifasnet_v2se.onnx
  minifasnet_v2se_int8.onnx
)
model_count=0
for model in "${runtime_models[@]}"; do
  if [[ -f "data/models/$model" ]]; then
    if [[ "$model_count" -eq 0 ]]; then
      install -d -m 0755 "$pkgdir/usr/share/doorman/models"
    fi
    install -m 0644 "data/models/$model" "$pkgdir/usr/share/doorman/models/$model"
    model_count=$((model_count + 1))
  else
    echo "warning: data/models/$model missing; it will not be bundled" >&2
  fi
done
if [[ "$model_count" -gt 0 ]]; then
  echo "Bundled $model_count runtime model file(s)."
else
  echo "warning: no runtime models bundled; install models before starting doormand" >&2
fi

install -m 0755 packaging/deb/postinst "$pkgdir/DEBIAN/postinst"
install -m 0755 packaging/deb/prerm "$pkgdir/DEBIAN/prerm"
install -m 0755 packaging/deb/postrm "$pkgdir/DEBIAN/postrm"

installed_size="$(du -ks "$pkgdir" | awk '{print $1}')"
sed \
  -e "s/@VERSION@/$version/g" \
  -e "s/@ARCH@/$arch/g" \
  -e "s/@INSTALLED_SIZE@/$installed_size/g" \
  packaging/deb/control.in > "$pkgdir/DEBIAN/control"

find "$pkgdir" -type d -exec chmod 0755 {} +

artifact="$distdir/doorman_${version}_${arch}.deb"
dpkg-deb --build --root-owner-group "$pkgdir" "$artifact"

echo "Built $artifact"
