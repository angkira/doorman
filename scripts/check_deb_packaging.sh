#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

required_files=(
  "packaging/deb/control.in"
  "packaging/deb/postinst"
  "packaging/deb/prerm"
  "packaging/deb/postrm"
  "packaging/deb/README.Debian"
  "scripts/build_deb.sh"
  ".github/workflows/release.yml"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required packaging file: $path" >&2
    exit 1
  fi
done

bash -n scripts/build_deb.sh
bash -n packaging/deb/postinst
bash -n packaging/deb/prerm
bash -n packaging/deb/postrm

grep -q '@VERSION@' packaging/deb/control.in
grep -q '@ARCH@' packaging/deb/control.in
grep -q '@INSTALLED_SIZE@' packaging/deb/control.in
grep -q 'socket_path = "/run/doorman/doorman.sock"' packaging/doorman.toml.example

if grep -R '/etc/pam.d' packaging/deb/postinst packaging/deb/prerm packaging/deb/postrm; then
  echo "Debian maintainer scripts must not edit PAM configuration" >&2
  exit 1
fi

grep -q 'dpkg-deb --build --root-owner-group' scripts/build_deb.sh
grep -q '/lib/systemd/system/doormand.service' scripts/build_deb.sh
grep -q '/usr/lib/systemd/user/doormand.service' scripts/build_deb.sh
grep -q 'libpam_doorman.so' scripts/build_deb.sh
grep -q 'pam_doorman.so' scripts/build_deb.sh
grep -q '/usr/share/doorman/doorman.toml.example' scripts/build_deb.sh
grep -q '/usr/lib/doorman/adapters/sway/doorman-sway-lock' scripts/build_deb.sh
grep -q '/usr/lib/doorman/adapters/sway/doorman-sway-watch' scripts/build_deb.sh
grep -q '/usr/share/doorman/models' scripts/build_deb.sh
grep -q 'edgeface_s.onnx' scripts/build_deb.sh
grep -q '/etc/doorman/doorman.toml' packaging/deb/postinst
grep -q '/usr/share/doorman/doorman.toml.example' packaging/deb/postinst
grep -q 'systemctl enable doormand.service' packaging/deb/postinst
grep -q 'systemctl restart doormand.service' packaging/deb/postinst
grep -q '/usr/share/doorman/models' packaging/deb/postinst
grep -q '/var/lib/doorman/models' packaging/deb/postinst

grep -q 'scripts/check_deb_packaging.sh' .github/workflows/release.yml
grep -q 'scripts/build_deb.sh' .github/workflows/release.yml
grep -q 'gh release upload' .github/workflows/release.yml

shopt -s nullglob
for deb in target/debian/dist/*.deb; do
  contents="$(dpkg-deb --contents "$deb")"
  if grep -q '^drwx------' <<<"$contents"; then
    echo "Debian package contains private 0700 directories: $deb" >&2
    exit 1
  fi
  if grep -q '^drwxrwx' <<<"$contents"; then
    echo "Debian package contains group-writable directories: $deb" >&2
    exit 1
  fi
  grep -q './usr/bin/doormand$' <<<"$contents"
  grep -q './usr/bin/doorman$' <<<"$contents"
  grep -q './lib/systemd/system/doormand.service$' <<<"$contents"
  grep -q './usr/lib/systemd/user/doormand.service$' <<<"$contents"
  if grep -q './etc/doorman/doorman.toml$' <<<"$contents"; then
    echo "Debian package must not ship /etc/doorman/doorman.toml as payload" >&2
    exit 1
  fi
  grep -q './usr/share/doorman/doorman.toml.example$' <<<"$contents"
  grep -q './usr/lib/doorman/adapters/sway/doorman-sway-lock$' <<<"$contents"
  grep -q './usr/lib/doorman/adapters/sway/doorman-sway-watch$' <<<"$contents"
  grep -q './usr/lib/doorman/adapters/sway/sway.conf.example$' <<<"$contents"
  grep -q './usr/lib/.*/security/libpam_doorman.so$' <<<"$contents"
  grep -q './usr/lib/.*/security/pam_doorman.so -> libpam_doorman.so$' <<<"$contents"
  if grep -q './usr/share/doorman/models/$' <<<"$contents"; then
    grep -q './usr/share/doorman/models/face_detection_yunet_2023mar.onnx$' <<<"$contents"
    grep -q './usr/share/doorman/models/edgeface_s.onnx$' <<<"$contents"
    grep -q './usr/share/doorman/models/minifasnet_v2se.onnx$' <<<"$contents"
  fi
done

echo "Debian packaging contract OK"
