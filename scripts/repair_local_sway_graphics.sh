#!/usr/bin/env sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: sudo $0" >&2
    exit 1
fi

ts="$(date +%Y%m%d-%H%M%S)"
target_user="${SUDO_USER:-angkira}"
session_name="sway-system-mesa.desktop"
wrapper="/usr/local/bin/sway-system-mesa"
xwayland_wrapper="/usr/local/bin/Xwayland-system-mesa"
desktop="/usr/share/wayland-sessions/${session_name}"

tmp="$(mktemp)"
cat > "$tmp" <<'WRAPPER'
#!/usr/bin/env sh
set -eu

# Keep Sway/wlroots on the distro graphics stack. The AMDGPU installer adds
# /opt/amdgpu to ld.so.conf, which can make Sway mix Ubuntu wlroots with AMDGPU
# Mesa/Wayland/GBM libraries and crash in the renderer.
#
# Use dynamic-linker options instead of exporting LD_LIBRARY_PATH. This keeps
# the library override local to compositor/Xwayland processes and avoids
# leaking it into terminals, training jobs, ROCm tools, or apps launched from
# Sway.
export WLR_XWAYLAND=/usr/local/bin/Xwayland-system-mesa
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export GBM_BACKENDS_PATH=/usr/lib/x86_64-linux-gnu/gbm
export LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
exec /lib64/ld-linux-x86-64.so.2 \
    --library-path /usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    /usr/bin/sway "$@"
WRAPPER
install -m 0755 "$tmp" "$wrapper"
rm -f "$tmp"

tmp="$(mktemp)"
cat > "$tmp" <<'XWAYLAND_WRAPPER'
#!/usr/bin/env sh
set -eu

# Keep Xwayland on the same distro graphics stack as Sway. Without this,
# ld.so.cache can prefer /opt/amdgpu for libwayland/libdrm/libgbm while Sway
# itself is using Ubuntu Mesa, which has produced wlroots renderer aborts.
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export GBM_BACKENDS_PATH=/usr/lib/x86_64-linux-gnu/gbm
export LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
exec /lib64/ld-linux-x86-64.so.2 \
    --library-path /usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    /usr/bin/Xwayland "$@"
XWAYLAND_WRAPPER
install -m 0755 "$tmp" "$xwayland_wrapper"
rm -f "$tmp"

tmp="$(mktemp)"
cat > "$tmp" <<DESKTOP
[Desktop Entry]
Name=Sway (system Mesa)
Comment=Sway using distro Mesa/Wayland/GBM libraries
Exec=${wrapper}
Type=Application
DesktopNames=sway
DESKTOP
install -m 0644 "$tmp" "$desktop"
rm -f "$tmp"

for conf in /etc/sddm.conf.d/kde_settings.conf /etc/sddm.conf.d/20-default-session.conf /etc/sddm.conf.d/99-local.conf; do
    if [ -f "$conf" ]; then
        cp -a "$conf" "${conf}.bak-${ts}"
        if grep -q '^Session=' "$conf"; then
            sed -i "s/^Session=.*/Session=${session_name}/" "$conf"
        else
            printf '\nSession=%s\n' "$session_name" >> "$conf"
        fi
        if grep -q '^User=' "$conf"; then
            sed -i "s/^User=.*/User=${target_user}/" "$conf"
        fi
    fi
done

echo "Wrapper dependency check:"
/lib64/ld-linux-x86-64.so.2 \
    --library-path /usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    --list /usr/bin/sway \
    | grep -E 'lib(drm|wayland-server|wayland-client|gbm|EGL|GLESv2|vulkan)\.so' \
    || true

if /lib64/ld-linux-x86-64.so.2 \
    --library-path /usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    --list /usr/bin/sway | grep -q '/opt/amdgpu'; then
    echo "ERROR: /opt/amdgpu is still present in Sway dependency resolution" >&2
    exit 1
fi

echo "Xwayland wrapper dependency check:"
"$xwayland_wrapper" -version >/dev/null 2>&1 || true
/lib64/ld-linux-x86-64.so.2 \
    --library-path /usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    --list /usr/bin/Xwayland \
    | grep -E 'lib(drm|wayland-client|gbm|EGL|GLX_mesa)\.so' \
    || true

if /lib64/ld-linux-x86-64.so.2 \
    --library-path /usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    --list /usr/bin/Xwayland | grep -q '/opt/amdgpu'; then
    echo "ERROR: /opt/amdgpu is still present in Xwayland dependency resolution" >&2
    exit 1
fi

echo "SDDM active session config:"
find /etc/sddm.conf.d -maxdepth 1 -name '*.conf' -exec grep -H '^Session=' {} + 2>/dev/null || true
