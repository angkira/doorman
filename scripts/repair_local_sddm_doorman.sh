#!/usr/bin/env sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: sudo $0" >&2
    exit 1
fi

script_dir="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(CDPATH='' cd -- "${script_dir}/.." && pwd)"
ts="$(date +%Y%m%d-%H%M%S)"
target_user="${SUDO_USER:-${USER:?set SUDO_USER or USER}}"
session_name="sway-system-mesa.desktop"
sddm_override="/etc/sddm.conf.d/zz-doorman-sway.conf"
sddm_theme="${DOORMAN_SDDM_THEME:-breeze}"
sddm_cursor_theme="${DOORMAN_SDDM_CURSOR_THEME:-breeze_cursors}"
sddm_font="${DOORMAN_SDDM_FONT:-Roboto Medium,11,-1,5,500,0,0,0,0,0,0,0,0,0,0,1,Regular}"

if [ -x "${script_dir}/repair_local_sway_graphics.sh" ]; then
    "${script_dir}/repair_local_sway_graphics.sh"
fi

if [ "$sddm_theme" != "breeze" ] && [ ! -d "/usr/share/sddm/themes/${sddm_theme}" ]; then
    echo "ERROR: SDDM theme /usr/share/sddm/themes/${sddm_theme} is missing" >&2
    exit 1
fi

if [ -f "$sddm_override" ]; then
    cp -a "$sddm_override" "${sddm_override}.bak-${ts}"
fi

cat > "$sddm_override" <<EOF
[General]
Numlock=on

[Autologin]
User=${target_user}
Session=${session_name}
Relogin=false

[Theme]
Current=${sddm_theme}
CursorTheme=${sddm_cursor_theme}
Font=${sddm_font}
EOF

if [ -d "${repo_root}/packaging/adapters/sway" ]; then
    install -d -m 0755 /usr/lib/doorman/adapters/sway
    install -m 0755 "${repo_root}/packaging/adapters/sway/doorman-sway-lock" /usr/lib/doorman/adapters/sway/doorman-sway-lock
    install -m 0755 "${repo_root}/packaging/adapters/sway/doorman-sway-watch" /usr/lib/doorman/adapters/sway/doorman-sway-watch
    install -m 0644 "${repo_root}/packaging/adapters/sway/sway.conf.example" /usr/lib/doorman/adapters/sway/sway.conf.example
fi

if [ -f "${repo_root}/scripts/sway-recover-display" ]; then
    install -m 0755 "${repo_root}/scripts/sway-recover-display" "/home/${target_user}/.local/bin/sway-recover-display"
fi

if [ -f "${repo_root}/scripts/sway-doorman-watch" ]; then
    watcher="/home/${target_user}/.local/bin/sway-doorman-watch"
    if [ -f "$watcher" ]; then
        cp -a "$watcher" "${watcher}.bak-${ts}"
    fi
    install -m 0755 "${repo_root}/scripts/sway-doorman-watch" "$watcher"
    chown "$target_user:$target_user" "$watcher" 2>/dev/null || true
fi

if [ ! -e /lib/x86_64-linux-gnu/security/pam_doorman.so ] && [ -e /lib/x86_64-linux-gnu/security/libpam_doorman.so ]; then
    ln -s libpam_doorman.so /lib/x86_64-linux-gnu/security/pam_doorman.so
fi

if [ ! -e /lib/x86_64-linux-gnu/security/pam_doorman.so ]; then
    echo "ERROR: pam_doorman.so is missing under /lib/x86_64-linux-gnu/security" >&2
    exit 1
fi

if [ -f /etc/pam.d/swaylock ]; then
    cp -a /etc/pam.d/swaylock "/etc/pam.d/swaylock.bak-${ts}"
else
    printf '# PAM configuration for swaylock\n' > /etc/pam.d/swaylock
fi

if ! grep -Eq '^[[:space:]]*auth[[:space:]]+sufficient[[:space:]]+(pam_doorman\.so|libpam_doorman\.so)' /etc/pam.d/swaylock; then
    if grep -q '^auth[[:space:]]\+include[[:space:]]\+login' /etc/pam.d/swaylock; then
        sed -i '0,/^auth[[:space:]]\+include[[:space:]]\+login/s//auth sufficient pam_doorman.so\
auth include login/' /etc/pam.d/swaylock
    else
        printf 'auth sufficient pam_doorman.so\nauth include login\n' >> /etc/pam.d/swaylock
    fi
fi

bindings="/home/${target_user}/.config/sway/conf.d/bindings.conf"
if [ -f "$bindings" ] && ! grep -q 'sway-recover-display' "$bindings"; then
    # shellcheck disable=SC2016 # the sway config needs a literal $mod; it must not expand here
    printf '\n# Emergency display/lock recovery\nbindsym --locked $mod+Shift+Escape exec /home/%s/.local/bin/sway-recover-display\nbindsym $mod+Shift+Escape exec /home/%s/.local/bin/sway-recover-display\n' "$target_user" "$target_user" >> "$bindings"
fi

if [ "${DOORMAN_ENABLE_SDDM_PAM:-0}" = "1" ]; then
    cp -a /etc/pam.d/sddm "/etc/pam.d/sddm.bak-${ts}"
    if ! grep -Eq '^[[:space:]]*auth[[:space:]]+sufficient[[:space:]]+(pam_doorman\.so|libpam_doorman\.so)' /etc/pam.d/sddm; then
        sed -i '0,/^@include common-auth/s//auth    sufficient      pam_doorman.so\
@include common-auth/' /etc/pam.d/sddm
    fi
fi

systemctl reload sddm.service 2>/dev/null || true

echo "SDDM final active override:"
sed -n '1,80p' "$sddm_override"
echo
echo "Sway lock target:"
# shellcheck disable=SC2016 # grep pattern: \$ is a literal dollar sign
grep -H '^set \$lock' "/home/${target_user}/.config/sway/config" 2>/dev/null || true
grep -Hn 'doorman --socket /run/doorman/doorman.sock' "/home/${target_user}/.local/bin/sway-doorman-watch" 2>/dev/null || true
echo
echo "swaylock PAM:"
grep -nE 'pam_doorman|auth include login|@include common-auth' /etc/pam.d/swaylock || true
echo
echo "Doorman services:"
systemctl is-active doormand.service || true
systemctl is-active doorman-depthd.service || true
