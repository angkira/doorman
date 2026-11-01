#!/usr/bin/env sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: sudo $0" >&2
    exit 1
fi

ts="$(date +%Y%m%d-%H%M%S)"
target_user="${SUDO_USER:-angkira}"

if [ -f /etc/doorman/doorman.toml ]; then
    cp -a /etc/doorman/doorman.toml "/etc/doorman/doorman.toml.bak-disable-unlock-${ts}"
    if grep -q '^[[:space:]]*proactive_unlock[[:space:]]*=' /etc/doorman/doorman.toml; then
        sed -i 's/^[[:space:]]*proactive_unlock[[:space:]]*=.*/proactive_unlock = false  # disabled: Sway red-screen lockup debugging/' /etc/doorman/doorman.toml
    else
        printf '\nproactive_unlock = false  # disabled: Sway red-screen lockup debugging\n' >> /etc/doorman/doorman.toml
    fi
    python3 -c 'import tomllib; tomllib.load(open("/etc/doorman/doorman.toml","rb")); print("doorman.toml: ok")'
fi

if [ -f /etc/pam.d/swaylock ]; then
    cp -a /etc/pam.d/swaylock "/etc/pam.d/swaylock.bak-disable-doorman-${ts}"
    sed -i 's/^[[:space:]]*auth[[:space:]]\+sufficient[[:space:]]\+\(pam_doorman\.so\|libpam_doorman\.so\)/# disabled: Sway red-screen lockup debugging: auth sufficient pam_doorman.so/' /etc/pam.d/swaylock
fi

if [ -f /etc/pam.d/sddm ]; then
    cp -a /etc/pam.d/sddm "/etc/pam.d/sddm.bak-disable-doorman-${ts}"
    sed -i 's/^[[:space:]]*auth[[:space:]]\+sufficient[[:space:]]\+\(pam_doorman\.so\|libpam_doorman\.so\)/# disabled: Sway red-screen lockup debugging: auth sufficient pam_doorman.so/' /etc/pam.d/sddm
fi

lock_script="/home/${target_user}/.local/bin/sway-lock"
if [ -f "$lock_script" ]; then
    cp -a "$lock_script" "${lock_script}.bak-disable-doorman-${ts}"
    perl -0pi -e 's/\n    if \[ "\$rc" -eq 0 \]; then\n        .*?sway-doorman-watch.*?\n    fi\n/\n    if [ "\$rc" -eq 0 ]; then\n        echo "[\$(date -Is)] doorman watcher disabled while debugging red-screen lockup"\n    fi\n/s' "$lock_script"
fi

systemctl restart doormand.service
pkill -u "$target_user" -x swaylock >/dev/null 2>&1 || true

echo "Doorman auto-unlock disabled."
grep -n 'proactive_unlock' /etc/doorman/doorman.toml || true
grep -n 'doorman' /etc/pam.d/swaylock /etc/pam.d/sddm 2>/dev/null || true
systemctl is-active doormand.service || true
