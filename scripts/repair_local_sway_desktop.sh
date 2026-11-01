#!/usr/bin/env sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: sudo $0" >&2
    exit 1
fi

ts="$(date +%Y%m%d-%H%M%S)"
target_user="${SUDO_USER:-angkira}"

if [ -f /etc/doorman/doorman.toml ]; then
    cp -a /etc/doorman/doorman.toml "/etc/doorman/doorman.toml.bak-${ts}"
    perl -0pi -e 's/aggregation_window\s*=\s*\n\s*([0-9]+)/aggregation_window = $1/g' /etc/doorman/doorman.toml
    python3 -c 'import tomllib; tomllib.load(open("/etc/doorman/doorman.toml","rb")); print("doorman.toml: ok")'
fi

for conf in /etc/sddm.conf.d/kde_settings.conf /etc/sddm.conf.d/20-default-session.conf /etc/sddm.conf.d/99-local.conf; do
    if [ -f "$conf" ]; then
        cp -a "$conf" "${conf}.bak-${ts}"
        if grep -q '^Session=' "$conf"; then
            sed -i 's/^Session=.*/Session=sway.desktop/' "$conf"
        else
            printf '\nSession=sway.desktop\n' >> "$conf"
        fi
        if grep -q '^User=' "$conf"; then
            sed -i "s/^User=.*/User=${target_user}/" "$conf"
        fi
    fi
done

if systemctl list-unit-files --no-legend doorman-depthd.service 2>/dev/null | grep -q '^doorman-depthd.service'; then
    if [ -f /etc/doorman/doorman.toml ] && grep -Eq '^[[:space:]]*device[[:space:]]*=[[:space:]]*"rocm"' /etc/doorman/doorman.toml; then
        systemctl reset-failed doorman-depthd.service 2>/dev/null || true
        systemctl restart doorman-depthd.service || true
    else
        systemctl disable --now doorman-depthd.service 2>/dev/null || true
        systemctl reset-failed doorman-depthd.service 2>/dev/null || true
    fi
fi

systemctl restart doormand.service

echo "SDDM default session:"
find /etc/sddm.conf.d -maxdepth 1 -name '*.conf' -exec grep -H '^Session=' {} + 2>/dev/null || true
# shellcheck disable=SC2015 # best-effort diagnostic: '|| true' must run on either failure
[ -f /etc/sddm.conf ] && grep -H '^Session=' /etc/sddm.conf || true
systemctl is-active doormand.service || true
systemctl is-active doorman-depthd.service || true
