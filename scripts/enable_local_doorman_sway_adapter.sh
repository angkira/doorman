#!/usr/bin/env sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: sudo $0" >&2
    exit 1
fi

ts="$(date +%Y%m%d-%H%M%S)"
target_user="${SUDO_USER:-angkira}"
home="/home/${target_user}"
uid="$(id -u "$target_user")"

if ! command -v gtklock >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        DEBIAN_FRONTEND=noninteractive apt-get install -y gtklock
    else
        echo "WARN: gtklock is not installed and apt-get is unavailable; falling back to swaylock." >&2
    fi
fi

if [ -f /etc/doorman/doorman.toml ]; then
    cp -a /etc/doorman/doorman.toml "/etc/doorman/doorman.toml.bak-sway-adapter-${ts}"
    if grep -q '^[[:space:]]*proactive_unlock[[:space:]]*=' /etc/doorman/doorman.toml; then
        sed -i 's/^[[:space:]]*proactive_unlock[[:space:]]*=.*/proactive_unlock = false  # Sway adapter owns face unlock/' /etc/doorman/doorman.toml
    else
        printf '\nproactive_unlock = false  # Sway adapter owns face unlock\n' >> /etc/doorman/doorman.toml
    fi
    python3 -c 'import tomllib; tomllib.load(open("/etc/doorman/doorman.toml","rb")); print("doorman.toml: ok")'
fi

if [ -f /etc/pam.d/swaylock ]; then
    cp -a /etc/pam.d/swaylock "/etc/pam.d/swaylock.bak-sway-adapter-${ts}"
    sed -i 's/^[[:space:]]*auth[[:space:]]\+sufficient[[:space:]]\+\(pam_doorman\.so\|libpam_doorman\.so\)/# disabled: Sway adapter owns face unlock: auth sufficient pam_doorman.so/' /etc/pam.d/swaylock
fi

if [ -f /etc/pam.d/gtklock ]; then
    cp -a /etc/pam.d/gtklock "/etc/pam.d/gtklock.bak-sway-adapter-${ts}"
    sed -i 's/^[[:space:]]*auth[[:space:]]\+sufficient[[:space:]]\+\(pam_doorman\.so\|libpam_doorman\.so\)/# disabled: Sway adapter owns face unlock: auth sufficient pam_doorman.so/' /etc/pam.d/gtklock
elif command -v gtklock >/dev/null 2>&1; then
    printf 'auth include login\n' > /etc/pam.d/gtklock
fi

watch="${home}/.local/bin/doorman-lock-watch"
lock="${home}/.local/bin/doorman-lock"
recover="${home}/.local/bin/sway-recover-display"
install -d -m 0755 "${home}/.local/bin"
install -d -m 0755 "${home}/.config/gtklock"

if [ -f "$watch" ]; then
    cp -a "$watch" "${watch}.bak-sway-adapter-${ts}"
fi
cat > "$watch" <<'WATCH'
#!/bin/sh
set -u

log=/tmp/doorman-lock-watch.log
user="${1:-${USER:-angkira}}"
locker="${2:-swaylock}"
socket="${DOORMAN_SOCKET:-/run/doorman/doorman.sock}"
interval="${DOORMAN_SWAY_POLL_INTERVAL:-2}"

{
    echo "[$(date -Is)] start user=$user locker=$locker socket=$socket"

    i=0
    while pgrep -u "$user" -x "$locker" >/dev/null 2>&1; do
        i=$((i + 1))

        out="$(doorman --socket "$socket" test "$user" 2>&1)"
        rc=$?
        printf '[%s] attempt=%s rc=%s %s\n' "$(date -Is)" "$i" "$rc" "$out"

        if [ "$rc" -eq 0 ] && printf '%s\n' "$out" | grep -q '^recognized:'; then
            echo "[$(date -Is)] recognized; closing $locker"
            pkill -TERM -u "$user" -x "$locker" >/dev/null 2>&1 || true
            sleep 1
            if pgrep -u "$user" -x "$locker" >/dev/null 2>&1; then
                echo "[$(date -Is)] $locker still alive; killing"
                pkill -KILL -u "$user" -x "$locker" >/dev/null 2>&1 || true
            fi
            command -v swaymsg >/dev/null 2>&1 && swaymsg "output * power on" >/dev/null 2>&1 || true
            exit 0
        fi

        sleep "$interval"
    done

    echo "[$(date -Is)] $locker gone; stop"
} >> "$log" 2>&1
WATCH
chmod 0755 "$watch"
chown "$target_user:$target_user" "$watch" 2>/dev/null || true

if [ -f "$lock" ]; then
    cp -a "$lock" "${lock}.bak-sway-adapter-${ts}"
fi
cat > "$lock" <<'LOCK'
#!/bin/sh
set -u

log=/tmp/doorman-lock.log
gtk_config="${XDG_CONFIG_HOME:-$HOME/.config}/gtklock/config.ini"
swaylock_config="${XDG_CONFIG_HOME:-$HOME/.config}/swaylock/config"
backend="${DOORMAN_LOCK_BACKEND:-auto}"
watcher="$HOME/.local/bin/doorman-lock-watch"

{
    echo "[$(date -Is)] start pid=$$ user=${USER:-?} display=${WAYLAND_DISPLAY:-?} swaysock=${SWAYSOCK:-?}"

    if [ "$backend" = "auto" ]; then
        if command -v gtklock >/dev/null 2>&1; then
            backend=gtklock
        else
            backend=swaylock
        fi
    fi

    case "$backend" in
        gtklock)
            if ! command -v gtklock >/dev/null 2>&1; then
                echo "[$(date -Is)] gtklock missing"
                exit 1
            fi
            gtklock --daemonize --config "$gtk_config"
            rc=$?
            locker=gtklock
            ;;
        swaylock)
            if ! command -v swaylock >/dev/null 2>&1; then
                echo "[$(date -Is)] swaylock missing"
                notify-send "Sway" "No supported screen locker is installed" 2>/dev/null || true
                exit 1
            fi
            if [ -f "$swaylock_config" ]; then
                swaylock --daemonize --config "$swaylock_config" --color 2E3440
            else
                swaylock --daemonize --color 2E3440
            fi
            rc=$?
            locker=swaylock
            ;;
        *)
            echo "[$(date -Is)] unsupported backend=$backend"
            exit 2
            ;;
    esac

    echo "[$(date -Is)] backend=$backend rc=$rc"

    if [ "$rc" -eq 0 ]; then
        "$watcher" "${USER:-angkira}" "$locker" >/dev/null 2>&1 &
        echo "[$(date -Is)] doorman watcher pid=$! locker=$locker"
    fi

    exit "$rc"
} >> "$log" 2>&1
LOCK
chmod 0755 "$lock"
chown "$target_user:$target_user" "$lock" 2>/dev/null || true

if [ -f "${home}/.local/bin/sway-recover-display" ]; then
    cp -a "${home}/.local/bin/sway-recover-display" "${home}/.local/bin/sway-recover-display.bak-sway-adapter-${ts}"
fi
install -m 0755 "$(dirname "$0")/sway-recover-display" "$recover"
chown "$target_user:$target_user" "$recover" 2>/dev/null || true

cat > "${home}/.config/gtklock/config.ini" <<EOF
[main]
gtk-theme=Adwaita-dark
style=${home}/.config/gtklock/style.css
time-format=%H:%M
date-format=%A, %d %B
idle-hide=false
background=${home}/Pictures/Wallpapers/daniil-silantev-3pW91fGAKiE-unsplash.jpg
EOF

cat > "${home}/.config/gtklock/style.css" <<'CSS'
window {
    background-color: #2e3440;
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
}

#window-box {
    background-color: rgba(46, 52, 64, 0.74);
    border: 1px solid rgba(136, 192, 208, 0.70);
    border-radius: 8px;
    padding: 22px;
}

#clock-label {
    color: #eceff4;
    font-family: Roboto, sans-serif;
    font-size: 56px;
    font-weight: 500;
}

#date-label {
    color: #8fbcbb;
    font-family: Roboto, sans-serif;
    font-size: 18px;
}

#input-field {
    color: #eceff4;
    background-color: rgba(59, 66, 82, 0.92);
    border: 1px solid #5e81ac;
    border-radius: 6px;
    padding: 8px 10px;
    font-family: Roboto, sans-serif;
    font-size: 16px;
}

#unlock-button {
    color: #2e3440;
    background: #88c0d0;
    border: 0;
    border-radius: 6px;
    padding: 8px 14px;
    font-family: Roboto, sans-serif;
    font-weight: 500;
}
CSS
chown -R "$target_user:$target_user" "${home}/.config/gtklock" 2>/dev/null || true

if [ -f "${home}/.config/swaylock/config" ]; then
    cp -a "${home}/.config/swaylock/config" "${home}/.config/swaylock/config.bak-sway-adapter-${ts}"
    sed -i '/^[[:space:]]*daemonize[[:space:]]*$/d' "${home}/.config/swaylock/config"
    chown "$target_user:$target_user" "${home}/.config/swaylock/config" 2>/dev/null || true
fi

if [ -f "${home}/.config/sway/config" ]; then
    cp -a "${home}/.config/sway/config" "${home}/.config/sway/config.bak-sway-adapter-${ts}"
    sed -i "s|^set \\\$lock .*|set \\\$lock ${lock}|" "${home}/.config/sway/config"
    chown "$target_user:$target_user" "${home}/.config/sway/config" 2>/dev/null || true
fi

for helper in "${home}/.local/bin/sway-idle" "${home}/.local/bin/waybar-click"; do
    [ -f "$helper" ] || continue
    cp -a "$helper" "${helper}.bak-sway-adapter-${ts}"
    sed -i "s|${home}/.local/bin/sway-lock|${lock}|g" "$helper"
    chown "$target_user:$target_user" "$helper" 2>/dev/null || true
done

systemctl restart doormand.service

for sock in "/run/user/${uid}"/sway-ipc."${uid}".*.sock; do
    [ -S "$sock" ] || continue
    runuser -u "$target_user" -- env XDG_RUNTIME_DIR="/run/user/${uid}" SWAYSOCK="$sock" swaymsg reload >/dev/null 2>&1 || true
    runuser -u "$target_user" -- env XDG_RUNTIME_DIR="/run/user/${uid}" SWAYSOCK="$sock" pkill -x swayidle >/dev/null 2>&1 || true
    runuser -u "$target_user" -- env XDG_RUNTIME_DIR="/run/user/${uid}" SWAYSOCK="$sock" swaymsg exec "${home}/.local/bin/sway-idle" >/dev/null 2>&1 || true
    break
done

echo "Doorman Sway adapter enabled."
echo "Lock backend: $(command -v gtklock >/dev/null 2>&1 && echo gtklock || echo swaylock)"
grep -n 'proactive_unlock' /etc/doorman/doorman.toml || true
grep -n 'doorman' /etc/pam.d/swaylock /etc/pam.d/gtklock 2>/dev/null || true
sh -n "$watch" "$lock"
systemctl is-active doormand.service || true
