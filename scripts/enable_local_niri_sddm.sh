#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    printf 'run as root: sudo %s\n' "$0" >&2
    exit 1
fi

session_file="/usr/share/wayland-sessions/niri.desktop"
if [ ! -f "$session_file" ]; then
    printf 'missing session file: %s\n' "$session_file" >&2
    exit 1
fi

target="/etc/sddm.conf.d/zzzz-doorman-niri.conf"
stamp="$(date +%Y%m%d-%H%M%S)"

if [ -e "$target" ]; then
    cp -a "$target" "$target.bak-$stamp"
    printf 'backup: %s.bak-%s\n' "$target" "$stamp"
fi

autologin_user="${SUDO_USER:?run via sudo from the account that should autologin}"

cat > "$target" <<EOF
[General]
Numlock=on
DisplayServer=x11

[Autologin]
User=${autologin_user}
Session=niri.desktop
Relogin=false

[Theme]
Current=breeze
CursorTheme=breeze_cursors
Font=Roboto Medium,11,-1,5,500,0,0,0,0,0,0,0,0,0,0,1,Regular
EOF

printf 'SDDM default session set to niri.desktop via %s\n' "$target"
grep -R '^Session=' /etc/sddm.conf.d || true
