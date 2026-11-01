#!/bin/sh
set -eu

repo_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
stamp="$(date +%Y%m%d-%H%M%S)"

backup_file() {
    path="$1"
    if [ -e "$path" ]; then
        cp -a "$path" "$path.bak-$stamp"
        printf 'backup: %s.bak-%s\n' "$path" "$stamp"
    fi
}

need() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'missing required command: %s\n' "$1" >&2
        exit 1
    }
}

need niri
need waybar
need fuzzel
need xwayland-satellite
need gtklock

niri validate -c "$repo_dir/packaging/adapters/niri/config.kdl"

backup_file "$HOME/.config/niri/config.kdl"
backup_file "$HOME/.config/waybar-niri/config"
backup_file "$HOME/.config/waybar-niri/style.css"
backup_file "$HOME/.config/waybar-niri/waybar-niri-click"

install -D -m 0644 "$repo_dir/packaging/adapters/niri/config.kdl" "$HOME/.config/niri/config.kdl"
install -D -m 0644 "$repo_dir/packaging/adapters/niri/waybar/config" "$HOME/.config/waybar-niri/config"
install -D -m 0644 "$repo_dir/packaging/adapters/niri/waybar/style.css" "$HOME/.config/waybar-niri/style.css"
install -D -m 0755 "$repo_dir/packaging/adapters/niri/waybar-niri-click" "$HOME/.config/waybar-niri/waybar-niri-click"

mkdir -p "$HOME/.config/autostart"
printf '%s\n' \
    '[Desktop Entry]' \
    'Type=Application' \
    'Name=Xwayland Video Bridge' \
    'Hidden=true' \
    'X-GNOME-Autostart-enabled=false' \
    > "$HOME/.config/autostart/org.kde.xwaylandvideobridge.desktop"
printf '%s\n' \
    '[Desktop Entry]' \
    'Type=Application' \
    'Name=1Password' \
    'Hidden=true' \
    'X-GNOME-Autostart-enabled=false' \
    > "$HOME/.config/autostart/1password.desktop"
chmod 0644 "$HOME/.config/autostart/org.kde.xwaylandvideobridge.desktop" "$HOME/.config/autostart/1password.desktop"

if command -v systemctl >/dev/null 2>&1; then
    systemctl --user mask --now waybar.service >/dev/null 2>&1 || true
fi

printf 'installed niri config: %s\n' "$HOME/.config/niri/config.kdl"
printf 'installed niri waybar: %s\n' "$HOME/.config/waybar-niri/config"
printf 'installed niri click helper: %s\n' "$HOME/.config/waybar-niri/waybar-niri-click"
printf 'disabled duplicate waybar.service, 1Password autostart, and Xwayland Video Bridge autostart for this user\n'
