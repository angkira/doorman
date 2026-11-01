#!/bin/bash
# Wire pam_doorman.so into the sudo PAM service, or take it out again.
#
#   doorman-sudo-pam.sh status
#   doorman-sudo-pam.sh enable  --i-have-a-root-shell-open
#   doorman-sudo-pam.sh disable --i-have-a-root-shell-open
#
# Edits ${PAM_DIR:-/etc/pam.d}/sudo: a timestamped backup is taken first, the
# line `auth sufficient pam_doorman.so` is inserted before the first `auth`
# line or `@include`, and the result is validated (a password auth path must
# remain) before it replaces the file. Never run by a package install: a wrong
# PAM edit can lock you out of sudo. See docs/SUDO.md.
set -eu

PAM_DIR="${PAM_DIR:-/etc/pam.d}"
PAM_FILE="$PAM_DIR/sudo"
MODULE_LINE="auth sufficient pam_doorman.so"
FLAG="--i-have-a-root-shell-open"

EX_USAGE=64
EX_REFUSED=2
EX_FAILED=1

die() {
    echo "doorman-sudo-pam: $*" >&2
    exit "$EX_FAILED"
}

usage() {
    echo "usage: $0 status | enable $FLAG | disable $FLAG" >&2
    exit "$EX_USAGE"
}

# A line that already wires the module (any control flag).
is_enabled() {
    grep -Eq '^[[:space:]]*auth[[:space:]].*pam_doorman\.so' "$PAM_FILE"
}

warn_root_shell() {
    cat >&2 <<EOF
================================================================
 WARNING: this edits $PAM_FILE
 A mistake can lock you out of sudo. Keep a root shell open in
 ANOTHER terminal until you have tested 'sudo -k; sudo true'.
 Recovery from a root shell:   cp -p BACKUP $PAM_FILE
 Without one: log in as root on a TTY (or 'su -', or boot into
 recovery mode) and restore the newest ${PAM_FILE}.doorman-backup.*
================================================================
EOF
}

# Require the explicit acknowledgement flag before touching anything.
require_ack() {
    warn_root_shell
    if [ "${1:-}" != "$FLAG" ]; then
        echo "doorman-sudo-pam: refusing: pass $FLAG to confirm a root shell is open" >&2
        exit "$EX_REFUSED"
    fi
}

module_installed() {
    if [ -n "${PAM_MODULE_DIR:-}" ]; then
        [ -e "$PAM_MODULE_DIR/pam_doorman.so" ]
        return
    fi
    local dir
    for dir in /lib/security /lib64/security /usr/lib/security /usr/lib64/security \
        /lib/*-linux-gnu/security /usr/lib/*-linux-gnu/security; do
        [ -e "$dir/pam_doorman.so" ] && return 0
    done
    return 1
}

# A password auth path remains: an auth line or include that is not ours.
has_other_auth() {
    grep -Ev 'pam_doorman\.so' "$1" | grep -Eq '^[[:space:]]*(-?auth[[:space:]]|@include[[:space:]])'
}

backup() {
    local stamp dest n=0
    stamp="$(date +%Y%m%dT%H%M%S)"
    dest="$PAM_FILE.doorman-backup.$stamp"
    while [ -e "$dest" ]; do
        n=$((n + 1))
        dest="$PAM_FILE.doorman-backup.$stamp.$n"
    done
    cp -p "$PAM_FILE" "$dest"
    echo "backup: $dest" >&2
    echo "recovery: cp -p '$dest' '$PAM_FILE'" >&2
}

# Replace $PAM_FILE with $1 (same directory, so the rename is atomic).
install_file() {
    mv -f "$1" "$PAM_FILE"
}

cmd_enable() {
    require_ack "${1:-}"
    is_enabled && {
        echo "already enabled: $PAM_FILE"
        return 0
    }
    module_installed || die "pam_doorman.so not found in a PAM module directory (set PAM_MODULE_DIR to override)"
    has_other_auth "$PAM_FILE" || die "$PAM_FILE has no auth line or @include to insert before; refusing"

    local tmp
    tmp="$(mktemp "$PAM_DIR/.sudo.doorman.XXXXXX")"
    trap 'rm -f "$tmp"' EXIT
    cp -p "$PAM_FILE" "$tmp"
    awk -v line="$MODULE_LINE" '
        !done && /^[[:space:]]*(-?auth[[:space:]]|@include[[:space:]])/ { print line; done = 1 }
        { print }
    ' "$PAM_FILE" >"$tmp"

    has_other_auth "$tmp" || die "validation failed: no password auth path left; $PAM_FILE untouched"
    grep -Fxq "$MODULE_LINE" "$tmp" || die "validation failed: module line missing; $PAM_FILE untouched"
    backup
    install_file "$tmp"
    trap - EXIT
    echo "enabled: $PAM_FILE (also set [pam.sudo] enabled = true in doorman.toml)"
    echo "test:    sudo -k; sudo true"
}

cmd_disable() {
    require_ack "${1:-}"
    is_enabled || {
        echo "already disabled: $PAM_FILE"
        return 0
    }
    local tmp
    tmp="$(mktemp "$PAM_DIR/.sudo.doorman.XXXXXX")"
    trap 'rm -f "$tmp"' EXIT
    cp -p "$PAM_FILE" "$tmp"
    grep -Ev '^[[:space:]]*auth[[:space:]].*pam_doorman\.so' "$PAM_FILE" >"$tmp" || true

    has_other_auth "$tmp" || die "validation failed: no password auth path left; $PAM_FILE untouched"
    backup
    install_file "$tmp"
    trap - EXIT
    echo "disabled: $PAM_FILE"
}

cmd_status() {
    if is_enabled; then
        echo "enabled: $PAM_FILE wires pam_doorman.so"
    else
        echo "disabled: $PAM_FILE does not wire pam_doorman.so"
    fi
}

[ $# -ge 1 ] || usage
action="$1"
shift
[ $# -le 1 ] || usage
[ -f "$PAM_FILE" ] || die "$PAM_FILE not found"
case "$action" in
    status) [ $# -eq 0 ] || usage; cmd_status ;;
    enable | disable)
        [ -w "$PAM_DIR" ] || die "$PAM_DIR is not writable (run as root)"
        "cmd_$action" "${1:-}"
        ;;
    *) usage ;;
esac
