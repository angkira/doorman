#!/usr/bin/env bash
# Fail when the systemd exposure score of doormand.service gets worse than the
# stored threshold (packaging/systemd/security-exposure.max). Lower is better.
# Needs systemd >= 252 for `systemd-analyze security --offline=true`.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit="${1:-$root/packaging/systemd/doormand.service}"
max_file="${2:-$root/packaging/systemd/security-exposure.max}"

max="$(tr -d '[:space:]' <"$max_file")"
if ! [[ "$max" =~ ^[0-9]+\.[0-9]$ ]]; then
    echo "error: $max_file must hold one number like 5.2, got '$max'" >&2
    exit 2
fi

out="$(systemd-analyze security --offline=true --no-pager "$unit")"
printf '%s\n' "$out"

score="$(printf '%s\n' "$out" | sed -n 's/.*Overall exposure level for [^:]*: \([0-9]*\.[0-9]\).*/\1/p')"
if [ -z "$score" ]; then
    echo "error: could not parse the overall exposure level" >&2
    exit 2
fi

echo "exposure score: $score (threshold: $max)"
if awk -v s="$score" -v m="$max" 'BEGIN { exit !(s > m) }'; then
    echo "error: exposure $score is worse than the stored threshold $max" >&2
    echo "Harden the unit, or justify and raise $max_file in the same change." >&2
    exit 1
fi
