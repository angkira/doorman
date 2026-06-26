#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-/etc/doorman/doorman.toml}"
AUTH_FRAMES="${DOORMAN_AUTH_FRAMES:-10}"
AGGREGATION_WINDOW="${DOORMAN_RECOGNITION_AGGREGATION_WINDOW:-5}"
HYSTERESIS_FRAMES="${DOORMAN_RECOGNITION_HYSTERESIS_FRAMES:-3}"
MIN_SHARPNESS="${DOORMAN_MIN_SHARPNESS:-8.0}"
MIN_FACE_AREA_FRAC="${DOORMAN_MIN_FACE_AREA_FRAC:-0.005}"

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

if [ "$(id -u)" -ne 0 ] && [ ! -w "$CONFIG" ]; then
  echo "Run with sudo: sudo $0 [config-path]" >&2
  exit 1
fi

backup="${CONFIG}.bak-auth-latency-$(date +%Y%m%d-%H%M%S)"
cp -a "$CONFIG" "$backup"

tmp="$(mktemp)"
awk \
  -v auth_frames="$AUTH_FRAMES" \
  -v aggregation_window="$AGGREGATION_WINDOW" \
  -v hysteresis_frames="$HYSTERESIS_FRAMES" \
  -v min_sharpness="$MIN_SHARPNESS" \
  -v min_face_area_frac="$MIN_FACE_AREA_FRAC" '
function flush_auth_missing() {
  if (in_auth && !seen_auth_frames) print "auth_frames = " auth_frames;
}

function flush_recognition_missing() {
  if (in_recognition) {
    if (!seen_aggregation_window) print "aggregation_window = " aggregation_window;
    if (!seen_hysteresis_frames) print "hysteresis_frames = " hysteresis_frames;
    if (!seen_min_sharpness) print "min_sharpness = " min_sharpness;
    if (!seen_min_face_area_frac) print "min_face_area_frac = " min_face_area_frac;
  }
}

BEGIN {
  in_auth = 0;
  in_recognition = 0;
  seen_auth = 0;
  seen_recognition = 0;
  seen_auth_frames = 0;
  seen_aggregation_window = 0;
  seen_hysteresis_frames = 0;
  seen_min_sharpness = 0;
  seen_min_face_area_frac = 0;
}

/^[[:space:]]*\[[^]]+\][[:space:]]*$/ {
  flush_auth_missing();
  flush_recognition_missing();
  in_auth = ($0 == "[authentication]");
  in_recognition = ($0 == "[recognition]");
  if (in_auth) {
    seen_auth = 1;
    seen_auth_frames = 0;
  }
  if (in_recognition) {
    seen_recognition = 1;
    seen_aggregation_window = 0;
    seen_hysteresis_frames = 0;
    seen_min_sharpness = 0;
    seen_min_face_area_frac = 0;
  }
  print;
  next;
}

in_auth && /^[[:space:]]*auth_frames[[:space:]]*=/ {
  print "auth_frames = " auth_frames;
  seen_auth_frames = 1;
  next;
}

in_recognition && /^[[:space:]]*aggregation_window[[:space:]]*=/ {
  print "aggregation_window = " aggregation_window;
  seen_aggregation_window = 1;
  next;
}

in_recognition && /^[[:space:]]*hysteresis_frames[[:space:]]*=/ {
  print "hysteresis_frames = " hysteresis_frames;
  seen_hysteresis_frames = 1;
  next;
}

in_recognition && /^[[:space:]]*min_sharpness[[:space:]]*=/ {
  print "min_sharpness = " min_sharpness;
  seen_min_sharpness = 1;
  next;
}

in_recognition && /^[[:space:]]*min_face_area_frac[[:space:]]*=/ {
  print "min_face_area_frac = " min_face_area_frac;
  seen_min_face_area_frac = 1;
  next;
}

{
  print;
}

END {
  flush_auth_missing();
  flush_recognition_missing();
  if (!seen_auth) {
    print "";
    print "[authentication]";
    print "auth_frames = " auth_frames;
  }
  if (!seen_recognition) {
    print "";
    print "[recognition]";
    print "aggregation_window = " aggregation_window;
    print "hysteresis_frames = " hysteresis_frames;
    print "min_sharpness = " min_sharpness;
    print "min_face_area_frac = " min_face_area_frac;
  }
}
' "$CONFIG" > "$tmp"

install -m 0644 "$tmp" "$CONFIG"
rm -f "$tmp"

echo "Updated $CONFIG"
echo "Backup: $backup"
echo "Active auth/recognition latency settings:"
awk '
  /^[[:space:]]*\[(authentication|recognition)\][[:space:]]*$/ { print; in_section = 1; next }
  /^[[:space:]]*\[[^]]+\][[:space:]]*$/ { in_section = 0 }
  in_section && /^[[:space:]]*(auth_frames|aggregation_window|hysteresis_frames|min_sharpness|min_face_area_frac)[[:space:]]*=/ { print }
' "$CONFIG"

if [ "${DOORMAN_SKIP_RESTART:-0}" != "1" ] && command -v systemctl >/dev/null 2>&1; then
  systemctl restart doormand.service
  systemctl is-active doormand.service
fi
