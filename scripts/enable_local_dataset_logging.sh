#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-/etc/doorman/doorman.toml}"
MAX_ATTEMPTS="${DOORMAN_DATASET_MAX_ATTEMPTS:-5000}"
KEEP_RAW_FRAMES="${DOORMAN_KEEP_RAW_FRAMES:-true}"
RAW_FRAME_BUNDLE_SIZE="${DOORMAN_RAW_FRAME_BUNDLE_SIZE:-10}"
RAW_FACE_FRAME_BUNDLE_SIZE="${DOORMAN_RAW_FACE_FRAME_BUNDLE_SIZE:-120}"
MAX_CAPTURE_PACKS="${DOORMAN_MAX_CAPTURE_PACKS_PER_DAY:-500}"
RAW_FRAME_JPEG_QUALITY="${DOORMAN_RAW_FRAME_JPEG_QUALITY:-92}"

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

if [ "$(id -u)" -ne 0 ] && [ ! -w "$CONFIG" ]; then
  echo "Run with sudo: sudo $0 [config-path]" >&2
  exit 1
fi

backup="${CONFIG}.bak-dataset-$(date +%Y%m%d-%H%M%S)"
cp -a "$CONFIG" "$backup"

tmp="$(mktemp)"
awk \
  -v max_attempts="$MAX_ATTEMPTS" \
  -v keep_raw_frames="$KEEP_RAW_FRAMES" \
  -v raw_frame_bundle_size="$RAW_FRAME_BUNDLE_SIZE" \
  -v raw_face_frame_bundle_size="$RAW_FACE_FRAME_BUNDLE_SIZE" \
  -v max_capture_packs="$MAX_CAPTURE_PACKS" \
  -v raw_frame_jpeg_quality="$RAW_FRAME_JPEG_QUALITY" '
function flush_dataset_missing() {
  if (in_dataset) {
    if (!seen_logging) print "logging_enabled = true";
    if (!seen_keep_raw) print "keep_raw_frames = " keep_raw_frames;
    if (!seen_bundle_size) print "raw_frame_bundle_size = " raw_frame_bundle_size;
    if (!seen_face_bundle_size) print "raw_face_frame_bundle_size = " raw_face_frame_bundle_size;
    if (!seen_max_capture) print "max_capture_packs_per_day = " max_capture_packs;
    if (!seen_jpeg_quality) print "raw_frame_jpeg_quality = " raw_frame_jpeg_quality;
    if (!seen_max) print "max_attempt_records_per_day = " max_attempts;
  }
}

BEGIN {
  in_dataset = 0;
  seen_dataset = 0;
  seen_logging = 0;
  seen_keep_raw = 0;
  seen_bundle_size = 0;
  seen_face_bundle_size = 0;
  seen_max_capture = 0;
  seen_jpeg_quality = 0;
  seen_max = 0;
}

/^[[:space:]]*\[[^]]+\][[:space:]]*$/ {
  flush_dataset_missing();
  in_dataset = ($0 == "[dataset]");
  if (in_dataset) {
    seen_dataset = 1;
    seen_logging = 0;
    seen_keep_raw = 0;
    seen_bundle_size = 0;
    seen_face_bundle_size = 0;
    seen_max_capture = 0;
    seen_jpeg_quality = 0;
    seen_max = 0;
  }
  print;
  next;
}

in_dataset && /^[[:space:]]*logging_enabled[[:space:]]*=/ {
  print "logging_enabled = true";
  seen_logging = 1;
  next;
}

in_dataset && /^[[:space:]]*keep_raw_frames[[:space:]]*=/ {
  print "keep_raw_frames = " keep_raw_frames;
  seen_keep_raw = 1;
  next;
}

in_dataset && /^[[:space:]]*raw_frame_bundle_size[[:space:]]*=/ {
  print "raw_frame_bundle_size = " raw_frame_bundle_size;
  seen_bundle_size = 1;
  next;
}

in_dataset && /^[[:space:]]*raw_face_frame_bundle_size[[:space:]]*=/ {
  print "raw_face_frame_bundle_size = " raw_face_frame_bundle_size;
  seen_face_bundle_size = 1;
  next;
}

in_dataset && /^[[:space:]]*max_capture_packs_per_day[[:space:]]*=/ {
  print "max_capture_packs_per_day = " max_capture_packs;
  seen_max_capture = 1;
  next;
}

in_dataset && /^[[:space:]]*raw_frame_jpeg_quality[[:space:]]*=/ {
  print "raw_frame_jpeg_quality = " raw_frame_jpeg_quality;
  seen_jpeg_quality = 1;
  next;
}

in_dataset && /^[[:space:]]*max_attempt_records_per_day[[:space:]]*=/ {
  print "max_attempt_records_per_day = " max_attempts;
  seen_max = 1;
  next;
}

{
  print;
}

END {
  flush_dataset_missing();
  if (!seen_dataset) {
    print "";
    print "[dataset]";
    print "logging_enabled = true";
    print "keep_raw_frames = " keep_raw_frames;
    print "raw_frame_bundle_size = " raw_frame_bundle_size;
    print "raw_face_frame_bundle_size = " raw_face_frame_bundle_size;
    print "max_capture_packs_per_day = " max_capture_packs;
    print "raw_frame_jpeg_quality = " raw_frame_jpeg_quality;
    print "max_attempt_records_per_day = " max_attempts;
  }
}
' "$CONFIG" > "$tmp"

install -m 0644 "$tmp" "$CONFIG"
rm -f "$tmp"

echo "Updated $CONFIG"
echo "Backup: $backup"

if [ "${DOORMAN_SKIP_RESTART:-0}" != "1" ] && command -v systemctl >/dev/null 2>&1; then
  systemctl restart doormand.service
  systemctl is-active doormand.service
fi
