#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-/etc/doorman/doorman.toml}"
MAX_ATTEMPTS="${DOORMAN_DATASET_MAX_ATTEMPTS:-5000}"

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
awk -v max_attempts="$MAX_ATTEMPTS" '
function flush_dataset_missing() {
  if (in_dataset) {
    if (!seen_logging) print "logging_enabled = true";
    if (!seen_max) print "max_attempt_records_per_day = " max_attempts;
  }
}

BEGIN {
  in_dataset = 0;
  seen_dataset = 0;
  seen_logging = 0;
  seen_max = 0;
}

/^[[:space:]]*\[[^]]+\][[:space:]]*$/ {
  flush_dataset_missing();
  in_dataset = ($0 == "[dataset]");
  if (in_dataset) {
    seen_dataset = 1;
    seen_logging = 0;
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
