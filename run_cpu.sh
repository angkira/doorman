#!/usr/bin/env bash
# run_cpu.sh — Launch doorman daemon under a systemd transient user scope
# with cgroups v2 CPU and memory caps. CPU-only inference mode.
#
# Ryzen 8700G (8C/16T): defaults cap the daemon to 2 full cores (200%)
# and 1.5 GB RSS so it cannot starve the desktop under load.
#
# Usage:
#   ./run_cpu.sh [--show] [extra doormand args...]
#
#   --show   Print effective cgroup caps for any running doorman-cpu.scope
#            and exit without launching.
#
# Env overrides (set before calling):
#   DOORMAN_CPU_QUOTA   default 200%     (200% = 2 full cores on a 100%-per-core system)
#   DOORMAN_MEM_MAX     default 1500M    (hard OOM kill threshold)
#   DOORMAN_MEM_HIGH    default 1200M    (soft throttle / reclaim pressure)
#   DOORMAN_CPU_WEIGHT  default 20       (relative scheduler weight; 100 = normal)
#   DOORMAN_CONFIG      default doorman.toml
#
# CONFIG NOTE 2026-06-06: default config is doorman.toml (the canonical, tracked
# config). Its backend="tract" and models_dir/socket_path system paths do NOT
# matter here: all backend strings map to the single ORT backend in code, and the
# --user flag overrides socket_path/data_dir/models_dir to XDG paths AFTER the
# config is loaded — so the daemon uses $XDG_RUNTIME_DIR/doorman.sock and
# ~/.local/share/doorman/models regardless. We deliberately do NOT use
# doorman-lightweight.toml: its "ort-lightweight" backend is dead code (unwired),
# and it carries hardcoded paths / SCRFD-w600k model refs that the code ignores.
# The config file only effectively supplies camera params + auth thresholds.
# The CLI (doorman binary) reads $XDG_RUNTIME_DIR automatically and connects
# to the same socket without any extra flags.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (all overridable via environment)
# ---------------------------------------------------------------------------
DOORMAN_CPU_QUOTA="${DOORMAN_CPU_QUOTA:-200%}"
DOORMAN_MEM_MAX="${DOORMAN_MEM_MAX:-1500M}"
DOORMAN_MEM_HIGH="${DOORMAN_MEM_HIGH:-1200M}"
DOORMAN_CPU_WEIGHT="${DOORMAN_CPU_WEIGHT:-20}"
DOORMAN_CONFIG="${DOORMAN_CONFIG:-doorman.toml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/target/release/doormand"
SCOPE_UNIT="doorman-cpu"

# ---------------------------------------------------------------------------
# --show mode: inspect running scope and exit
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--show" ]]; then
    printf 'Effective cgroup properties for %s.scope:\n' "$SCOPE_UNIT"
    systemctl --user show "${SCOPE_UNIT}.scope" \
        -p CPUQuotaPerSecUSec \
        -p MemoryMax \
        -p MemoryHigh \
        -p CPUWeight \
        -p TasksMax
    exit 0
fi

# ---------------------------------------------------------------------------
# Pre-flight: binary must exist
# ---------------------------------------------------------------------------
if [[ ! -x "$BINARY" ]]; then
    printf 'ERROR: binary not found or not executable: %s\n' "$BINARY" >&2
    printf 'Build it first: cargo build --release -p doorman\n' >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Launch inside transient user scope
# ---------------------------------------------------------------------------
printf 'Launching doormand under systemd user scope %s.scope\n' "$SCOPE_UNIT"
printf '  CPUQuota    = %s\n'  "$DOORMAN_CPU_QUOTA"
printf '  MemoryMax   = %s\n'  "$DOORMAN_MEM_MAX"
printf '  MemoryHigh  = %s\n'  "$DOORMAN_MEM_HIGH"
printf '  CPUWeight   = %s\n'  "$DOORMAN_CPU_WEIGHT"
printf '  Config      = %s\n'  "$DOORMAN_CONFIG"
printf '\n'

systemd-run --user --scope --unit="$SCOPE_UNIT" \
    -p CPUQuota="$DOORMAN_CPU_QUOTA" \
    -p MemoryMax="$DOORMAN_MEM_MAX" \
    -p MemoryHigh="$DOORMAN_MEM_HIGH" \
    -p CPUWeight="$DOORMAN_CPU_WEIGHT" \
    -p TasksMax=256 \
    "$BINARY" --user --device cpu --config "$DOORMAN_CONFIG" "$@"

# ---------------------------------------------------------------------------
# Post-launch hints (printed after daemon exits or is backgrounded)
# ---------------------------------------------------------------------------
printf '\nInspect / manage the scope:\n'
printf '  systemctl --user status %s.scope\n'     "$SCOPE_UNIT"
printf '  systemd-cgtop\n'
printf '  systemctl --user stop %s.scope\n'       "$SCOPE_UNIT"
