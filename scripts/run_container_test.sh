#!/usr/bin/env bash
# run_container_test.sh — in-container Linux validation harness for doorman.
#
# Runs ENTIRELY inside the Dockerfile.test image (Ubuntu 24.04). It:
#   1. Starts doormand (--user) headless against a looping "face video"
#      (personA) as its camera source.
#   2. Enrolls user "alice" from that source via the `doorman` CLI.
#   3. Runs `doorman test alice` -> expects RECOGNIZED (genuine match).
#   4. Restarts the daemon pointed at an IMPOSTOR video (personB), keeping the
#      same data dir, and runs `doorman test alice` -> expects NOT recognized.
#
# Exit code 0 == all assertions green. Non-zero == a step failed (the daemon
# log is dumped for diagnosis).
set -uo pipefail

BIN_DAEMON=/build/target/release/doormand
BIN_CLI=/build/target/release/doorman
export XDG_RUNTIME_DIR=/run/doorman
export XDG_DATA_HOME=/root/.local/share
mkdir -p "$XDG_RUNTIME_DIR"

VIDEO_A=/test/personA.mp4
VIDEO_B=/test/personB.mp4
LOG=/test/daemon.log

PASS=0
FAIL=0
note()  { echo "  -> $*"; }
green() { echo "[ OK ] $*"; PASS=$((PASS+1)); }
red()   { echo "[FAIL] $*"; FAIL=$((FAIL+1)); }

start_daemon() {
    local video="$1"
    rm -f "$XDG_RUNTIME_DIR"/doorman*.sock
    # --start-unlocked + --preview: keep the pipeline streaming and the debug
    # socket open without auto-locking a (nonexistent) desktop session.
    RUST_LOG=doormand=info "$BIN_DAEMON" --user --preview --start-unlocked \
        --video-file "$video" >"$LOG" 2>&1 &
    DAEMON_PID=$!
    # Wait for the command socket to appear (models load + warmup can take a bit).
    for _ in $(seq 1 60); do
        [ -S "$XDG_RUNTIME_DIR/doorman.sock" ] && break
        if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
            echo "daemon exited early; log:"; cat "$LOG"; return 1
        fi
        sleep 1
    done
    [ -S "$XDG_RUNTIME_DIR/doorman.sock" ] || { echo "socket never appeared"; cat "$LOG"; return 1; }
    sleep 2  # let warmup + first frames flow
    return 0
}

stop_daemon() {
    [ -n "${DAEMON_PID:-}" ] && kill "$DAEMON_PID" 2>/dev/null
    wait "${DAEMON_PID:-}" 2>/dev/null
    DAEMON_PID=""
}

echo "================ doorman Linux container test ================"
echo "arch: $(uname -m)"
# NOTE: doormand has no --help (it parses args manually and otherwise starts the
# daemon), so we never probe it that way — it would block forever.

# --- Phase 1: genuine enroll + match (personA video) ------------------------
echo
echo "### Phase 1: enroll + genuine match (source = personA)"
start_daemon "$VIDEO_A" || exit 1

note "status:"
"$BIN_CLI" status || true

note "enrolling alice..."
ENROLL_START=$(date +%s)
if "$BIN_CLI" enroll alice; then
    ENROLL_END=$(date +%s)
    green "enroll alice succeeded (${ENROLL_END}-${ENROLL_START}s wall: $((ENROLL_END-ENROLL_START))s)"
else
    red "enroll alice failed"
fi

note "list (expect num_embeddings > 1):"
LIST_JSON=$("$BIN_CLI" --json list)
echo "$LIST_JSON"
NEMB=$(echo "$LIST_JSON" | grep -oE '"num_embeddings"[[:space:]]*:[[:space:]]*[0-9]+' | head -1 | grep -oE '[0-9]+')
if [ "${NEMB:-0}" -gt 1 ]; then
    green "num_embeddings reported as ${NEMB} (>1, bug fixed)"
else
    red "num_embeddings reported as ${NEMB:-missing} (expected >1)"
fi

note "test alice against genuine video (expect: recognized):"
OUT=$("$BIN_CLI" test alice)
echo "$OUT"
if echo "$OUT" | grep -q '^recognized:'; then
    green "genuine match recognized"
else
    red "genuine match NOT recognized (expected recognized)"
fi

stop_daemon

# --- Phase 2: impostor reject (personB video, same data dir) ----------------
echo
echo "### Phase 2: impostor reject (source = personB, alice still enrolled)"
start_daemon "$VIDEO_B" || exit 1

note "test alice against impostor video (expect: !recognized):"
OUT=$("$BIN_CLI" test alice)
echo "$OUT"
if echo "$OUT" | grep -q '^!recognized:'; then
    green "impostor correctly rejected"
else
    red "impostor NOT rejected (expected !recognized)"
fi

stop_daemon

# --- Summary ----------------------------------------------------------------
echo
echo "================ summary: ${PASS} passed, ${FAIL} failed ================"
echo "---- daemon detection FPS lines (last run) ----"
grep -i "Detection processing" "$LOG" || echo "(no FPS lines captured in last run window)"

[ "$FAIL" -eq 0 ]
