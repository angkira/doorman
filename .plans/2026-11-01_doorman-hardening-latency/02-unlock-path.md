# WS2 — Unlock path: event-driven, in-repo, no polling (P0, effort M, depends on WS1)

## Problem
Face unlock on niri/gtklock is implemented by two host-local shell scripts (`~/.local/bin/doorman-lock`, `doorman-lock-watch`, described in `00-context.md`). The watcher polls: each iteration spawns `doorman test <user>` (a full new capture session), sleeps 2 s, and finally `pkill`s the locker. Cost ≈ 2.5 s per unlock plus fragility (watcher never started when `gtklock --daemonize` returned rc=1 although the lock was held). The scripts are not in the repo, not tested.

## Design (primary)
Daemon-side, user-space-side:
1. **Daemon pushes the decision.** Add a request `WaitForMatch{username, timeout_ms}` (or `Subscribe`) on the IPC socket (protocol version from WS1). When the session is armed, the daemon already runs a background recognition loop (log lines `User recognized: …`). Make that loop publish a `Matched{username, similarity, ts}` event to subscribers that pass the WS1 policy (peer uid == username). One long-lived connection replaces N × `doorman test`.
2. **Thin user-space helper** `doorman-unlockd` (small Rust bin in the workspace, or a mode of the `doorman` CLI: `doorman unlock-watch --locker gtklock`), started by `doorman-lock`:
   - connects, waits for `Matched`, then asks the locker to quit gracefully: `SIGTERM` to the locker process it started (gtklock handles only SIGTERM; **do not use SIGUSR1** — unhandled → terminate), wait up to 2 s for exit, only then `SIGKILL` as last resort and log a warning (a hard kill may leave the session locked without a client).
   - exits if the locker disappears (user typed the password) or the session is unlocked (watch logind `LockedHint` via D-Bus/zbus or poll `loginctl` at low frequency).
3. **Locker start script** `doorman-lock` moves into the repo (`packaging/adapters/niri/bin/doorman-lock`): start locker; treat "already locked" as success (check `LockedHint` / niri `refusing lock … active client` collision); **start the helper whenever a locker process is alive**, regardless of the locker's exit code; keep `/tmp/doorman-lock.log`-style logging but move to `$XDG_RUNTIME_DIR/doorman/lock.log`; no `--daemonize` if a foreground+background pattern gives a reliable PID.
4. Keep the polling watcher only as a fallback behind a flag until the new path is validated on the host.

## Alternatives to evaluate and record (do not just pick silently)
- logind `Session.Unlock()` via the existing `lock_supervisor.rs proactive_unlock`: **[unverified]** whether gtklock/swaylock/niri honour the logind Unlock signal; ext-session-lock-v1 requires the lock client itself to unlock. Prototype only if cheap; document the result.
- `pam_doorman` inside gtklock: gtklock only runs PAM after the user submits input, so it cannot auto-unlock by face. Document and drop.

## Tasks
- Implement `WaitForMatch` in daemon + client lib (`shared` request/response types, serde tests).
- `doorman-unlockd` / CLI mode with graceful unlock + timeouts + clear exit codes.
- Move/port scripts into `packaging/adapters/niri/` and `scripts/`, with `shellcheck`-clean bash, `$XDG_RUNTIME_DIR`, no personal paths.
- Update `scripts/enable_local_niri_sddm.sh` / `install_local_niri_desktop.sh` to install them.
- Add a **fake locker** test double (a tiny script that traps SIGTERM and exits) and a **fake daemon** (mock IPC server emitting `Matched`) so the helper's behaviour is testable in CI without Wayland.

## Tests
- Helper: matched → SIGTERM sent → fake locker exits → exit 0; locker ignores SIGTERM → SIGKILL after timeout with warning; locker exits first → helper exits; daemon unreachable → helper exits non-zero without touching the locker; mismatched username → ignored.
- Daemon: subscriber receives exactly one `Matched` per lock session; second subscriber with wrong uid is denied (WS1); no event while not armed; no busy loop.
- Script tests (existing `tests/test_local_sway_scripts.py` style): `bash -n`, shellcheck, behaviour with stubbed `gtklock`/`loginctl` on PATH.

## Acceptance
- Removing `sleep 2` + per-iteration process spawn: median recognised→unlocked delay ≤ 0.6 s on the host (measure with timestamps in the helper log).
- Helper starts even when the locker returns non-zero but is alive.
- No `pkill -x` by name remains on the happy path (use the PID the script recorded).

## Host validation checklist (human)
Press `Mod+L`; face unlock works; typing the password still works and the helper exits; locking twice in a row; unplug camera → falls back to password without hanging the lock screen; check `$XDG_RUNTIME_DIR/doorman/lock.log` shows PIDs and the decision.
