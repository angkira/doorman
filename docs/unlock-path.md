# Face unlock of the screen locker

How a recognized face unlocks the niri lock screen (gtklock, swaylock as
fallback) without polling.

## Design

```
Mod+L -> doorman-lock -> gtklock (background job, PID = $!)
                      -> doorman unlock-watch --locker-pid $!  --(WaitForMatch)--> doormand
                                                                 <--(Matched)-----
         gtklock <--SIGTERM-- unlock-watch
```

1. `doorman-lock` (`packaging/adapters/niri/bin/`) starts the locker as a
   background job of the script, so `$!` is the PID of the real locker. It does
   not use `--daemonize`. If the locker exits within `DOORMAN_LOCKER_SETTLE`
   (0.3 s) the session was already locked: no second locker is started. With
   status 0 that is the end. With a non-zero status (gtklock returns 1 while a
   previous instance holds the lock) the script decides whether the lock is held
   and logs `rc`, the chosen PID and the decision to `lock.log`:
   * a live process of the locker name owned by the current user
     (`pgrep -n -u "$(id -u)" -x gtklock`, newest PID): the watcher is attached
     to that PID (`unlock-watch --locker-pid <pid>`);
   * none, but `loginctl show-session "$XDG_SESSION_ID" -p LockedHint` says
     `yes`: the lock is held by an unknown client; this is logged and no
     watcher is started (there is no PID to SIGTERM);
   * neither: logged, nothing started.

   One watcher at a time: its PID is kept in
   `$XDG_RUNTIME_DIR/doorman/unlock-watch.pid`. A second run finds the live
   watcher, logs `already running` and starts none.
   Checking the pidfile and starting the watcher is one atomic step under a
   non-blocking `flock` on `$XDG_RUNTIME_DIR/doorman/unlock-watch.lock`: when
   two key presses arrive within the settle time and the lock is held, the
   second run logs `holds ... not starting one` and starts no watcher. A
   recorded PID counts as a live watcher only if `/proc/<pid>/cmdline` shows
   `unlock-watch` (or the poll watcher script); a dead PID or a PID reused by
   an unrelated process is stale, logged as `not an unlock watcher` and the
   pidfile is removed. If `flock` is missing the script logs it and falls back
   to the unlocked check-then-start.
   While the locker is alive the script starts the watcher.
2. `doorman unlock-watch` sends `WaitForMatch{username, timeout_ms}` and
   blocks on the socket. One connection replaces one `doorman test` process
   (a full capture session) plus a `sleep 2` per attempt. Calls are re-issued
   while the locker is alive.
3. On `Matched` for its user it sends SIGTERM to that PID (gtklock handles only
   SIGTERM; SIGUSR1 would terminate it uncleanly and is never used), waits
   `--term-grace-ms` (2000), then SIGKILL with a warning. A hard kill can leave
   the session locked without a client, so it is a last resort.
4. It never signals by name. Before every signal it checks that the PID is
   alive and that `/proc/<pid>/comm` equals `--locker-name` (set by the script),
   so a recycled PID is not signalled. PIDs <= 1 and its own PID are refused.

### Daemon side: `WaitForMatch`

* Authorization is identical to `Authenticate` (self, root or `trusted_uids`),
  it counts once per call against the rate limit, and the timeout is capped by
  `[ipc] wait_for_match_max_ms` (default 60000). No unbounded waits.
* The answer is one terminal message: `Matched{username, similarity,
  ts_unix_ms}` or `NoMatch{reason: timeout | not_armed | not_enrolled}`.
* A `Matched` can only come from a complete `Authenticate` run (frame quality
  gate, liveness, aggregated cosine threshold): every attempt calls the same
  `handle_authenticate` the PAM path uses. There is no second, weaker path.
* **Background recognition loop.** One exists: `recognition_pipeline.rs`
  (`User recognized:` log line, `proactive_unlock`), running while a preview
  client is connected or the session is armed. It is not reused because it is
  weaker than `Authenticate`: it scores a rolling window against every enrolled
  user, and its liveness gate runs only when `proactive_unlock` is on. So
  `WaitForMatch` repeats the Authenticate pipeline server-side instead; each
  attempt waits for camera frames, there is no sleep and no busy loop.
* `Matched` is delivered only while the lock supervisor reports `armed`
  (`LockedHint && !IdleHint`). While disarmed the call awaits the gate (no
  polling); if it never arms the answer is `NoMatch{not_armed}`. In
  `--user`/dev mode there is no gate. If logind cannot be reached the
  supervisor pins `armed=false`, so event-driven unlock never fires; use
  `DOORMAN_UNLOCK_MODE=poll` there.
* **Pre-arm** (`[ipc] prearm_camera`, default true). niri sets `LockedHint`
  only after the compositor has fully locked (see "Why the hint lags" below),
  about 1.1 s after the request on the measured host. Instead of idling until
  then, a call that arrives with the gate closed starts the full decision at
  once, so camera open (0.9 s) and frame scoring overlap the lag. Rules:
  * the result is *held*; `Matched` goes out the moment the gate is armed and
    never before (a closed gate can only produce `NoMatch`);
  * frames captured before the request are skipped (same freshness rule as the
    sudo profile), so nothing from before the call is ever scored;
  * the whole pre-armed attempt, including the camera lease, is dropped
    `[ipc] prearm_max_hold_secs` (default 5) after the request if the gate has
    not opened; a result held that long is discarded, not reused, and the call
    then behaves as without pre-arm (waits for the gate, then makes a fresh
    attempt). The camera is released when the call ends, the client
    disconnects, the gate closes after having been open, or the hold expires;
  * the decision function is the unchanged `Authenticate` pipeline.
  `prearm_camera = false` restores "no camera until the gate is armed".
* An attempt in flight is dropped when the session disarms, the deadline
  passes, or the client closes the connection (the helper does so when the
  locker exits), which releases the camera lease.

  Two properties of pre-arm to be aware of (documented, not bugs):
  * **Camera on an unlocked session.** With pre-arm a `WaitForMatch` that
    arrives while the session is still unlocked keeps the camera on for up to
    `prearm_max_hold_secs` (default 5, max 30). The call comes from the user's
    own helper (same peer-credential policy and 20 calls/min limit as
    `Authenticate`, which can already open the camera), but a same-uid process
    can keep the camera on nearly continuously by repeating the call within
    that limit. `prearm_camera = false` removes this.
  * **Freshness is relative to the request, not to the lock.** A frame captured
    after the request but before the lock completed can count, and a held
    result can be up to `prearm_max_hold_secs` old when the lock is confirmed.
    A user who presses lock and walks away within that window can be unlocked
    by their own face seen just before leaving. A hint arriving exactly when the
    hold expires discards the held result (pinned by a test); one millisecond
    earlier delivers it.
* **Failure budget** (`[ipc] face_max_judged_failures`, default 5). Every
  `WaitForMatch` attempt goes through the same decision core as
  `Authenticate`, which counts judged failures (a completed rejection: below
  threshold, liveness rejected, identity floor, ambiguous faces; not camera
  errors, timeouts, no face or an unavailable liveness model) per user. After 5
  within 300 s face auth is suspended for 30 s, then 2 min, then 10 min (the last
  repeats; a match or 30 quiet minutes reset the escalation). While suspended the
  daemon answers `NoMatch{suspended}` (`Denied [face_auth_suspended]` to
  `Authenticate`/PAM) without opening the camera, and `doorman unlock-watch`
  asks again only every 5 s; the locker is never touched, so the password
  works. Attempts inside one call are also spaced by 250 ms, so an instantly
  failing source cannot spin. The audit lines are on the `doorman::audit`
  target. Side effect: someone who fails on purpose can switch the owner's face
  unlock off for the cooldown; the owner then uses the password, and the next
  match resets the budget.

### Why the hint lags (niri + gtklock)

`LockedHint` is set by the compositor, not by gtklock, the script or
doorman: niri 26.04 calls logind `Session.SetLockedHint` from a short-lived
thread only in its fully `Locked` state, i.e. after the lock client has
created and mapped a lock surface on every output and niri has redrawn them
(`niri.rs` `lock()`, `continue_to_locking()`, `update_locked_hint()`; the
1 s timer there is only a fallback). The daemon already receives it as a
D-Bus `PropertiesChanged` event (`lock_supervisor.rs`), so nothing is polled
on the daemon side and nothing there can be sped up. The hint is a real
protection: signalling the locker before niri is `Locked` would leave the
session locked with a dead client. It is therefore never bypassed.

### Timing stages

`journalctl -u doormand | grep "WaitForMatch timing"` gives `prearm`,
`hint_wait_ms` (request until armed) and `total_ms`. With `[telemetry]
timing_log = true` each attempt record also carries `hint_wait_ms` (when the
gate had opened by the end of the attempt), `lock_to_camera_ready_ms` (negative
when pre-arm had the camera ready before the lock), `camera_open_ms`,
`frames_collect_ms` and `decision_ms`. No identity is recorded.

* An attempt in flight is dropped when the session disarms, the deadline
  passes, or the client closes the connection (the helper does so when the
  locker exits), which releases the camera lease.

### Concurrency (IPC server)

`run_server` used to await each connection inline, so one `Authenticate`
blocked every other request. Connections are now served in their own tasks
(`ipc_policy::serve`), capped at 64 open connections, and a peer must send its
request line within 10 s.

Camera contention: the camera has a single owner thread that publishes frames
on a `watch` channel; requests only read that channel and take counted leases
(`capture_requests`). Concurrent captures therefore share frames and cannot
fight over the device, and a failed camera surfaces as the explicit error
`Camera not available` after the 8 s lease wait, not a deadlock. What is shared
is CPU: ONNX sessions come from a round-robin pool behind `std::sync::Mutex`
and inference runs inside async tasks, so many simultaneous recognitions
queue on the sessions and each takes longer. Authenticate and `WaitForMatch`
are rate limited per uid, which bounds this.

`DOORMAN_WATCH_EARLY` (start the watcher before the settle wait) applies to
`DOORMAN_UNLOCK_MODE=event` only. The poll watcher has no lock gate, so it is
always started after the settle wait.

## `doorman test`

`doorman test NAME` prints `recognized: NAME (...)` or `!recognized: NAME
(reason)`; those lines are stable. Exit codes: 0 recognized, 1 error (daemon
unreachable, denied, rate limited, unexpected answer), 2 not recognized,
rejected, unavailable or suspended. `doorman --json test NAME` prints one JSON
object instead (also for errors): `decision` (`recognized`, `not_recognized`,
`suspended`, `error`), `reason_code` (`matched`, `not_matched`,
`liveness_rejected`, `liveness_unavailable`, `identity_floor`,
`ambiguous_faces`, `insufficient_frames`, `no_clear_face`, `no_frames`,
`not_enrolled`, `face_auth_suspended`, `denied`, `rate_limited`,
`daemon_error`, `other_failure`), `similarity_bucket` (0.05 steps),
`liveness_ok` (true/false/null), `frames_judged` (null for now: the wire answer
does not carry it), `duration_ms`, `suspended_until` (RFC 3339, only when
suspended). No frames, embeddings or exact scores.

## Exit codes of `doorman unlock-watch`

| Code | Meaning | Locker touched |
|---|---|---|
| 0 | Matched, locker exited after SIGTERM | yes |
| 10 | Locker exited by itself first (password etc.) | no |
| 11 | Daemon socket unreachable | no |
| 12 | Daemon denied the request | no |
| 13 | Daemon rate limited the request | no |
| 14 | Unsupported protocol version | no |
| 15 | PID invalid or not the expected process | no |
| 16 | Unexpected daemon response | no |
| 17 | Matched, SIGTERM ignored, SIGKILL removed the locker | yes |
| 18 | Matched, but signalling failed or the locker survived SIGKILL | yes |
| 19 | User not enrolled | no |

Logs are timestamped lines on stderr; the script appends them to
`$XDG_RUNTIME_DIR/doorman/lock.log` with the locker PID and the decision.
The helper is not restarted if it exits (for example 11 after a daemon
restart); the password always remains available.

## Fallback

`DOORMAN_UNLOCK_MODE=poll` makes `doorman-lock` start
`doorman-lock-watch-poll`, the old watcher (a `doorman test` per iteration,
`pkill -x` by name). Keep it until the event path is validated on the host.
`DOORMAN_LOCKER_NAME` overrides the expected process name when the locker is
started through a wrapper.

Each `WaitForMatch` attempt that fails still writes the usual attempt record
and, with `keep_raw_frames`, a capture pack (capped per day), as repeated
`doorman test` calls did.

## Alternatives not taken

* **logind `Session.Unlock()`** (`lock_supervisor.rs`, `proactive_unlock`).
  With `ext-session-lock-v1` the lock client itself must destroy the lock, so
  a logind signal cannot unlock a locker that does not act on it
  **[unverified]**: not prototyped here (no Wayland session available). The
  path stays as it is, off by default.
* **`pam_doorman` in gtklock.** gtklock runs PAM only after the user submits
  input, so PAM cannot unlock by face on its own. Dropped.
* **Watching logind `LockedHint`** for the end of the lock: locker PID liveness
  is sufficient, so it was skipped.

## Host validation checklist

- [ ] `Mod+L` locks; the face unlocks; `lock.log` shows locker PID, `matched`,
      `sent SIGTERM`, `locker exited after SIGTERM`.
- [ ] Measure recognized-to-unlocked from the log timestamps (target <= 0.6 s).
- [ ] Typing the password still works and the watcher exits with 10.
- [ ] Locking twice in a row: the second run starts no second locker or watcher
      (`lock.log`: `rc=1` ... `already running`, or `already locked`);
      `pgrep -fc unlock-watch` is 1.
- [ ] Host case: gtklock returns rc=1 while niri reports the session locked by
      an active client: `lock.log` shows `rc=1`, the PID of the live gtklock and
      `attaching watcher`, and the face unlock still works.
- [ ] Camera unplugged: the lock screen stays responsive and the password works.
- [ ] Screen idle (display off) then wake: `not_armed` waits, unlock still works.
- [ ] `gtklock` comm equals `gtklock` (`cat /proc/$(pgrep -x gtklock)/comm`);
      with a wrapper set `DOORMAN_LOCKER_NAME`.
- [ ] `DOORMAN_UNLOCK_MODE=poll` still unlocks.
