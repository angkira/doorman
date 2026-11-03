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
* Attempts run only while the lock supervisor reports `armed`
  (`LockedHint && !IdleHint`). While disarmed the call awaits the gate (no
  polling), makes no attempt, and the camera is not switched on; if it never
  arms the answer is `NoMatch{not_armed}`. In `--user`/dev mode there is no
  gate. If logind cannot be reached the supervisor pins `armed=false`, so
  event-driven unlock never fires; use `DOORMAN_UNLOCK_MODE=poll` there.
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
