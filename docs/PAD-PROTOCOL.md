# Presentation-attack screening (owner protocol)

This page is for the owner of an installation who wants numbers, not opinions,
about how well doorman resists a photo or a replay of their own face.

```
sudo doorman calibrate      # 1. start a 45 minute measuring session (root, self-ending)
doorman pad run             # 2. guided run; no root, no flags
                            # 3. done: calibration ends by itself
```

`doorman pad run` is the native replacement of `scripts/pad_protocol.py`, which
stays in the repository but is **deprecated**; use `doorman pad run`.

**It is screening, not certification.** About 100 attempts with one phone, one
print and your own room cannot prove a low attack-acceptance rate; they can only
show a high one. Treat a clean result as "no weakness found with these
instruments", never as "safe".

## What it measures

Each attempt sends the same IPC request as `doorman test` (the same decision code
as the lock screen and PAM) and records only: class, index, decision, reason code,
similarity bucket (rounded to 0.05), liveness flag, duration and time.

| Class | Kind | What you present |
|---|---|---|
| `genuine` | bona fide | yourself, normal conditions (10 attempts) |
| `phone_photo` | attack | a photo of your face on a phone screen |
| `phone_replay` | attack | a video of your face on a phone |
| `print_matte` | attack | a printed photo of your face, about 8-10 cm wide |
| `mixed_track` | attack | your live face plus a photo/phone with a face beside it |
| `genuine_dark`, `genuine_backlight` | bona fide | dark room, backlight (5 each) |
| `genuine_glasses_on`, `genuine_glasses_off` | bona fide | with and without glasses (5 each) |
| `genuine_yaw_left`, `genuine_yaw_right` | bona fide | head turned about 20 degrees (5 each) |
| `genuine_near`, `genuine_far` | bona fide | about 30 cm and about 1 m (5 each) |

The attack classes have 10 attempts each by default. The `phone_photo`,
`phone_replay` and `print_matte` names are the species names of
`doorman-bench pad-report`.

## Running it (about 20 minutes)

You need: a phone with a photo and a video of your face (brightness at maximum),
one printed photo, and the daemon running with your face enrolled.

1. **Start calibration mode** (once, needs root):

   ```
   sudo doorman calibrate            # default 45 minutes; --ttl 1h30m for longer (max 120m)
   ```

   Calibration mode changes exactly two things for its duration: the failure
   budget is frozen (rejections do not count toward a suspension, so the session
   has no pauses) and shadow PAD logging is on. Every decision rule is the same.
   See "Calibration mode" below.
2. **Run the protocol** (no root):

   ```
   doorman pad run                   # 10 genuine + 10 per attack class, 5 per condition
   doorman pad run --quick           # smoke test: 3 genuine, 2 per attack class, 2 per condition
   ```

   For each class it prints the instruction; Enter starts it (type `s` and Enter
   to skip a class, for example glasses if you have none). Per attempt: press
   Enter, a 3 second countdown, the attempt, the outcome. Ctrl-C stops cleanly
   and the partial results are still written.
3. It prints a table and writes (mode 0600) `~/doorman-pad-<date>.json`, `.csv` and
   `-bench.csv` (`--out PREFIX` changes the location; `--user NAME` tests another
   enrolled user, default is you).
4. Nothing to restore. Calibration ends by itself after the ttl; end it earlier
   with `sudo doorman calibrate --end`. `doorman status` and `doorman calibrate
   --status` show whether it is on and for how long.

Without calibration mode the run still works: it starts by printing
`Для 20-минутной сессии без пауз: sudo doorman calibrate` and continues, after
you press Enter, in paced mode. Every rejection then counts toward the failure
budget; when face auth is suspended it says why and waits out the cooldown
before repeating the attempt (the refused run is not recorded). Failures from
other callers (the lock screen) are invisible to it. If calibration mode ends
in the middle of a run it says so and warns five minutes before.

An infrastructure error (daemon unreachable, timeout) is recorded, excluded from
the rates and the attempt is repeated; three in a row stop the run (results so
far are still written).

Keep `[dataset] keep_raw_frames = false` (the default) for the session, otherwise
the daemon stores the attack frames and yours. Calibration mode does not change
it and never stores frames or embeddings.

## Calibration mode

`sudo doorman calibrate [--ttl 45m]` sends `Calibrate{ttl_secs}` to the daemon.

* **Root only**: the daemon accepts it solely from a peer whose kernel-reported
  uid (`SO_PEERCRED`) is 0. `[ipc] admin_uids` and `trusted_uids` do not count.
  Anyone else gets `denied: calibration mode requires root (uid 0): run it with
  sudo`. `doorman calibrate` as a non-root user does not even connect; it prints
  the sudo line. `doorman calibrate --status` works for every user.
* **Time-limited**: default 45 minutes (`[calibration] default_ttl_secs`), at most
  120 minutes (`max_ttl_secs`, a hard ceiling that a config cannot raise), at
  least 1 minute. A longer request is refused with the limit in the message.
* **Self-expiring and never persisted**: active means "now < expiry"; the daemon
  does not need a timer to end it. It lives in memory only, so a daemon restart
  ends it. Re-arming needs a new request (it replaces the session).
* **What it relaxes**: (1) shadow PAD logging is forced on, through the same
  collector as `[liveness] shadow_pad_logging`; (2) the failure budget is frozen:
  nothing is enforced, counted or reset (an earlier suspension is not enforced
  during the session and applies again afterwards if it has not run out). Judged
  failures are still logged and counted in a separate calibration counter
  (shown by `doorman calibrate --status`). On expiry or end nothing has to be
  put back, because nothing was rewritten: the budget state is exactly what it
  was.
* **What it does not touch**: thresholds, `auth_frames`, the depth threshold,
  liveness enforcement, N-of-M, identity floor, multi-face policy, minimum frames,
  model integrity, the sudo profile's own suspension, `keep_raw_frames`. A test
  compares whole decisions with calibration on and off.
* **Loud**: WARN at start (`CALIBRATION MODE ACTIVE: failure budget suspended`),
  the same WARN every `warn_interval_secs` (default 5 minutes) while active, INFO
  at end and at expiry, a `CALIBRATION` line in `doorman status`, and
  `"calibration": true` in `doorman --json test` while it is on. Look in the
  journal: `journalctl -u doormand | grep -i calibration`.

## Why the budget is frozen, and the risk it carries

While calibration mode is on, anyone who can reach the camera can try faces
without the lockout. That is the point of the session and the reason it is root
only, short and self-ending. The password path is unaffected. Do not leave it on
unattended; `sudo doorman calibrate --end` when you are done.

## Reading the numbers

* **APCER** (attack class): attacks accepted / attacks that got a decision.
  **BPCER** (genuine class): genuine presentations rejected / those that got a
  decision. "No face found" and "no frames" are counted apart as non-responses
  (column `nr`); they are not accepts, and for genuine classes `bpcer_with_nr` in
  the JSON counts them as failures too. Errors are excluded and listed.
* **95% upper** is the exact one-sided Clopper-Pearson upper bound of the rate.
  With zero errors it is `1 - 0.05^(1/n)`: 0 of 10 allows a true rate up to
  25.9%, 0 of 20 up to 13.9%, 0 of 30 up to 9.5%, 0 of 100 up to 2.95% (the
  two-sided 97.5% version of the last one is 3.6%; the script reports the
  one-sided figure). To argue for 1% you need about 300 clean attempts per class.
* **One accepted attack in ten is already a finding**: the point estimate is 10%
  and the bound 39%. Look at its reason codes and, with shadow logging, at the
  depth margin and MiniFASNet score of that attempt.
* A genuine rejection with `liveness_rejected` and a depth margin close to zero
  means the depth threshold (0.05) is too tight for your camera and lighting.
* What the numbers do **not** mean: they say nothing about masks, 3D prints,
  videos on a better screen than yours, other cameras, other people's faces
  (lookalikes), or attackers who learn from earlier attempts.

`doorman-bench pad-report --scores ~/doorman-pad-<date>-bench.csv --threshold 0.5`
recomputes the per-class APCER/BPCER from the `-bench.csv` file (score 1.0 =
accepted, 0.0 = rejected, no-face rows are non-responses) as an independent check
of the arithmetic. The bounds are computed exactly (Clopper-Pearson, bisection
on the binomial CDF) and tested against reference beta quantiles, for example
0 of 10 gives 25.89% one-sided and 30.85% two-sided, 0 of 30 gives 9.50% and
11.57%, 0 of 100 gives 2.95% and 3.62%.

## Privacy

`doorman pad run` never captures, stores or shows a frame and writes no screenshots. Its
files hold decisions, reason codes, buckets and timestamps. They do not hold the
user name. Keep them out of the repository like any measurement from your own
face.

## Shadow PAD logging (optional, off by default)

`[liveness] shadow_pad_logging = true` makes the daemon append one JSON line per
authentication attempt to `[telemetry] shadow_pad_log_path` (default
`/run/doorman/shadow-pad.jsonl`, mode 0640, rotated to `.1` at
`timing_log_max_bytes`). The record holds aggregates only:

* decision, reason code, duration, decision mode, frame counters
  (seen, judged, and dropped for each reason: stale, no face, detector error,
  several faces, quality gate, embedding error);
* per-frame similarity: min / p10 / median / max, the final score, the threshold
  and the margin;
* depth relief of every frame the depth model ran on (min / p10 / median / max)
  and the margin to the threshold;
* the MiniFASNet real-minus-spoof logit of the deciding frame (state `scored`,
  `model_not_loaded`, `error`, `timeout` or `no_frame`);
* quality of the deciding frame: face area fraction, sharpness, face brightness,
  and a yaw proxy from the landmarks.

Every record also has `source` (`test` for `doorman test` and PAM logins, `sudo`,
or `lock_screen`) and `calibration` (true when calibration mode was on).
Lock-screen attempts (`WaitForMatch`, including every pre-arm retry) are **not
logged** unless `[liveness] shadow_pad_include_lockscreen = true` (default
false), so a measuring session is not drowned by lock-screen traffic.

No frame, embedding, user name or uid is written. The record is **shadow only**:
it is built after the decision exists, the MiniFASNet pass runs in a detached
task, and a failure, panic or missing model changes only the record. The decision
is identical with the flag on and off (tests in `daemon/src/ipc/real_path_tests.rs`
run the same sequences both ways, and with injected panics).

Cost: collecting is a few pushes per frame. The extra MiniFASNet pass (128x128
input, a model of about 1.8 MB) is estimated at 5-20 ms on CPU from the model
size; this is an estimate, not a measurement. It happens once, on the last frame
that passed the quality gate, after the response is built. Enable
`[telemetry] timing_log` as well if you want measured stage times.

The file is `doorman:doorman` 0640 in a world-listable directory, so you can see
that it exists without being able to read it. `doorman pad run` opens it for
reading (a successful `stat` is not enough) and, if that works, attaches the
matching record's depth margin, MiniFASNet score and median similarity to each
attempt. If it cannot read it, it says so at the end and how to read it:
`sudo cat /run/doorman/shadow-pad.jsonl` (match by timestamp). With calibration
mode you do not need the config flag at all.

To turn the config flag off again: `sudo doorman config set liveness.shadow_pad_logging false`,
restart `doormand`, and delete the log file if you do not want to keep it.
Calibration mode needs none of this.

## The old script

`scripts/pad_protocol.py` still works but is deprecated in favour of `doorman pad
run` (no Python, no flags, calibration-aware, shadow-log readability checked
correctly). Its budget pacing options (`--budget-max`) exist only there.
