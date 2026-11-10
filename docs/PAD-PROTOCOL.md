# Presentation-attack screening (owner protocol)

This page is for the owner of an installation who wants numbers, not opinions,
about how well doorman resists a photo or a replay of their own face.
It describes the screening script `scripts/pad_protocol.py`, the optional
daemon-side "shadow PAD log", and how to read the results.

**It is screening, not certification.** About 100 attempts with one phone, one
print and your own room cannot prove a low attack-acceptance rate; they can only
show a high one. Treat a clean result as "no weakness found with these
instruments", never as "safe".

## What it measures

Each attempt runs `doorman --json test <user>` (the same decision code as the
lock screen and PAM) and records only: class, index, decision, reason code,
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

1. Prepare the daemon for the session (a deliberate, temporary change):
   * `[ipc] face_max_judged_failures = 0` in `/etc/doorman/doorman.toml` and restart
     `doormand`. Every attack attempt is meant to fail, and `doorman test` counts
     toward the failure budget (5 judged failures in 5 minutes, then a 30 s,
     2 min, 10 min lockout). Without this the script still works, but waits out
     every cooldown. With the default budget run it with `--budget-max 5` so the
     attempts are spaced to stay under it (it takes much longer).
   * `[dataset] keep_raw_frames = false`, otherwise the daemon stores the attack
     frames and yours.
   * Optional, for correlation with the daemon's own view: `[liveness]
     shadow_pad_logging = true` (see below).
2. Run, from the repository:

   ```
   python3 scripts/pad_protocol.py --user YOUR_USER_NAME
   ```

   For a short smoke test: `--genuine 3 --attempts 3 --condition-attempts 2`.
   A subset: `--classes genuine,phone_photo,print_matte`.
3. For each attempt the script prints the instruction, you press Enter, it counts
   down 3 seconds, runs the test, prints the outcome. If face auth is suspended it
   says so and waits until the end of the cooldown, then repeats the attempt.
4. At the end it prints a table and writes (mode 0600) `~/doorman-pad-<date>.json`,
   `.csv` and `-bench.csv` (`--out PREFIX` changes the location).
5. Restore the daemon settings from step 1 and restart `doormand`.

Exit codes of `doorman --json test` that the script understands: `0` recognized,
`2` not recognized (or suspended), `1` error. An error, a timeout or unreadable
output is recorded as infrastructure, excluded from the rates, and the attempt is
repeated; three in a row stop the run (results so far are still written).

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
of the script's arithmetic.

## Privacy

The script never captures, stores or shows a frame and writes no screenshots. Its
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

The file is readable by the `doorman` group only; read it as a user in that group
(or with sudo). The protocol script reads it if it can and attaches the matching
record's depth margin, MiniFASNet score and median similarity to each attempt.

To turn it off again: set `shadow_pad_logging = false` (or delete the key),
restart `doormand`, and delete the log file if you do not want to keep it.
