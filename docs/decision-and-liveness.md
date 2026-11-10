# Decision pipeline and liveness

All settings below are optional. With none of them set the daemon behaves as
before: the mean of `auth_frames` embeddings is compared with the best stored
template, and one depth-relief check runs on the first quality frame. Old
config files parse unchanged. Invalid values are rejected when the config is
loaded, with an error naming the key.

## Settings

| Key | Default | Meaning |
|---|---|---|
| `[decision] mode` | `"mean"` | `mean`: average `auth_frames` embeddings, score once. `n_of_m`: score every frame, accept early. |
| `[decision] accept_n` | `3` | `n_of_m`: frames that must reach `similarity_threshold`. |
| `[decision] window_m` | `5` | `n_of_m`: sliding window of the most recent quality frames. `1 <= accept_n <= window_m`. |
| `[decision] template_score` | `"max"` | How one embedding is scored against the stored templates: `max`, `top3_mean`, `centroid`. |
| `[decision] flip_tta` | `false` | Also embed the mirrored aligned crop and average (one extra recognizer pass per frame). |
| `[recognition] min_brightness` | `0` (off) | Minimum mean luma (0..=255) inside the face box. |
| `[enrollment] prune_cosine` | `0.95` | Enrol drops a template whose cosine to a kept one is at or above this. Range (0, 1]. |
| `[liveness] frames` | `1` | Frames the depth check is evaluated on (odd, 1..=9). |
| `[liveness] aggregate` | `"median"` | Only `median` exists. |
| `[liveness] relief_stat` | `"minmax"` | `minmax` (original) or `p10_p90`. |
| `[liveness] depth_input_size` | `518` | Depth network input, a multiple of 14. |
| `[liveness] mean_mode_depth_samples` | `0` | `mean` mode only: extra frames depth-checked, evenly sampled across the sequence (`0..=auth_frames`). See "Sampled depth checks in mean mode". |
| `[liveness] signals` | `[]` | Placeholder for extra signals; any entry is rejected (none implemented). |
| `[authentication] liveness_depth_threshold` | `0.05` | Relief threshold (unchanged). |

`doorman enroll <user> --add` appends templates without evicting any (see
Enrolment). The IPC `Enroll` request keeps working for old clients (`add`
defaults to false).

## Decision modes

**mean** is the original rule: collect `auth_frames` quality embeddings,
renormalised mean, one score. Same decision as before this change.

**n_of_m** scores each quality frame as it arrives (`template_score` against the
gallery). It accepts as soon as `accept_n` of the last `window_m` frame scores
are `>= similarity_threshold` and liveness has passed. It rejects early when
liveness failed, or when `accept_n` can no longer be reached within the
remaining frame budget. The budget is `authentication.auth_frames` (so it must
be `>= accept_n`; checked at load) and the run is still bounded by
`authentication.timeout_secs`; when time runs out the result is "not matched".
The reported similarity is the mean of the window scores that reached the
threshold. Isolated spikes do not accept: three hits must fall inside five
frames. Calibrate the threshold on per-frame scores (docs/bench/README.md).

Expected effect: a typical genuine unlock needs 3 quality frames instead of 10,
at the frame rate of the camera (about 7-8 fps measured). This is a hypothesis
until measured on the host (checklist below).

## Template scoring

The comparison the daemon always did is `max`: cosine of the (aggregated)
embedding to every stored template, best one wins
(`daemon/src/ipc.rs`, `run_authenticate`; before this change an inline
`fold(0.0, f32::max)`), so `max` is the default and gives the same numbers.
`top3_mean` averages the three best template cosines (a single lucky template
counts for less). `centroid` scores against the renormalised mean of all
templates (smooths the gallery, usually lower for genuine and impostor alike).
Which one wins is an empirical question for your data: compare them with
`doorman-bench accuracy`; do not assume.

## Enrolment

* `embeddings.bin` is versioned (magic + version 2). The old headerless file is
  still read; it is rewritten in the new layout on the next save. An older
  daemon cannot read version 2.
* Before that first version-2 write the daemon copies the old file to
  `embeddings.bin.v1.bak` (mode 0600, same directory, `create_new`, so an
  existing backup is never overwritten). It is made once; later saves and a
  v1 file that reappears do not touch it. **Rollback:** stop the daemon, install
  the older build, `mv embeddings.bin.v1.bak embeddings.bin` (users enrolled
  after the upgrade are lost), start it. To get a fresh backup for a later
  upgrade, move the old backup away first.
* Saves write `embeddings.bin.tmp` with mode 0600 from creation, fsync it,
  rename it over `embeddings.bin` and fsync the directory. `/var/lib/doorman`
  is 0700 (`StateDirectoryMode=0700` in the unit, `0700` in the installer and
  the deb; the daemon also tightens a directory it owns).
* Near-duplicates (`cosine >= prune_cosine`) are dropped before the existing
  union and diverse selection (at most 30 templates).
* Plain `doorman enroll` merges with the stored templates and may evict
  similar older ones when the 30-template cap is reached (unchanged). `--add`
  never evicts: it fills free slots with new distinct templates and fails with
  an explicit message if the new faces are all near-duplicates or the gallery
  is full. Use it for another lighting condition.
* Enrol stays root/admin only. The CLI prints a hint to turn the head slightly
  and vary the lighting; there are no interactive pose prompts.

## Quality gate

Frames are skipped when the face is too small, too blurry or (optionally) too
dark. Sharpness, face area and face-box brightness are logged at debug level
for every decision. Start with `min_brightness` unset, look at the logged
values of your failures in dim light, then set it just below them.

## Liveness

### Result type and errors

Each evaluation returns `Pass{relief}`, `Fail{relief}` or `Unavailable{reason}`.
A missing depth model or an inference error is `Unavailable`: fail closed, with
its own error code. `doorman test` shows
`Liveness check unavailable [liveness_unavailable]: <reason>` instead of the
spoof rejection (`Liveness check failed (not a live face).`, code
`liveness_rejected`, also the dataset failure reason). The timing JSONL has
outcome `liveness_error` for the first and `liveness_rejected` for the second.

At debug level each evaluation logs: relief, face box size in pixels,
sharpness, brightness and depth latency. No frames are logged.

### Multi-frame

`[liveness] frames = 3` evaluates the depth check on the first quality frames
until the verdict is decided. The verdict is "median relief >= threshold",
which for an odd count equals "a majority of frames pass", so it is final as
soon as one side has a majority. For 3 frames that is the early exit "the first
two frames agree" (both at or above the threshold, or both below): the third
frame is only evaluated when they disagree. Each evaluation costs one depth
pass (about 0.5-1.3 s on CPU at 518), so the worst case is three passes; the
run is bounded by `authentication.timeout_secs`, and if the timeout ends it
before a verdict the result is `liveness_unavailable`. In `n_of_m` mode accept
requires the liveness verdict to have passed; frame scores wait (`Pending`)
until then. The default stays 1 (current behaviour); 3 is the recommended value
to try.

### Binding liveness to the frames that decide

Liveness is evaluated on the first quality frames, before the frames that carry
the match are collected. Without more, a live face followed by a photo or phone
of the owner passes both checks (review finding M2). Therefore, **when an
attempt is about to be accepted** (`n_of_m` accept, or `mean` over the
aggregated frames), the deciding frame, i.e. the last quality frame whose
embedding was scored, must itself pass the depth check
(`confirm_on_deciding_frame` in `daemon/src/liveness.rs`, called from
`run_authenticate`). In `n_of_m` mode that is the frame that completes the Nth
hit; in `mean` mode it is the last frame of the aggregate, which lies in its
last third. If that frame was already the one liveness was decided on, no second
inference is made. A failed or unavailable recheck rejects fail-closed with the
usual `liveness_rejected` / `liveness_unavailable` code; non-matching attempts
never pay for it.

Latency cost: one extra depth pass (about 0.5-1.3 s on CPU at 518) for every
attempt that would otherwise be accepted and whose deciding frame was not
already checked, which is the normal case (default `frames = 1`, accept at
frame 3 or later). The recheck is a single-frame decision (no median), so the
per-frame false-reject rate (relief 0.0491 vs 0.0500 was seen once) now applies
twice per unlock; a genuine retry is the cost of a rejection.

Residual risk: only the first checked frame(s) and the deciding frame are
verified. An attacker who can show a live face at the start and at the end and
a photo on the frames in between could still supply the accepting hits from the
unchecked middle frames (`n_of_m` needs N hits, so this needs N >= 2 photo
frames between two live ones within roughly a second; `mean` averages them in).
Closing that fully means a depth pass on every accepting frame
(`~N x 0.5-1.3 s`), which was not chosen for latency. A deeper fix needs
identity-bound liveness (same face track), listed under future work. For
`mean` mode the cost/coverage trade-off is configurable, see the next section. The
synthetic swap tests are in `daemon/src/ipc.rs`
(`live_then_photo_swap_is_rejected_by_the_deciding_frame_check`); they mirror
the control flow with synthetic frames because `run_authenticate` needs the ML
models and a camera.

### Relief statistic and depth input size

`minmax` is the original `std(face depth) / (max - min of the whole depth map)`.
`p10_p90` is `(p90 - p10 of the face region) / (p90 - p10 of the whole map)`,
less sensitive to single outlier pixels. The two are on different scales:
recalibrate `liveness_depth_threshold` before switching. `depth_input_size`
(multiple of 14) must match the depth ONNX in the models directory; the shipped
fp32 export is fixed at 518. Sizes such as 308 or 252 need a model exported for
that size (or with dynamic axes) under the same file name. The ncnn hybrid
backend keeps its fixed 518 input.

### Stronger signals (future work, not implemented)

`LivenessSignal` (`daemon/src/liveness.rs`) is the interface for signals beyond
depth relief; `[liveness] signals` is rejected while none exist. Candidates and
their known weaknesses:

* **Screen flash** (random colour sequence on the display, check the face
  colour response). Weak under bright ambient light, and defeated by a live
  relay (an attacker streaming a live face).
* **Head pose challenge** (ask the user to turn). Costs time on every unlock
  and is defeated by a live relay or a replayed video of the right motion.

Treat depth relief from an RGB camera as a deterrent, not a security boundary.

### Sampled depth checks in mean mode

`[liveness] mean_mode_depth_samples = K` (default `0`, behaviour unchanged)
depth-checks K more frames in `mean` mode, only when the attempt would be
accepted. The frames are evenly spread over the quality frames that were
scored before the deciding frame, skipping any that already had a depth check
(`sample_unchecked_frames` in `daemon/src/liveness.rs`: midpoints of K equal
slices). Any failed or unavailable sample rejects fail-closed with the usual
`liveness_rejected` / `liveness_unavailable` code. Valid range
`0..=authentication.auth_frames`; it has no effect in `n_of_m` mode. With K > 0
the daemon keeps at most `auth_frames` decoded frames in memory for the attempt.

Coverage is statistical, not exhaustive: a single photo frame that no sample
lands on is still averaged in; K = `auth_frames - 2` checks every middle frame.
Swap tests: `mean_mode_samples_catch_photo_frames_mid_sequence` (and the
default-behaviour record `mean_mode_default_checks_only_first_and_deciding_frames`)
in `daemon/src/ipc.rs`.

Cost: about K x the per-frame depth latency, added to accepting attempts only.
Measure the per-frame figure on the host with the `liveness (depth gate)` stage
of
`doorman-bench latency --frames ~/doorman-data/me/day --iters 200 --warmup 10 --profile /etc/doorman/doorman.toml`
(docs/bench/README.md), then multiply by K; the timing log's `liveness_frames`
and `liveness_ms` show the real count and total per attempt.

### Frame-level rules in mean mode (Round 9a)

Applied by `run_authenticate` before a frame can enter the aggregate; the pure
rules are in `daemon/src/frame_policy.rs`.

| Rule | Key (default) | Effect |
|---|---|---|
| Identity floor | `[decision] frame_identity_floor` (0.30; 0 = off) | Each frame scored against the templates (same `template_score` as the decision) must reach the floor, otherwise the attempt ends as `identity_floor` (fail closed, a judged failure). Binds the liveness-checked frames to the owner: a live other face on the first and deciding frames with owner photos in between was accepted before (pinned by `identity_floor_rejects_a_live_attacker_around_owner_photos`, which shows the old acceptance with the floor off). Genuine frames sit far above it (host mean 0.74-0.82); a frame below 0.30 would drag the mean down anyway. `n_of_m` mode already scores every frame. |
| Multi-face gate | `[recognition] multi_face_policy` (`reject`) | The detector reports the other faces it found anyway. A frame with a second face of at least `multi_face_min_area_ratio` (0.25) of the main face area and detector score `multi_face_min_score` (0.75) is not accepted; if that leaves too few frames the attempt ends as `ambiguous_faces`. `largest` uses the largest face; `off` keeps the old highest-score pick. |
| Minimum frames | `[authentication] min_decision_frames` (0 = `auth_frames` - 2) | An attempt that ends with fewer judged-good frames is rejected as `insufficient_frames` (not a judged failure) instead of deciding on whatever the deadline left. |
| Frame age | `[camera] max_frame_age_ms` (1000; 0 = off) | A frame older than this when taken up is skipped; with no usable frame the attempt ends `no_frames`. |

Added work: the per-frame score is one dot product per template, already
computed for the aggregate; no extra inference. All four are fail closed.

## Calibration recipes

**Cosine threshold:** docs/bench/README.md, "Calibrating the cosine threshold".

**Liveness threshold from the timing log.** The relief of each attempt is in the
timing JSONL (`liveness_relief`; with `frames > 1` it is the median at the
moment the verdict was decided, so calibrate with the same `frames`).

```sh
# genuine run: 200+ attempts at different times of day
#   [telemetry] timing_log = true, timing_log_path = "/var/lib/doorman/relief-genuine.jsonl"
for i in $(seq 200); do doorman test "$USER"; sleep 1; done
# attack run: point timing_log_path at relief-attack.jsonl, replay phone/print attacks
jq -r 'select(.liveness_relief != null) | "bona_fide,\(.liveness_relief)"' relief-genuine.jsonl >  relief.csv
jq -r 'select(.liveness_relief != null) | "attack,\(.liveness_relief)"'    relief-attack.jsonl  >> relief.csv
doorman-bench liveness --scores relief.csv --target-bpcer 0.01
```

The report gives APCER/BPCER per threshold and the highest threshold with
BPCER <= 1%. The attack set is your own small one: report it as such, not as a
certification. Do not change the default 0.05 without such data.

## Host experiments (command lines only)

```sh
# depth precision: fp32 vs w8a16. Put the variant under the fp32 file name in a copy of the models dir.
cp -r "$MODELS_DIR" ~/models-w8a16
cp depth_anything_v2_small_w8a16.onnx ~/models-w8a16/depth_anything_v2_small_fp32.onnx
doorman-bench latency --frames ~/doorman-data/me/day --iters 200 --warmup 10 --threads 1,4,8 --models-dir "$MODELS_DIR"
doorman-bench latency --frames ~/doorman-data/me/day --iters 200 --warmup 10 --threads 1,4,8 --models-dir ~/models-w8a16
# input size: export the model for 308 (or 252), then in the config: [liveness] depth_input_size = 308
```

Adopt a variant only if relief separation (genuine vs your attacks) is
preserved: compare `doorman-bench liveness` reports.

## Recommended host settings to try (one at a time)

```toml
[telemetry]
timing_log = true            # needed for the relief and latency numbers
[liveness]
frames = 3                   # fewer false rejects from a single bad frame
[decision]
mode = "n_of_m"              # accept_n = 3, window_m = 5 by default
template_score = "max"       # compare top3_mean with doorman-bench accuracy
```

Then enrol a second condition with `doorman enroll <user> --add`.

## Host validation checklist

- [ ] Defaults: `doorman test` and the lock-screen unlock behave as before.
- [ ] `n_of_m`: 20 unlocks in two lighting conditions; the timing log shows
      `frames_used` (target median <= 4) and no false rejects. Try the same
      with an old photo on a phone: still rejected.
- [ ] `liveness.frames = 3`: `liveness_frames` is 2 for most attempts, 3 on
      disagreement; no extra false rejects over 20 attempts.
- [ ] Deciding-frame check: `[telemetry] timing_log` shows
      `liveness_frames` one higher than before for accepted attempts (the
      recheck); show a photo on a phone after covering the camera and
      presenting a face for the first moment: rejected as `liveness_rejected`.
- [ ] Rename the depth model: `doorman test` prints `[liveness_unavailable]`
      and the password still works; restore it.
- [ ] `enroll --add` in a second lighting condition keeps the first
      templates (`doorman list` count grows; unlocks still work in both).
- [ ] Old `embeddings.bin` (copy it first) loads after the upgrade.
- [ ] `flip_tta = true`: note the extra embed latency in `doorman-bench latency`.
- [ ] Relief recipe above produces `relief.csv` with sensible genuine values.

## Not verified in the development environment

The ONNX Runtime code could not be built or run there: the flip pass in the
recognizer, the `[liveness]` settings passed to the depth session, and the ncnn
hybrid relief call were type-checked against a different `ort` release only
(three unrelated pre-existing errors remained). All logic that does not need a
model is covered by unit tests; everything else is in the checklist above.
