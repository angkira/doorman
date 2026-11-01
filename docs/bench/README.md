# doorman-bench

Benchmark and evaluation harness for the doorman daemon. It reuses the daemon
library (`doormand`) for detection, embedding and liveness, so it measures the
code that actually runs on the unlock path.

Build (on the host, with models available):

```sh
cargo build --release -p doorman-bench            # default features: ort + camera-ffmpeg
```

Model-free subset (what CI builds; `latency` and directory-mode `accuracy`
then fail with an explicit error):

```sh
cargo build -p doorman-bench --no-default-features
```

Outputs go to `--out <dir>` (default `bench/results/<date>/`, git-ignored).
Every report directory contains deterministic report files plus `machine.md`
(CPU model, cores, ONNX Runtime linkage/version, git rev, timestamp, command).
Only `machine.md` contains volatile data.

## Dataset layout

Datasets live OUTSIDE the repository and must never be committed.

```
<root>/
  <user>/<condition>/0001.jpg ...    genuine captures; also the enrolment source
  impostor/<identity>/0001.jpg ...   other people (e.g. a public face set you download)
  attacks/<type>/0001.jpg ...        phone_replay, screen_replay, printout, ...
```

Directory mode of `accuracy` takes plain directories: `--enroll`, `--genuine`
(images directly inside, or in `<condition>/` subfolders; the folder name is the
condition) and `--impostor` (any depth). A probe is scored as the maximum
cosine similarity against the enrolment embeddings; images without a detected
face are skipped and counted in `dataset_notes.md`. Suggested conditions:
`day`, `evening_lamp`, `dark_screen_light`, `backlit`.

Record with consent (refuses to write inside a git repo, refuses the mock camera):

```sh
doorman-bench record --out ~/doorman-data --user me --condition day --frames 60
doorman-bench record --out ~/doorman-data --user me --condition evening_lamp
doorman-bench record --out ~/doorman-data --impostor p01
doorman-bench record --out ~/doorman-data --attack phone_replay
```

## Scores CSV formats

Accuracy: `label,score[,condition]`, label in `genuine|impostor`, higher score
means more similar. Liveness: `label,score`, label in `bona_fide|attack`, score
is the depth relief (accepted as live iff `score >= threshold`, like the daemon).
Lines starting with `#` are ignored; errors report the line number.

## Metric definitions

- Percentiles: linear interpolation between closest ranks (R-7, numpy default).
- Accuracy accept rule `score > threshold`. TAR@FAR=f uses the threshold that
  allows at most `floor(f*N)` of the N impostor scores. If `f*N < 1` the point
  is marked NOT resolvable: FAR 1e-4 needs at least 10,000 impostor scores.
- EER: threshold minimising `|FAR-FRR|`, EER = `(FAR+FRR)/2`.
- Per-condition worst case: TAR of each genuine condition at the global
  threshold; the minimum is reported.
- Liveness: APCER = attacks accepted, BPCER = bona fide rejected. The
  recommended threshold is the highest one with `BPCER <= --target-bpcer`
  (default 1%).

## Commands per experiment-matrix item

Model-free (work anywhere):

```sh
doorman-bench synth --out /tmp/bench-synth --seed 1          # CI-style synthetic data
doorman-bench accuracy --scores /tmp/bench-synth/accuracy_scores.csv --out /tmp/bench-report
doorman-bench liveness --scores /tmp/bench-synth/liveness_scores.csv --thresholds 0.05,0.1 --target-bpcer 0.01
```

Supported now, run on the host (models required; pass `--profile /etc/doorman/doorman.toml`
or `--models-dir <dir>`):

| Matrix item | Command |
|---|---|
| 10. ORT intra-op thread sweep, cold vs steady | `doorman-bench latency --frames ~/doorman-data/me/day --iters 200 --warmup 10 --threads 1,2,4,8,16 --profile /etc/doorman/doorman.toml` |
| 3, 8, 9 (accuracy part). Genuine accept rate at FAR 1e-3/1e-4, EER, worst condition | `doorman-bench accuracy --enroll ~/doorman-data/me/day --genuine ~/doorman-data/me --impostor ~/doorman-data/impostor --profile /etc/doorman/doorman.toml` |
| 7. Relief threshold calibration | produce a `label,score` CSV of relief values for genuine and attack captures, then `doorman-bench liveness --scores relief.csv --target-bpcer 0.01` |
| 5, 6, 9 (model variants) | point `--models-dir` at a directory with the variant model (same file names) and repeat `latency` / `accuracy`; compare reports |
| Timing in the real daemon | set `[telemetry] timing_log = true` in the daemon config; see below |

Not supported by the harness yet (TODO): ORT spinning on/off, graph optimisation
level and `optimized_model_filepath` toggles (not exposed by the daemon backend),
dumping per-image liveness relief values from images (live relief values come from
the timing log's `liveness_relief`, see docs/decision-and-liveness.md), N-of-M / template-scoring
ablations (WS5; the daemon now has `[decision]` settings, but the harness does not drive them yet), camera-open/MJPEG/resolution comparisons (WS4; use the daemon
timing log, which records `camera_open` and `first_frame`).

Stage granularity of `latency`: `detect`, `embed (incl. align)` and `liveness
(depth gate)`. Face alignment runs inside the backend's embedding call and
cannot be timed separately without changing the daemon.

## Calibrating the cosine threshold for a target FAR

The daemon keeps a plain configured threshold
(`[authentication] similarity_threshold`, default 0.4); nothing adapts at
runtime. To derive it from data:

1. Build an accuracy scores CSV (`label,score[,condition]`) from your genuine
   captures and an impostor set, either with directory mode
   (`doorman-bench accuracy --enroll ... --genuine ... --impostor ...`, which
   writes `scores.csv` into the report directory) or from your own scorer.
   Score each probe the way the daemon does: with the same `[decision]
   template_score` you run (see docs/decision-and-liveness.md).
2. Run `doorman-bench accuracy --scores scores.csv --far 1e-3,1e-4 --out report/`.
3. Read `accuracy.md`: the table lists, per FAR target, the threshold that
   allows at most `floor(FAR*N)` impostor scores above it, and the TAR (genuine
   accept rate) at that threshold. Targets with `resolvable = no` need more
   impostor scores (FAR 1e-4 needs at least 10,000). Pick the threshold for the
   FAR you accept and put it in `similarity_threshold`.
4. Check the per-condition table: the worst genuine condition at that threshold
   is the one that will fail at the lock screen first.

For `n_of_m` the per-frame score is the quantity compared with the threshold,
so calibrate on per-frame scores, not on scores of averaged embeddings.

## Daemon timing log

Config (all optional, default off):

```toml
[telemetry]
timing_log = true
timing_log_path = "/run/doorman/timing.jsonl"   # default
timing_log_max_bytes = 1048576                  # default 1 MiB; rotates to .1, one generation kept
```

One JSON line per authentication attempt, always also emitted at debug level
(`target doorman::timing`). Fields: stage durations in ms (`camera_open_ms`,
`first_frame_ms`, `first_valid_frame_ms`, `detect_sum_ms`/`detect_count`,
`embed_sum_ms`/`embed_count`, `liveness_ms`, `decision_ms`, `total_ms`),
`align_ms` (always null, see above), `frames_seen`, `frames_used`, `outcome`,
`liveness_passed`, `liveness_relief`, `liveness_frames` and `similarity`.

Privacy: no username, uid, frame or embedding is written. The final aggregate
cosine `similarity` IS written because it is the calibration signal for the
threshold; the depth `liveness_relief` is written for the same reason (recipe
in docs/decision-and-liveness.md). The attempt id is a counter plus a timestamp. The file
is created with mode 0640.

## ONNX Runtime: bundled or system?

The default and CUDA builds (`backend-ort`, `backend-ort-cuda`) enable
`ort/download-binaries`: the build downloads and statically links Microsoft's
CPU runtime, so the distro `libonnxruntime` packages are NOT used. Only
`backend-ort-rocm` uses `ort/load-dynamic` and loads a system
`libonnxruntime.so` (ORT 1.22.x) via `ORT_DYLIB_PATH`. See
`daemon/Cargo.toml` (`backend-ort`, `backend-ort-rocm` features). The daemon
logs `ONNX Runtime: <linkage>; <ORT build info>` at startup and `machine.md`
records the same string.
