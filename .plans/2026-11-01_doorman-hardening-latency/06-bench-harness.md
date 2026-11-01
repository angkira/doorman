# WS6 — Benchmark & evaluation harness + model experiment matrix (P0, start first, effort M)

## Why first
Every other workstream claims a latency or accuracy win. Today nothing measures them, per-stage timings are not logged, and the owner's real recordings live on the host (`/var/lib/doorman/dataset`, unreadable to normal users, not in the repo). Build the harness so the cloud dev can develop against synthetic/public data and the owner can run it on real data.

## Deliverables
1. **Per-stage timing instrumentation** in the daemon (debug-level, structured): capture open/first-frame/first-valid, detect, align, embed, liveness, decision, total; one JSON line per auth attempt to a bounded rotating file under `$XDG_RUNTIME_DIR`/`/run/doorman` (no frames, no embeddings). Config flag `telemetry.timing_log`.
2. **Offline evaluation binary / subcommand** `doorman-bench` (Rust bin in the workspace, or `doorman bench …`):
   - `bench latency --models <profile> --frames <dir|video> --iters N`: p50/p95/p99 per stage, cold-start (session creation + first inference) separate from steady state, ORT thread sweep (`intra_op` 1/2/4/8/16, spinning on/off, optimisation level), pre-optimised graph file (`optimized_model_filepath`) on/off.
   - `bench accuracy --genuine <dir> --impostor <dir> --enroll <dir>`: genuine accept rate at FAR 1e-3 and 1e-4, EER, per-condition worst case, score histograms (CSV), ablations from WS5 (template scoring, flip TTA, N-of-M).
   - `bench liveness --genuine … --attack …`: relief distributions, ROC, APCER/BPCER at configurable thresholds (WS3).
   - `bench record`: guided recording of a labelled dataset from the real camera (conditions: day, evening lamp, dark with screen light, backlit; attack captures: phone replay, screen replay, printout) into a directory layout documented in the README, with consent notice, **stored outside the repo, `.gitignore`d**.
3. **Dataset layout + loader** (documented): `data/<user>/<condition>/*.jpg|*.png`, `attacks/<type>/…`, `impostor/…` (e.g. a public face set the owner downloads; do not vendor any dataset). Provide a synthetic-data generator for CI (random textured images + stub models) so tests run without real faces.
4. **Report generator**: markdown + CSV into `bench/results/<date>/` with machine info (CPU, ORT version, threads) so runs are comparable.
5. **CI job** running the harness on synthetic data (smoke + regression thresholds only on algorithmic invariants, not on absolute timings).

## Experiment matrix (run on the owner's host; record results in `docs/bench/`)
Ranked by expected value / effort (from the research digest; effects are hypotheses):
1. Cold camera vs warm persistent camera vs prearm-on-lock (WS4) — expect −1.3 s.
2. Early-exit N-of-M (3/5) vs mean-of-10 (WS5) — expect large cut in frames processed.
3. Warm-up frames 0/3/10 + brightness gate under varied light (WS4/5).
4. Capture 1280x720 vs 800x600 vs 640x480; MJPEG vs YUYV incl. decode cost (WS4).
5. Detector input 640 vs 480 vs 320 using the dynamic-shape YuNet `2026may` ONNX (same weights as 2023mar; check recall at the working distance).
6. Liveness depth input 518/308/252 and fp32 vs w8a16 (WS3); relief separation must hold.
7. Relief threshold calibration from real genuine/attack distributions (WS3).
8. Multi-template enrolment: bright/dim/side-light; max vs top-3 mean vs centroid (WS5).
9. EdgeFace-S fp32 vs int8 (static QDQ, per-channel, calibration frames from the owner's camera); inspect graph for transpose overhead; also AdaFace R50 / SFace-int8 as alternative embedders **if** a clean ONNX export exists (check licences: insightface weights are non-commercial; EdgeFace weight licence unverified — personal use only).
10. ORT CPU tuning sweep (threads, spinning, graph optimisation level, pre-optimised model file); `ort` bundled runtime vs distro ORT (the host has distro ORT 1.21 packages while the crate bundles its own — **[unverified]** which one the daemon really loads; print the loaded version at startup).
Not worth time now: iGPU (ROCm gfx1103 unverified; dispatch overhead likely dominates for tiny models; it also must never touch the dedicated training GPU), NPU (AMD Linux stack starts at Strix Point; Phoenix not listed).

## Tests
- Harness unit tests (metric computation vs hand-computed examples: EER, TAR@FAR, percentile maths).
- Determinism test: same synthetic input → same report (seeded).
- CLI smoke tests.

## Acceptance
- `doorman-bench latency` and `accuracy` run end-to-end in CI on synthetic data; documented command lines for the owner's real data; a first baseline report committed under `docs/bench/` (with the baseline numbers from `00-context.md` reproduced or corrected).
