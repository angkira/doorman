# WS3 — Liveness: stop false rejects, make the check stronger and cheaper (P1, effort M; uses data from WS6)

## Problem
- Active gate: Depth-Anything-V2-Small fp32 518x518 on CPU (heaviest per-attempt step), single frame (the first quality frame), relief metric `std(face_depth)/(max-min+eps)` vs global threshold **0.05**.
- A genuine-user attempt measured `relief=0.0491` → rejected, costing a whole extra attempt (~3 s incl. watcher delay). Calibration range for this user was 0.068–0.245, so the threshold sits at the edge of the distribution and is lighting/pose sensitive.
- Single-frame check: a spoof presented after that frame is unchecked.
- MiniFASNet (RGB PAD) files are shipped but unused; RGB-only PAD is known to generalise poorly. Do not oversell any option below as a security boundary.

## Tasks
1. **Instrument**: log (debug) the relief value, face-box size, frame sharpness/brightness, depth latency for every liveness evaluation (no frames). Needed by WS6 to calibrate.
2. **Multi-frame decision**: evaluate liveness on up to K quality frames (config `liveness.frames`, default 3) and decide on the **median** relief (or ≥ M of K pass). Early-exit if the first two agree. Keep wall-clock bounded by the overall auth timeout.
3. **Threshold calibration tool** (in WS6 harness or `scripts/`): given genuine and attack relief distributions, emit an ROC and suggest a threshold at a chosen FRR; threshold stays configurable (`liveness.min_relief`). Do not change the default threshold until WS6 data from the owner's real recordings exists; ship a documented recipe instead.
4. **Cheaper depth**: make input size and precision configurable per model profile; benchmark (via WS6) 518 vs 308 vs 252 input (dynamic ONNX export or re-export; Depth-Anything-V2 requires multiples of 14) and fp32 vs the published w8a16 ONNX (weight-only quantisation; check it actually runs fast on ORT CPU, it may be tuned for another runtime). Adopt only if relief separation (genuine vs screen/print) is preserved.
5. **Face-region normalisation**: compute relief on a robust face-region statistic (e.g. 10–90 percentile range inside the aligned face mask) rather than min/max; compare in the harness.
6. **Optional stronger signals (behind flags, default off)**: screen-flash reflectance challenge (random colour sequence on the display, check face colour response within a time bound); document that it is weak under bright ambient light and defeated by live relay attacks. Active head-pose challenge. These are required for any sudo-mode ambition (WS7); implement only the interfaces + one reference signal now.
7. Failure semantics: liveness model missing/ORT error → **fail closed** (as today) with an explicit error code distinguishing "model unavailable" from "spoof suspected"; surfaced in logs and `doorman test` output.

## Tests
- Unit: median/M-of-K logic with synthetic relief sequences; threshold boundary behaviour; fail-closed on error.
- Property: decision latency bounded by timeout.
- Model-shape tests with tiny stub ONNX models for the input-size configurability (no real weights in CI).

## Acceptance
- On the owner's recorded genuine set (WS6) false-reject ≤ 1% at the chosen threshold; on the replay/print attack set the harness reports APCER/BPCER honestly (state clearly that this is the owner's own small attack set, not a certification).
- Liveness stage p50 latency reduced vs baseline or equal with higher separation; numbers in the PR.

## Notes / risks
- No evidence found that another depth model (MiDaS-small, ZipDepth) is better here; ZipDepth (6.1M params, ECCV 2026) appeared in search but without ONNX/licence/CPU latency — only try it if the harness shows depth is still the bottleneck after the changes above.
