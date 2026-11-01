# WS5 — Decision pipeline: early exit, N-of-M, multi-template, quality (P1, effort S–M; evaluate with WS6)

## Problem
Auth currently collects `auth_frames=10` quality embeddings (≈1.25 s at ≈7.6 fps), averages them, and compares the mean to the gallery with cosine ≥ 0.4. It always waits for all 10 even when the first 3 are decisive. Enrolment/gallery handling and score aggregation are fixed; lighting changes hurt.

## Tasks
1. **Sequential decision (early exit)**: after each quality embedding, compute per-frame max cosine to the gallery templates; accept as soon as **N of the last M** frames exceed the threshold (config `decision.accept_n=3`, `decision.window_m=5`) **and** liveness has passed (WS3); reject early only when the remaining frames cannot reach N within the time budget. Keep the old "mean of K" mode behind `decision.mode = "mean" | "n_of_m"`, default **"mean"** until WS6 shows equal-or-better FRR/FAR; then flip the default in a separate commit.
2. **Scoring over multiple templates**: store several templates per user (cluster by appearance/lighting), score as `max` or `top-3 mean` per frame; keep `centroid` as option. Configurable `decision.template_score`. Evaluate all three in WS6 — do not guess a winner.
3. **Enrolment improvements**: on enrol, capture ≥ 10–30 quality frames over a few seconds, prompt for different head poses/lighting, **prune near-duplicate embeddings** (cosine > ~0.95), reject low-quality frames. Allow `doorman enroll --add` to add templates for another lighting condition without wiping existing ones. Embeddings file format change must be versioned and backward-compatible (read old `embeddings.bin`), with a migration test using a synthetic old-format fixture.
4. **Flip test-time augmentation** (embed the horizontally flipped crop, average, L2-normalise): config flag, default off; measure cost vs gain in WS6.
5. **Quality weighting**: the sharpness/size gate exists; add brightness/pose gate and let a poor-quality session require more agreeing frames instead of lowering the threshold. Optional: embedding-norm quality proxy (MagFace-style) — only if the model's norm correlates in WS6; otherwise skip.
6. **Adaptive threshold (calibration only)**: provide a script that derives the cosine threshold for a target FAR from an impostor-score file; the daemon keeps a plain configured threshold. No runtime self-adjusting thresholds (security-sensitive).

## Tests
- Unit: N-of-M state machine over scripted score sequences (accept at frame 3, never accept with isolated spikes, timeout behaviour, early reject).
- Property: decision never accepts with fewer than N frames above threshold; never accepts without liveness pass.
- Embeddings store: old-format read, new-format round trip, near-duplicate pruning.
- Deterministic fixtures only (synthetic unit vectors); no real faces in the repo.

## Acceptance
- With `n_of_m` mode and the WS6 genuine set: median frames processed per successful unlock ≤ 4 (vs 10) with FRR not worse than baseline and FAR at 1e-3/1e-4 not worse on the impostor set (report numbers).
- Default behaviour unchanged until explicitly flipped.

## Host validation checklist (human)
Re-enrol with the new flow in two lighting conditions; compare unlock time/feel over 20 attempts; `doorman test` shows frames used and per-frame scores at debug level.
