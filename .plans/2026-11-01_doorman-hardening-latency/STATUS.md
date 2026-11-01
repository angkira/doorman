# Status — 2026-11-01

Branch: `claude/great-ride-i3oqn6` (all workstreams merged sequentially; no per-workstream branches/PRs, by owner decision). Nothing released, no version bumps, nothing pushed to `main`.

## Verification in the cloud
- Build/test with `--no-default-features --features camera-mock,camera-ffmpeg,preview` (default build needs ORT download from `cdn.pyke.io`, blocked by the environment's network policy).
- ORT code is **type-checked** via `backend-ort-rocm` (`ort/load-dynamic`, no download) for daemon and bench; it was **never executed**.
- Gates at merge tip: `cargo fmt --check` OK; `clippy -D warnings` OK for both feature sets; `cargo test --workspace` all green except `test_detection_on_frames` (needs a real ORT backend); pytest 45 passed; shellcheck clean.
- No camera, Wayland or real PAM: every behaviour that needs them is unverified until the host checklists are run.

## Workstreams
| WS | State | Notes |
|---|---|---|
| WS1 IPC security | Done (host-unverified) | SO_PEERCRED policy, exhaustive authz, rate limiter, protocol_version, stream sockets 0600. `doorman enroll/list/remove` and `preview.py` (DetectAndRecognize) now need root. `docs/ipc-security.md` |
| WS6 Bench/telemetry | Done, partial | Metrics/synth/scores modes tested; `latency` and dir-mode `accuracy` compile-checked only; `record` camera path untested. Baseline numbers are from the host audit, not reproduced. ORT spinning/opt-level knobs TODO. |
| WS2 Event-driven unlock | Done (host-unverified) | `WaitForMatch` reruns the full Authenticate decision (background recognition loop deliberately not used: weaker). Concurrent IPC. `doorman unlock-watch` + `packaging/adapters/niri/bin/doorman-lock`; poll fallback via `DOORMAN_UNLOCK_MODE=poll`. logind `Session.Unlock()` not prototyped. |
| WS4 Camera | Done (host-unverified) | V4L2 root cause: hidden error + strict nokhwa format fulfil (hypothesis: 1280x720 not offered as YUYV) — needs `v4l2-ctl` output. Mode selection from offered formats, MJPEG for ffmpeg, warm-up discard, prearm bounded to `prearm_window_ms`. Defaults unchanged. |
| WS5 Decision | Done (host-unverified) | Structured `AuthDecision`; n_of_m, template scoring, flip TTA, brightness gate, versioned `embeddings.bin` (v2; old readable, older daemons can't read v2), pruning, `enroll --add`. Defaults = old behaviour. |
| WS3 Liveness | Done, partial | Relief value + `liveness_unavailable` code, multi-frame median, `p10_p90` stat, configurable depth input. Threshold not changed (no owner data). Challenge signals: interface only. No stub-ONNX shape tests. |
| WS7 sudo | Done (host-unverified) | Off by default; strict profile, freshness, presence, suspension, audit, fast PAM fallthrough, safe installer. Strict path inside `run_authenticate` only unit-tested in parts. |
| WS8 Hygiene | Done | ort pinned `=2.0.0-rc.12`, Cargo.lock committed, shellcheck clean, GPU removed from systemd DeviceAllow, licences in `MODELS.md`, `config check`, generated `docs/CONFIG.md`, smaller dataset caps, `dataset purge`, CI jobs, README trust-boundaries section. |

## Not done / blocked
- Anything needing the real host: all host validation checklists (in each doc), real latency/accuracy numbers, threshold calibration, default flips (n_of_m, warmup, prearm, 640x480 MJPEG, liveness frames=3).
- EdgeFace-S weights licence: Idiap HF card not readable here (likely CC BY-NC-SA 4.0).
- Root `tests/e2e_test.rs`, `tests/enrollment_e2e_test.rs`, `tests/video_tests.rs` belong to no Cargo target (dead); left in place.
- `DevicePolicy=closed` vs ffmpeg/GStreamer on real hardware: unverified.

## Owner actions (in order)
1. Back up `/var/lib/doorman/embeddings.bin` before the first enrol on the new build.
2. Build with default features, install, `doorman config check --config /etc/doorman/doorman.toml`.
3. Host config sets `keep_raw_frames=true` explicitly (per audit) — new defaults don't override explicit values; set it to `false` if you want the privacy default.
4. Remove `/usr/local/bin/doormand` and `/etc/doorman/*.bak*`.
5. Paste `v4l2-ctl --list-formats-ext` into `docs/camera.md`; run the WS4/WS2 checklists; enable `[telemetry] timing_log` and collect 20 unlocks.
6. Run `doorman-bench` on your recordings, then decide default flips.
7. sudo last, with a second root shell open: `docs/SUDO.md`.
