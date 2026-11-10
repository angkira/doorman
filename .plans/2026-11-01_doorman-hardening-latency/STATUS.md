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
| WS8 Hygiene | Done | ort pinned `=2.0.0-rc.12`, Cargo.lock committed, shellcheck clean, GPU removed from systemd DeviceAllow, licences in `docs/MODELS.md`, `config check`, generated `docs/CONFIG.md`, smaller dataset caps, `dataset purge`, CI jobs, README trust-boundaries section. |

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

## Round 2 (owner review, `ROUND2.md`)
Process: owner chose to keep ONE branch (`claude/great-ride-i3oqn6`) with a single PR to `main`. A 12-PR stack (#1–#12) was briefly opened and then closed unmerged; it is superseded by PR #13. Its `pr/*` branches still exist on origin: the cloud session's git proxy refuses branch deletion. Owner may delete them (`git push origin --delete pr/01-ort-pin … pr/12-round2-review-fixes`); nothing on them is needed.

Done (commits after `a0edca4`, tests in each commit):
- Task 1: `owner/host-facts` merged into this branch (`a0edca4`).
- Task 2: no GPU device access in the generated unit, postinst or packaged unit (`daemon_unit_grants_no_gpu_access`). `doorman-depthd` (`--ep rocm` only) keeps iGPU access by design. Host `10-drm.conf` is not generated by code; remove by hand.
- Task 3: M1 (64 KiB line cap, `max_connections_per_uid=8`), M2 (deciding frame must pass depth; residual risk: middle frames unchecked, mean mode weaker than n_of_m), M3 (0600 at creation, fsync, `StateDirectoryMode=0700`), M4 (`embeddings.bin.v1.bak` once), Low 5–8. Optional Low items not done.
- Task 4, answered with evidence in `docs/host-facts-answers.md`:
  1. V4L2 0.1 ms failure: CONFIRMED. 5f78657 requested `Closest(YUYV 1280x720@30)`; host YUYV offers only 640x480/640x360; error chain logged with `{}` hid the detail. HEAD selects MJPG 1280x720@30 (table test on the host format list).
  2. Depth gate: in-process Depth-Anything-V2-Small on ORT CPU (`backend="ort"`, `device="cpu"`); no fallback path. `/var/lib/doorman/models` exists (owner confirmed in ROUND3/HOST-FACTS §8) and holds `depth_anything_v2_small_fp32.onnx` plus `depth.ncnn.*`; which depth path the config selects is answered in `docs/host-facts-answers.md` "Round 3".
  3. ORT: deb build uses `ort/download-binaries` ⇒ statically linked; distro 1.21 and `/usr/lib/doorman/libonnxruntime.so` are not loaded unless the ROCm EP was installed. Startup now logs any `libonnxruntime` mapping from `/proc/self/maps`. Host check: `grep -i onnx /proc/$(pidof doormand)/maps`.
- Correction to round 1: the "ORT bundled" claim stands; the earlier note that the host config has `keep_raw_frames=true` comes from the audit and still applies.

CI on PR #13 @81a9686 (owner-verified): `check`, `ort-typecheck`, `deb` pass — so the default-feature build compiles and tests on GitHub. `shellcheck` failed (selected `.plans/HOST-FACTS.md` as a script) and `python` failed (numpy missing) — both fixed in round 3.

Still unverified: all host checklists (ORT linkage is now confirmed, see "Round 4").

## Round 3 (owner verification, `ROUND3.md`)
Owner decision: stays on one branch / PR #13 (ROUND3 task 8 not applied).
- Task 1: CI shellcheck selects tracked `*.sh` + shebang files, excluding `.plans/`, `docs/`, `*.md`.
- Task 2: python CI job installs numpy; `test_rnd_depth_fusion_experiment.py` uses `importorskip`.
- Task 3: this file corrected (pr/* branches remain — proxy refuses deletion; models_dir exists).
- Task 4: `[liveness] mean_mode_depth_samples` (default 0 = unchanged) depth-checks K evenly sampled frames in mean mode; swap tests in `daemon/src/ipc.rs`. Cost ≈ K × per-frame depth latency — measure with `doorman-bench latency` (command in `docs/decision-and-liveness.md`). Residual: a single photo frame no sample lands on is still averaged in.
- Task 5: `tests/test_packaging_gpu_free.py` (service + postinst).
- Task 6 (`docs/host-facts-answers.md` "Round 3"): `backend="ort", device="cpu"` ⇒ in-process ORT CPU on `depth_anything_v2_small_fp32.onnx`; `depth.ncnn.*` only with `backend="ncnn"`, which runs Vulkan even with `--device cpu` (`--device` overrides only `ml.device`). Which one the host runs: check the `Using ML backend:` startup line. Shipped configs are GPU-free (test `shared/tests/shipped_configs_no_gpu.rs`). **Fixed**: `doorman install` default `--auto` used to pick ROCm/CUDA when present; now CPU on Linux, GPU only via `--rocm`/`--cuda`/`--ncnn`.
- Task 7: `docs/MODELS.md` licence table + used/legacy table. Flag: the deb ships `edgeface_s.onnx` without a verified redistribution licence (likely CC BY-NC-SA; HF card unreadable here) — owner decides. `minifasnet_v2se_int8.onnx` is packaged but unused.

Owner check: `journalctl -u doormand | grep "Using ML backend"` (the ORT maps check is done, see "Round 4").

## Round 4 (owner follow-ups)
CI on PR #13 @e50cd65 (all green; check runs read via the GitHub API):
- `ci.yml` run https://github.com/angkira/doorman/actions/runs/41355386222: `check`, `python`, `shellcheck`, `ort-typecheck`, `deb`.
- `release.yml` run https://github.com/angkira/doorman/actions/runs/41355386218: `deb`.
This replaces the earlier "shellcheck/python fail" statements, which were fixed in round 3.

What was verified by reading the workflows (not by re-running them):
- `ci.yml` `check` runs `cargo test --locked --workspace $FEATURES -- --skip test_video_frame_detection` with `FEATURES=--no-default-features --features camera-mock,camera-ffmpeg,preview`: no ORT backend, and the test is skipped explicitly.
- `ci.yml` `deb` does not run `cargo test` at all (packaging check, `build_deb.sh`, `dpkg-deb` inspection).
- `release.yml` `deb` runs `cargo test --locked -p doormand` with the daemon's default features (`backend-ort`, see `daemon/Cargo.toml` `default`), with no `--skip`. It is therefore the only job that runs `test_video_frame_detection` with default features, as the owner stated. Whether that test passes there depends on `$HOME/.local/share/doorman/models`/frames being absent; the job is green at e50cd65 (the test body tolerates missing frames).
- The `.deb` artifact is uploaded by `release.yml` `deb` step "Upload CI artifact" (`actions/upload-artifact@v4`, name `doorman-deb`, path `target/debian/dist/*.deb`, `if-no-files-found: error`). `ci.yml` builds the .deb but does not upload it. The GitHub-release upload step only runs on `v*` tags.

Host facts (HOST-FACTS section 9): ORT is statically linked in the installed (old, ~5f78657) build (`grep -ci onnx /proc/<pid>/maps` = 0). `doorman config check` does not exist in the installed CLI ("unrecognized subcommand 'config'"); it exists only in the PR #13 build.

Step-by-step host install of this build: [docs/INSTALL-FROM-BRANCH.md](../../docs/INSTALL-FROM-BRANCH.md).

Round 4 code changes (each with tests): `MODELS.md` moved to `docs/`; mean-mode swap tests through the real `run_authenticate` (`daemon/src/ipc/real_path_tests.rs`, test-only DaemonState + scripted backend); `doorman install` keeps an existing config and writes `doorman.toml.new`; postinst warns about GPU grants in operator systemd files (never edits them); `dataset.raw_face_frame_bundle_size` is a deprecated key in `config check`; `doorman-lock` attaches the watcher to a live locker after a non-zero quick exit (pidfile prevents duplicates). All host-unverified.

## Round 3 review (8daf849)
Owner review of PR #13 found three issues; fixed, each with tests, all host-unverified:
- `doorman install` no longer overwrites `/var/lib/doorman/embeddings.bin` with `~/.local/share/doorman/embeddings.bin`: it copies only when the destination is absent (`cp --update=none`), otherwise skips and prints a WARNING with both paths and the deliberate import commands. It also prints what it will change at the very start and needs `yes` or `--yes`; non-interactive without `--yes` is refused (non-zero exit). Dry runs are unaffected.
- `docs/INSTALL-FROM-BRANCH.md`: do not use `doorman install` for this path; second root shell at step 0; timestamped backup of the niri scripts; do-not-purge warning with the exact `postrm` deletions; rollback from a root-owned 0700 tarball (replaces the impossible `apt --allow-downgrades`); rebuilding 5f78657 as a deb is marked untested.
- `doorman-lock`: pidfile check and watcher start run under a non-blocking `flock` on `unlock-watch.lock`, so two presses cannot start two watchers; a pidfile whose PID is alive but not an unlock watcher (reused PID) is removed; without flock the old behaviour is logged and used. `docs/unlock-path.md` updated.

## Known host-test findings (ROUND6, host test of 7e31f6b)
Nothing below is host-verified; the fixes are code and unit tests only. The next host test must confirm them.
- Regression: after `fdf33be` the native (nokhwa) open succeeded on the UGREEN Camera 4K (MJPG 1280x720@30), so the ffmpeg fallback was never reached; the producer thread then ended ~570 ms after "Camera gated ON" with no "Camera ready", "Capture error" or panic text, and `doorman test` said "Camera not available (no frames)".
- Proved by code: the producer thread can only end by the receivers-gone return (receiver was alive) or by an unwinding panic, so it was a panic on the native capture path. NOT proved: the exact panic site (candidates in `docs/camera.md`: v4l `Stream::drop` panic on STREAMOFF errors, `nokhwa::Camera::drop` unwrap, mozjpeg decode). Not reproduced without the host camera.
- Fixed (host-unverified): native backend must decode one real first frame within `[camera] native_first_frame_timeout_ms` (default 1500) or fall through to ffmpeg; `[camera] backend = "auto" | "native" | "ffmpeg"` (default `auto`, `ffmpeg` skips native, the host override if native misbehaves); producer panics are caught, logged at ERROR, and the camera is re-opened on the next request; `doorman test` exits 2 on `!recognized:` (0 only when recognized, 1 on CLI error), stdout unchanged.
- Known limit: a native first-frame read that blocks forever cannot be interrupted (v4l polls without timeout); after the timeout the device stays held until that read returns, so the ffmpeg fallback may find it busy. Set `backend = "ffmpeg"` on such a host.
- Docs: `docs/INSTALL-FROM-BRANCH.md` installs the niri scripts side by side under `~/.local/bin/doorman-event/`, no longer hard-codes a release run id, has jq p50/p95 commands for the timing log; `doorman install --help` says it is a from-source dev installer.

## Round 7 (native "Not a JPEG file" on the owner's host)
Symptom, **proved from the owner's journal**: with UGREEN Camera 4K (uvcvideo, MJPG 1280x720@30) the native open succeeded but the first frame read panicked with `libjpeg fatal error: Not a JPEG file: starts with 0x61 0xe8`; the ffmpeg fallback then worked (first_frame_ms about 1985, recognized 73.76%).

Proved by reading the dependency sources (not by running a camera): nokhwa's V4L2 path returns the full mmap'd buffer and drops `bytesused`; the format read-back after `set_camera_format` already errors on mismatch.

**Hypothesis, not proved:** the cause of the non-JPEG buffer (junk/partial first buffers after STREAMON, or a format/buffer mismatch inside nokhwa or the driver). The new WARN line (first 8 bytes as hex, length, fourcc, resolution, frame index) is there to settle it on the next host run.

Changes (tests in each; host-unverified): negotiated-format check that fails fast; validation of every buffer before the decoder (SOI + marker walk to EOI); `native_warmup_frames` (default 3) with retry inside `native_first_frame_timeout_ms`; per-(device, mode) negative cache `native_failure_cooldown_secs` (default 600) for `backend = "auto"`. No auth thresholds, liveness defaults or other defaults changed. Nothing here is claimed as verified on the host.

## Round 8 (time-to-unlock; accuracy and security outrank speed)
Goal: p50 4.3 s -> as low as possible WITHOUT touching the decision function. No threshold, liveness default, `auth_frames` or other existing default changed; two new `[ipc]` keys only.

### Proved (from the owner's journals read on 2026-11-10, and the niri 26.04 source)
- `LockedHint` is set by **niri**, not gtklock, the script or doorman: the niri binary contains the "Logind LockedHint Updater" thread (`GetSession` + `SetLockedHint`), and in `src/niri.rs` (v26.04) `update_locked_hint()` sets it only in `LockState::Locked`, i.e. after the lock client mapped a surface on every output and niri redrew (`lock()` -> `WaitingForSurfaces` -> `Locking` -> `Locked`; the 1 s timer is only a fallback).
- 22 locks in the journal: WaitForMatch request -> niri "locking session" 0.34-0.73 s (gtklock start-up); niri "locking session" -> armed 0.50-0.91 s (niri surfaces + redraw + hint thread); request -> armed 0.87-1.49 s (median about 1.1 s).
- The daemon is already event-driven on the hint (`lock_supervisor.rs`: D-Bus `PropertiesChanged`/`Lock` signal streams into a `watch` channel; `unlock_wait` awaits it). Nothing is polled on the daemon side, so there is no daemon-side lag to remove (task 3 needs no change). The 200 ms tick in `main.rs` only re-evaluates the camera gate and is also woken by the channels.
- Before this round the camera was opened only after the gate armed (`AuthenticateSource` ran only when armed); the host log shows "Authentication request" in the same millisecond as `armed=true` and "Camera gated ON" 0.84 s later.
- Auth loop: all 10 frames are processed (frames_seen = frames_used = 10 in the journal) at about 168 ms per frame, so the loop is inference-bound, not capture-bound (30 fps capture would be 33 ms per frame).

### Inferred / not measured
- The split of the 168 ms per frame between detection, embedding and (2 of the 10 frames) depth: no host timing log was enabled (`/run/doorman/timing.jsonl` absent), so it is NOT profiled. Enable `[telemetry] timing_log = true` for the next run.
- Why niri takes 0.5-0.9 s to reach `Locked` (gtklock surface mapping vs. vblank redraw): not determined.

### Implemented (host-unverified)
1. Pre-arm in `unlock_wait.rs` (`[ipc] prearm_camera = true`, `prearm_max_hold_secs = 5`): the decision runs while the gate is closed, `Matched` only after armed, frames older than the request are skipped, lease and result dropped after the hold. Unit tests with a fake source and paused clock; real-path test for frame freshness.
2. `doorman-lock` starts the watcher before the 0.3 s settle wait (`DOORMAN_WATCH_EARLY`, default 1; 0 restores the old order). flock/pidfile protections and exit codes unchanged; an early watcher for a refused locker is reaped before the old handling.
3. Telemetry: `hint_wait_ms`, `lock_to_camera_ready_ms`, `frames_collect_ms` in the timing record; `WaitForMatch timing` info line (prearm, hint_wait_ms, total_ms).

### Considered and dropped (not provably neutral for the decision)
- `early_accept_after_frames` (accepting on fewer than `auth_frames`): changes the decision; no safety argument, not added.
- Skipping the depth model on frames, or running inference on several frames in parallel: would change which frames/orderings the liveness and mean decision see; not argued neutral, not done. The loop is inference-bound, so overlapping capture with inference cannot help either.
- Lowering `native_warmup_frames` / the 0.6 s camera warm-up: an existing key the owner may experiment with; default untouched.

### Expected latency model (to be checked, not a host claim)
Before: T = script 0.45 + hint lag 1.1 + camera 0.9 + recognition 1.7 + unlock 0.1 = about 4.3 s.
After: lock 0.12 + max(hint lag, camera 0.9 + recognition 1.7). The recognition runs from camera-ready, in parallel with the remaining lag; the match is ready at about request + 2.6 s and delivered at max(2.6 s, hint), so T about 0.15 + 2.6 + 0.1 = about 2.9 s (saves about 1.4 s). The 2 s target is NOT reached by this round alone: the remainder is camera open + warm-up (0.9) and recognition (1.7), neither reducible without touching the camera policy or the decision.
The host re-measure must show: `hint_wait_ms` about 1100, `lock_to_camera_ready_ms` negative (camera ready before the hint), `camera_open_ms` about 900, no `Matched` before `armed=true` in the journal, "Camera gated OFF" within about 0.2 s of unlock, and no camera open for more than `prearm_max_hold_secs` when the lock fails.

### Security impact (every behavioural change)
- Pre-arm runs the same `run_authenticate` (quality gate, depth liveness, aggregated threshold) with unchanged config; only its start time moves earlier. A false accept needs exactly the frames and scores it needed before.
- `Matched` is delivered only when `armed` (LockedHint && !IdleHint) is observed true at delivery. While closed the function can only return `NoMatch`/hold; a test asserts no `Matched` with a gate that never opens. The unlock helper therefore still signals the locker only after niri is fully `Locked`.
- Freshness: WaitForMatch attempts now skip frames captured before the request (stricter than before, where an already-open camera could hand over a buffered frame). Held results are at most `prearm_max_hold_secs` old (default 5 s) and are discarded after that.
- A frame captured while the screen was still locking can count; it is a frame of whoever stood in front of the camera after the request, which is the same population as frames after the hint. No new impersonation route; a photo faces the same depth check.
- Camera exposure: with pre-arm the camera can run up to 5 s while the session is unlocked, only on a WaitForMatch from the user's own helper (same peer-credential policy and rate limit as `Authenticate`, which can already open the camera). `prearm_camera = false` removes it.
- Script: starting the watcher earlier does not change what it signals (same PID and comm checks). A watcher for a refused locker exits on its own and is reaped.
- Not changed: thresholds, liveness, `auth_frames`, `mean`/`n_of_m` rules.

### Weaknesses noticed in the auth path (NOT fixed here; Round 9)
- `mean` mode with `mean_mode_depth_samples = 0` (default): depth is checked on the first quality frame and the deciding frame only, so middle photo frames plus live first/deciding frames are accepted (pinned by `mean_mode_without_samples_still_averages_in_unseen_middle_photos`).
- Depth threshold 0.05 produced a genuine false reject at 0.0491 (owner journal): the margin is thin; calibration needed (accuracy, not only security).
- The default `Authenticate` (PAM, non-sudo) path still starts from the buffered latest frame without a freshness floor; only the strict sudo profile and, since this round, WaitForMatch have one.
- `WaitForMatch` retries attempts for up to the whole call (30-60 s) with no failure counter or backoff; the rate limit counts the call once.
- `DaemonState` carries `unsafe impl Send/Sync`.

## Round 9a (security hardening that does not need the owner's data)
Scope: controls that close proved code-level gaps without lowering any threshold. Round 9b (below) needs the owner's measurements. Unchanged on purpose: `similarity_threshold`, `liveness_depth_threshold` (0.05), `auth_frames` (10), `mean_mode_depth_samples`, N-of-M defaults. Commits are one per item, except B+C+D which share the per-frame loop and the config surface (the config keys of B-D are introduced in the commit of A).

### PROVED (code read at 16ed392 and tests, nothing run on a camera or on the host)
- Lock path had no attempt budget: `wait_for_match` re-ran the full decision back to back until the deadline; a source that fails instantly (camera unavailable) made it a busy loop. Now: shared budget in `run_authenticate` + 250 ms pause (`unlock_wait::RETRY_PAUSE`).
- `mean` mode scored every frame into the aggregate with no per-frame identity test, and liveness was only checked on the first and deciding frame: the live-other-face + owner-photos mix was ACCEPTED (test `identity_floor_rejects_a_live_attacker_around_owner_photos` pins the old acceptance with the floor off, and the rejection with it on).
- The detector kept one face per frame (highest score) and discarded the rest; only an empty set was checked before deciding (a single-frame mean was possible).
- No model hash was checked at load; `doorman models verify` did not exist (docs/MODELS.md only mentioned `doorman status`).
- Matched raw frames were stored whenever `keep_raw_frames` was on (`log_auth_capture_pack` is called for ConfirmedGenuine); `retention_days` never deleted anything.
- `DOORMAN_WATCH_EARLY=1` started the poll watcher before the settle wait (verifier finding); fixed, test fails without the fix.

### INFERRED / not measured
- Effect on FRR: the floor (0.30 vs genuine frames far above it), min frames (8 of 10) and the multi-face gate are argued from the host-observed genuine range and the detector rate, not measured. Watch `identity_floor`, `insufficient_frames`, `ambiguous_faces` in the journal after the rollout.
- Per-frame genuine cosine distribution is unknown (the host range 0.74-0.82 is the aggregate); the floor 0.30 is justified as "well below the match threshold and well above where impostor frames sit (about 0)", not fitted.
- That `max_frame_age_ms = 1000` never trims a genuine frame on the host (the frame is taken up within about one frame period of capture).
- The owner's installed model files match `packaging/models.sha256` (they come from `fetch_models.sh` pins if installed from the .deb; a locally exported recognizer would not). Check with `doorman models verify` before upgrading.

### Implemented
| Item | What | Where |
|---|---|---|
| A | Failure budget + cooldown 30 s / 2 min / 10 min for ALL entry points; `NoMatch{suspended}` and `Denied [face_auth_suspended]`; watcher backs off 5 s; WaitForMatch attempts spaced 250 ms | `daemon/src/face_budget.rs`, `ipc.rs` `run_authenticate`, `unlock_wait.rs`, `unlock_watch.rs`, `[ipc] face_*` |
| B | `[decision] frame_identity_floor = 0.30`, fail closed (`identity_floor`) | `frame_policy.rs`, `ipc.rs` |
| C | `[recognition] multi_face_policy = reject|largest|off` (+ ratio 0.25, score 0.75); detector reports `Face.other_faces` | `ml/backend.rs`, `ml/ort_backend.rs`, `ipc.rs` |
| D | `[authentication] min_decision_frames` (0 = auth_frames - 2), `[camera] max_frame_age_ms = 1000` | `frame_policy.rs`, `ipc.rs` |
| E | SHA-256 manifest compiled in, checked at start for the loaded models, fail closed; `[models] allow_unverified_models`, `manifest_path`; `doorman models verify [--print-manifest]` | `model_integrity.rs`, `packaging/models.sha256`, `main.rs` |
| F | `[dataset] capture_successful_unlocks = false`, `raw_frames_retention_days` (unset = never delete), dirs forced to 0700, INFO line with the retention state | `dataset.rs`, `dataset_purge.rs`, `main.rs` |
| G | `doorman --json test` structured record, exit codes 0/1/2 documented | `test_report.rs`, `bin/doorman.rs` |
| H | poll mode never starts the watcher early; flap and hold==hint tests; docs for camera-on-unlocked and request-relative freshness | `doorman-lock`, `unlock_wait.rs`, `docs/unlock-path.md` |

### Security impact (every behavioural change)
| Change | FAR | FRR | Latency | Fail-closed paths |
|---|---|---|---|---|
| A budget | lowers the achievable attack rate: k attempts become at most 5 per 5 min, then 30 s / 2 min / 10 min lockouts | none unless 5 judged failures in a row (the owner in a dark room: then the password is right) | 0 on success; 250 ms pause between failed retries inside a call | suspended -> Denied/NoMatch, no camera opened |
| A DoS | n/a | an attacker who fails on purpose switches the owner's face unlock off for the cooldown (password and a later match unaffected) | n/a | documented |
| B floor | closes the live-face + photo mix; does not stop a look-alike above 0.30 | none expected (genuine frames far above 0.30); a genuinely bad frame (blur, head turn) now fails the attempt, the next WaitForMatch attempt retries | about 0 (a dot product) | below floor or NaN -> reject |
| C multi-face | closes the phone-next-to-face alternation | a close bystander (>= 25% of the face area, score >= 0.75) causes a reject -> password | 0 (detections already computed) | ambiguous -> reject |
| D min frames | removes the 1-frame decision under load | rises only if fewer than 8 of 10 good frames arrive in the 3 s window (below about 3 fps); retry follows | 0 | too few -> reject (not a judged failure) |
| D frame age | stops a stalled feed from deciding | none expected | 0 | stale -> skipped, none usable -> `no_frames` |
| E manifest | blocks swapped detector/recognizer/liveness model | a host whose model files differ from the pins will not start the daemon until fixed or opted out | start-up hashing (about 0.1-0.5 s for 100-200 MB) | mismatch -> daemon exits non-zero |
| F | removes sharp owner photos of successful unlocks from disk by default | none | 0 | retention only deletes when configured |
| G, H | none | none | none | none |

Upgrade effects on the owner's host (explicit keys needed to keep the old behaviour): `capture_successful_unlocks = true` if genuine frames are still wanted for calibration; `doorman models verify` first.

### Mutation checks (each safety property, on the dev clone, reverted after)
See `$HOME/.cache/doorman-verify/round9a-report.md` for the full table. Every control above has a test that fails when the check is removed.

### Round 9b backlog (needs the owner's measurements; nothing below is enabled)
- Depth on several frames: `mean_mode_depth_samples` 0 -> 2 and a median over 3 spread frames (the depth threshold has a thin genuine margin: 0.0491 vs 0.05 observed).
- Second PAD model (MiniFASNet) in shadow mode, with a fail-closed wrapper (the legacy `check_liveness` returns `Ok(true)` when the model is missing).
- Quality gates (face-region sharpness, yaw proxy, inner-face brightness) in log-only mode first; restore `min_face_area_frac` from the host's 0.005.
- Threshold calibration with the owner's attack protocol (print, screen, curved print, cut-out, mixed track) and the production scorer; FAR at 0.4 is still unmeasured.
- `frames_judged` and the margin in the wire answer; `doorman stats`; lock-generation-bound `Matched`.
- Per-frame genuine cosine distribution from the timing log to tune the floor (0.30) and `min_decision_frames`.

### Not done / limits
- Honest limits: RGB-only camera, monocular-depth heuristic as the only PAD; a convenience factor, not a strong biometric.
- The depth model has no pinned hash in the repository (unlisted unless the owner adds it with `manifest_path`).
- Host verification of all of this is pending (no daemon, camera or `doorman test` was run).
