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
