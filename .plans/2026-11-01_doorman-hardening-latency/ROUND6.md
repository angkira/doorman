# ROUND 6 — camera regression found in host test of 7e31f6b (blocks merge of PR #13)
Fix on branch claude/great-ride-i3oqn6 (same branch, no new branches). Never push to main, no tags/releases/version bumps. Do not change auth thresholds/liveness defaults.

## Host facts (owner-verified; build = .deb from release run 41380333708, head 7e31f6b)
- UGREEN Camera 4K, MJPG 1280x720@30 offered. Permissions are NOT the cause (unit allows char-video4linux, user in `video`, /dev/video0 root:video 660, ffmpeg present).
- Journal after restart: "V4L2 camera opened" + "Camera gated ON" at 11:02:03.237; 570 ms later "Preview supervisor stopped (camera channel closed)" (main.rs:238 => producer thread/sender gone). No "Camera ready", no "Capture error", no panic text. Later requests log nothing about the camera. `doorman test` => "Camera not available (no frames)".
- Before the branch: the nokhwa open failed in ~0.1-0.4 ms on this host and the daemon fell back to ffmpeg (1280x720@30), which worked ("Captured 10 frames").
- Analysis: commit fdf33be "Negotiate native camera format" makes the nokhwa open succeed, so the ffmpeg fallback is never reached and the nokhwa MJPEG capture path (daemon/src/camera/v4l2_backend.rs) runs on this host for the first time; its producer thread dies.
- Side finding (also on main): `doorman test` prints `!recognized: <user> (<reason>)` and exits 0 on Response::Failure. Existing watchers anchor on `^recognized:` so they are safe today, but exit-code-only consumers would be fooled.

## Tasks (all with tests; fmt, clippy -D warnings, cargo test --workspace, shellcheck, pytest)
1. Validate a real first frame (short timeout as a named config/const, no magic numbers) before declaring the nokhwa backend usable; on failure/timeout/panic fall through to ffmpeg exactly as before. Test with a fake backend that opens OK but yields no frame / errors / panics.
2. Producer thread: catch_unwind, log panic reason and thread-exit cause at ERROR, never exit silently; if it dies mid-session the handle reports unavailable AND the next request re-opens / falls back to ffmpeg instead of staying dead until daemon restart. Test.
3. Find and fix the actual failing call (camera.frame() / MJPEG decode / buffers / format negotiation) if derivable from code; state clearly what is found vs. guessed; do not claim host verification.
4. `doorman test` must exit non-zero on Failure/CameraUnavailable, stdout format unchanged (`^recognized:` greps keep working). Audit every script in packaging/ and scripts/ that calls `doorman test` for exit-code assumptions; update + test. If this is too invasive, put the question at the top of the PR.
5. Config-driven camera backend: key default `auto` (nokhwa with validated first frame, then ffmpeg) with explicit `backend = "ffmpeg"` override. Document in docs/CONFIG.md.
6. Remaining doc items in docs/INSTALL-FROM-BRANCH.md: stale run id/commit (~line 130); step 10 `install -D` over ~/.local/bin/doorman-lock -> make side-by-side under ~/.local/bin/doorman-event/; add jq p50/p95 commands for /run/doorman/timing.jsonl; `doorman install --help` must say it is a from-source dev installer, not for upgrading a packaged install.
7. Update STATUS.md ("Known host-test findings"). Report head SHA + CI run ids.
