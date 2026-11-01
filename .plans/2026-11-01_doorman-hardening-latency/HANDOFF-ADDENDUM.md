# HANDOFF ADDENDUM (owner update 2026-11-01)

1. Read `HOST-FACTS.md` and `CI-AND-BRANCH-FINDINGS.md` first. They supersede assumptions in `00-context.md` where they conflict.

2. Process correction: the work so far sits on ONE branch (`claude/great-ride-i3oqn6`, ~45 commits, no PRs, WS8 merged into it). Re-split into one PR per workstream (`ws<N>-<slug>`) against `main`, or at minimum one PR per logical group (IPC peer-cred + rate limit; bench + telemetry; event-driven unlock; camera latency; packaging/hygiene), each independently green. Do not push to main directly.

3. PR order:
   - First PR (hotfix, tiny): pin ort `=2.0.0-rc.12` and track `Cargo.lock`, so main's CI goes green.
   - Then a separate PR for `cargo fmt --all` (formatting only, no logic).
   - Then a separate PR for the clippy fixes (a deliberate `#[allow]` only with a justification comment).
   - Only then the feature PRs, so CI is meaningful.

4. Host facts that change the work:
   - (a) GPU: the installed unit grants GPU devices: `DeviceAllow=char-drm`, `/dev/dri`, `/dev/kfd`, plus drop-in `10-drm.conf`. The daemon must never use the discrete GPU (the owner reserves it for ML training). The branch removed only `/dev/dri`; also remove `char-drm` and `/dev/kfd` from the shipped unit, and document a drop-in example for opt-in iGPU use. The unit header says `EP: ncnn` and comments mention Vulkan/radv: determine from code which backend the depth/liveness model uses in the installed build and report.
   - (b) Models: config `models_dir=/var/lib/doorman/models` does not exist on the host. Models are in `/usr/share/doorman/models` (edgeface_s, face_detection_yunet_2023mar, minifasnet_v2se (+int8)). No Depth-Anything model is present there, yet the log says `Liveness REJECTED (depth gate)`. `00-context.md` claims Depth-Anything-V2-Small 518x518: treat that as UNVERIFIED and resolve from code/git history of the installed build (~5f78657) which model produces the depth gate.
   - (c) Camera: MJPG offers 640x480/1280x720/1920x1080 @30fps (720p/1080p/360p also 60). YUYV offers only 640x480 and 640x360 @30fps (1280x720 is NOT offered in YUYV). The V4L2/nokhwa open fails in ~0.1-0.4 ms on every start with no error detail, then ffmpeg 1280x720@30 is used. HYPOTHESIS to test (unverified): nokhwa is asked for a format/resolution the device does not offer (e.g. YUYV 1280x720). Log the full error and request MJPEG explicitly.
   - (d) ONNX Runtime: the actually loaded ORT is unknown (distro libonnxruntime 1.21 vs `/usr/lib/doorman/libonnxruntime.so` with a 259 MB ROCm provider). doormand has no NEEDED entry for ORT, so it is dlopen'd or static. Add startup logging of the loaded ORT path/version.
   - (e) Dataset logging is on with `keep_raw_frames=true` on the host. New-install defaults must flip, but existing installs must not lose data silently.
   - (f) Numbers: the liveness false reject was relief=0.0491 vs threshold 0.0500 on the genuine owner. Baseline per-request time is 3.0-3.4 s, of which 1.3-1.5 s is camera start. Per-frame similarity is 0.75-0.82 vs threshold 0.4.

5. Keep: no push to main, no tags/releases, no version bumps. Host-only checks go into each PR's "Host validation checklist".

6. Update `STATUS.md` at the end, listing PR links and which of the above questions were answered with evidence.
