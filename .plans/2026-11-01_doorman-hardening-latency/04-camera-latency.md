# WS4 — Camera: warm start, working native capture, lower resolution (P1, effort M)

## Problem
- Every lock spawns an `ffmpeg` subprocess: ~1.3 s to first frame, 2.7 MB/frame of raw RGB over a pipe at 1280x720, effective ≈15 fps instead of 30.
- The native V4L2/nokhwa backend fails to open the device on every attempt (≈0.1 ms, `Failed to open camera`), so the slow fallback is always used. Root cause unknown.
- Camera is gated (closed) unless armed, so every unlock pays a cold open plus the camera's auto-exposure settling.

## Tasks
1. **Root-cause the V4L2 open failure** (reproduce with `v4l2-ctl --list-formats-ext -d /dev/video0` knowledge in tests/docs; read nokhwa error details; check requested format/resolution vs what the UVC device offers — typical culprit: requesting RGB/YUYV at 1280x720 where the device only offers MJPEG at that size, or nokhwa picking the metadata node `/dev/video1`). Fix by negotiating a format the device actually offers. Document the offered formats table you find in the PR (the human will paste real `v4l2-ctl` output from the host).
2. **Prefer MJPEG at a lower resolution** (config `camera.width/height/fourcc`, default 640x480 MJPEG when available; decode with `turbojpeg`/`zune-jpeg`/`image` crate — measure). Face-unlock distance faces are large; detection is resized to 640x640 anyway. Keep 1280x720 selectable.
3. **Warm start**: begin opening the camera on the lock *event* (before/while the locker draws), not after `armed` settles. Keep gating (privacy: LED off when unlocked), but overlap camera open with locker start. Add config `camera.prearm_on_lock = true`.
4. **Frame skipping for exposure**: discard the first N frames (config `camera.warmup_frames`, default 3, bound 0–10) and optionally gate on mean brightness/sharpness before running detection ("frame_ready" predicate); never discard when the stream is already warm.
5. **Optional**: lock exposure/white-balance via V4L2 controls after settle to stabilise embeddings (behind a flag; some UVC cams reset it, so reapply per session).
6. **Backend selection log**: one info line at arm time stating chosen backend, resolution, fourcc, and timestamps: `open_ms`, `first_frame_ms`, `first_valid_ms`. This feeds WS6.
7. Fallback behaviour: if native open fails, fall back to ffmpeg with the same lower resolution/MJPEG input args (`-input_format mjpeg -video_size 640x480`) — cheap win even if the native fix slips.
8. Keep the capture thread's lifecycle explicit (no leaked ffmpeg children on disarm/crash; add test that disarm terminates the child).

## Tests
- Unit/integration with the mock camera backend: warm-up frame discard, prearm timing, disarm closes device, fallback ordering.
- Backend-selection test with a fake "device capabilities" struct (no real device).
- Process-leak test for the ffmpeg backend using a stub `ffmpeg` script on PATH.

## Acceptance
- On the host (human-run): `camera open → first valid frame` p50 ≤ 0.5 s warm-prearm / ≤ 0.8 s cold, measured from the new log line, vs 1.3 s now. State the measured numbers in the PR checklist result when the human provides them.
- No camera LED activity while unlocked/idle (privacy invariant preserved; test the gating logic).
- Detection recall not worse at 640x480 than at 1280x720 on the WS6 genuine set.

## Host validation checklist (human)
Paste `v4l2-ctl --list-devices` and `v4l2-ctl -d /dev/video0 --list-formats-ext`; run 20 locks, collect `open_ms/first_frame_ms/first_valid_ms` lines; check nothing keeps `/dev/video0` open after unlock (`fuser /dev/video0`).
