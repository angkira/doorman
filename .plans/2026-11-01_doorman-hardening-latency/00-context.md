# 00 — Context, measurements, open questions

Everything here comes from a read-only audit of a real install on 2026-11-01 plus the repo at HEAD (`7c511d9` + local commits for niri/sway packaging). File:line refs may have drifted — re-verify before editing.

## Host (for realism of benchmarks)
AMD Ryzen 7 8700G (Zen4, 8c/16t, Phoenix1 iGPU Radeon 780M). Linux 6.17. Wayland compositor niri + DankMaterialShell. Locker: gtklock 4.0.0 (swaylock fallback). Camera: UGREEN 4K UVC webcam, **RGB only** (`/dev/video0`, `/dev/video1`). Display manager SDDM (doorman currently NOT wired into SDDM PAM; present in swaylock and kde PAM files). Inference currently `--device cpu`. A discrete GPU exists but is reserved for ML training and must never be used by the daemon.

## Pipeline as-is
| Stage | Implementation |
|---|---|
| Capture | producer thread `daemon/src/pipeline/camera_producer.rs`; config 1280x720@30, effective ≈15 fps. Backend order GStreamer → V4L2/nokhwa → ffmpeg CLI → mock (`daemon/src/camera/mod.rs` ~105–149). |
| V4L2/nokhwa | **fails on every open** within ~0.1 ms: `V4L2/nokhwa camera failed: Failed to open camera, trying next backend`. Cause unknown **[unverified]**; ffmpeg opens the same device fine so it is not permissions. |
| ffmpeg fallback | spawns `ffmpeg -f v4l2 … -pix_fmt rgb24 -f rawvideo pipe:1`; ~2.7 MB/frame over a pipe; **≈1.3 s from spawn to first frame**, every lock. |
| Camera gating | camera closed unless `armed` (= `LockedHint && !IdleHint`, `lock_supervisor.rs`, `camera_producer.rs` ~62–72) or capture lease. |
| Detection | YuNet 2023mar ONNX (232 KB), fixed 640x640 input, ≈7.6 fps. |
| Align/quality | 5-pt similarity → 112x112 (`ml/align.rs`); gate `min_sharpness=8.0`, `min_face_area_frac=0.005` (`ipc.rs` ~578). |
| Embedding | EdgeFace-S ONNX (14.8 MB), 512-d. |
| Liveness | Depth-Anything-V2-Small fp32 at 518x518 on CPU (`ort_backend.rs` ~1000, `model_config.rs` ~150). Metric: `std(face_depth)/(depth_max-depth_min+eps)`; threshold **0.05** (`shared/src/config.rs` ~446). Runs once per attempt on the **first** quality frame. MiniFASNet files are installed but not the active gate. |
| Matching | `auth_frames=10` quality embeddings, mean, renormalise, cosine ≥ **0.4** vs stored gallery (`ipc.rs` ~268–440). `timeout_secs=3`; `acquire_camera` waits up to 8 s. |
| ORT | `ort 2.0.0-rc.12`, CPU provider, `cpu_threads=0` (auto). `doorman-depthd` sidecar exists but is not running. |
| IPC | Unix socket `/run/doorman/doorman.sock`, mode **0666** (`ipc.rs` ~74), one JSON line per request (`ipc.rs` ~94), `Authenticate{username}`. **No SO_PEERCRED / uid check.** `Enroll`/`RemoveUser` on the same socket, same lack of checks. |
| PAM | `pam_module/src/lib.rs` (~173 lines) thin client: Success→PAM_SUCCESS, Failure→PAM_AUTH_ERR, daemon down/timeout/root/no username→PAM_AUTHINFO_UNAVAIL. |
| Dormant unlock | `lock_supervisor.rs` can call logind `Session.Unlock()` (`proactive_unlock`), disabled in host config (`proactive_unlock=false`, "Sway adapter owns face unlock"). |

## Unlock path as-is (lives OUTSIDE the repo on the host, in `~/.local/bin/`)
- `doorman-lock` (niri `Mod+L`): starts `gtklock --daemonize --config ~/.config/gtklock/config.ini` (swaylock fallback), logs to `/tmp/doorman-lock.log`, then launches `doorman-lock-watch`. 2026-11-01 patch (host-only): watcher now starts if rc==0 **or** a gtklock process is alive (old code started it only on rc==0 → watcher never ran → no face unlock) and extra pre-lock logging.
- `doorman-lock-watch`: loops while a process named `gtklock` exists: runs `doorman --socket /run/doorman/doorman.sock test <user>` (a **new full capture session per call**), on output starting with `recognized:` runs `pkill -TERM -x gtklock` (then KILL after 0.5 s), `sleep 2` between attempts.
- gtklock 4.0.0 facts (from binary analysis): handles **only SIGTERM** (graceful `g_application_quit`); **SIGUSR1 is not handled** (default action = terminate). `gtklock --daemonize` returned rc=1 while the lock was actually held: niri logs `refusing lock as already locked with an active client` ~1 s after `locking session` **[hypothesis: the daemonized child locks and the foreground parent's second request is refused]**.
- gtklock's PAM service `/etc/pam.d/gtklock` is `auth include login` — it does not use pam_doorman.

## Measured timeline, lock at 13:12 local (all numbers from journal/logs)
| Phase | Δ |
|---|---|
| lock start → armed | ~0.2 s |
| ffmpeg spawn → first frame (cold camera) | **1.3 s** |
| attempt 1: liveness rejected genuine user (`relief=0.0491 < 0.0500`; calibration range for this user 0.068–0.245) | **~1.3 s lost** |
| watcher `sleep 2` | **2.0 s** |
| attempt 2: ~10 quality frames (≈7.6 fps) + depth pass | 1.25 s |
| recognised → process exit → pkill → locker gone | ~0.5 s |
| **Total** | **≈7.5 s** (≈3.6 s if liveness passes first time) |
At 11:03 an earlier run recognised the user 1.65 s after arming; that run predates the `proactive_unlock=false` change.

## Other findings
- Installed `doormand` (built 2026-06-24 from ~`5f78657`) is **older than HEAD**: it lacks commits `210776c` ("Capture auth face frames before recognition") and `7c511d9` ("Decouple auth decision frames from raw capture") which change `ipc.rs`, `shared/src/config.rs`. dpkg marks `/etc/doorman/doorman.toml` as an *obsolete conffile* → check the config schema before any reinstall.
- Dataset logging defaults are privacy-heavy: `keep_raw_frames=true`, `raw_face_frame_bundle_size=120`, `max_capture_packs_per_day=500` → face frames stored under `/var/lib/doorman/dataset`.
- Stale `/usr/local/bin/doormand` (Nov 2025) and ~20 `/etc/doorman/*.bak*` files on the host.
- Depth model is not shipped in `/usr/share/doorman/models` (only in an unreadable `/var/lib/doorman/models`); MiniFASNet files shipped but unused.
- Nothing tests the watcher script, the lock/unlock flow, socket permissions, latency, or spoofing.

## Research digest (see `06-bench-harness.md` for the experiment matrix)
Treat as leads, not facts; sources were thin and several numbers are from different hardware.
- YuNet `2026may` ONNX exists with dynamic H/W (same weights as 2023mar) → allows a smaller detector input.
- EdgeFace-XS reported 25.9 ms ONNX FP32 / 7.0 ms INT8 (hardware unknown); a study saw transpose ops cost ~3.5 of 7.4 ms. No official EdgeFace ONNX (third-party exports only). License of weights possibly non-commercial — personal use here.
- Depth-Anything-V2-Small ONNX (fp32 and w8a16) published by Qualcomm; fixed 518x518 input.
- AMD's Linux NPU stack starts at Strix Point; Phoenix is not listed → do not plan on NPU. ROCm/MIGraphX on gfx1103 unverified; for models this small CPU is expected to win.
- Howdy-style tips: discard 3–10 warm-up frames, brightness gate, lower capture height (320), several distinct enrolment templates, prune near duplicates.
- RGB-only presentation-attack detection is generally weak cross-domain; for `sudo`, face must not be the only factor by default (WS7).

## Open questions the executor must answer in code/PRs
1. Why does V4L2/nokhwa fail to open the UVC device (format negotiation? MJPEG only? exclusive access? nokhwa version)? (WS4)
2. Does gtklock (or niri) react to logind `Session.Unlock()`? Likely **no** → treat logind unlock as exploratory only (WS2).
3. Where does time go inside the 1.25 s "attempt 2" — detection vs embedding vs depth? Add per-stage timing logs at debug level. (WS6)
4. Is there any state in `ipc.rs` that blocks concurrent Authenticate requests (matters for a subscription design)? (WS2)
